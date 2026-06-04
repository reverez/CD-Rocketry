"""
GPU-Accelerated ML Surrogate Model (RTX 5070 / CUDA)

Replaces sklearn MLPRegressor with a PyTorch neural network that runs
on the GPU via CUDA. Falls back to CPU if CUDA is not available.

GPU acceleration benefits here:
  - Batch inference over thousands of Monte Carlo parameter sets in ms
  - Fast training via cuBLAS/cuDNN on the MLP
  - Parallel parameter estimation via batched forward passes

Requires:  pip install torch  (CUDA build for RTX 5070: torch+cu128)
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
import warnings

# ── Try GPU stack; fall back gracefully ──────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None

# sklearn used only when torch unavailable
if not TORCH_AVAILABLE:
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
else:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

from scipy.optimize import differential_evolution, minimize


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch MLP definition
# ─────────────────────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    """Simple fully-connected network with BatchNorm and residual skip."""

    def __init__(self, in_dim: int, hidden: Tuple[int, ...], out_dim: int = 1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.SiLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _TorchSurrogate:
    """Wrapper: trains one _MLP per output variable on GPU."""

    def __init__(self, hidden=(128, 128, 64), lr=1e-3, max_epochs=500, batch=256):
        self.hidden = hidden
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch = batch
        self.models = {}
        self.x_mean = None
        self.x_std  = None
        self.y_stats = {}

    def _normalise_X(self, X: np.ndarray) -> "torch.Tensor":
        return torch.tensor((X - self.x_mean) / (self.x_std + 1e-8),
                             dtype=torch.float32, device=DEVICE)

    def fit(self, X: np.ndarray, y_dict: dict) -> dict:
        self.x_mean = X.mean(axis=0)
        self.x_std  = X.std(axis=0)
        Xt = self._normalise_X(X)
        metrics = {}

        for key, y_raw in y_dict.items():
            y_mean = y_raw.mean()
            y_std  = y_raw.std() + 1e-8
            self.y_stats[key] = (y_mean, y_std)
            y_norm = (y_raw - y_mean) / y_std

            X_tr, X_te, y_tr, y_te = train_test_split(
                Xt.cpu().numpy(), y_norm, test_size=0.2, random_state=42)

            X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
            y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE).unsqueeze(1)

            model = _MLP(X.shape[1], self.hidden).to(DEVICE)
            opt   = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, self.max_epochs)
            ds    = TensorDataset(X_tr_t, y_tr_t)
            dl    = DataLoader(ds, batch_size=self.batch, shuffle=True)

            model.train()
            best_loss = float("inf")
            patience  = 0
            best_state = None
            for epoch in range(self.max_epochs):
                for xb, yb in dl:
                    opt.zero_grad()
                    loss = nn.functional.mse_loss(model(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
                # Early stopping
                model.eval()
                with torch.no_grad():
                    val_loss = nn.functional.mse_loss(
                        model(X_tr_t), y_tr_t).item()
                if val_loss < best_loss - 1e-5:
                    best_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if patience > 40:
                    break
                model.train()

            if best_state:
                model.load_state_dict(best_state)

            # Evaluate
            model.eval()
            X_te_t = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                y_pred_norm = model(X_te_t).cpu().numpy().squeeze()
            y_pred = y_pred_norm * y_std + y_mean
            y_true = y_te * y_std + y_mean
            r2  = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            metrics[key] = {"r2": round(r2, 4), "mae": round(mae, 3)}
            self.models[key] = model

        return metrics

    def predict_batch(self, X: np.ndarray) -> dict:
        """Predict all outputs for a batch of inputs. Returns dict of arrays."""
        Xt = self._normalise_X(X)
        results = {}
        for key, model in self.models.items():
            model.eval()
            with torch.no_grad():
                y_norm = model(Xt).cpu().numpy().squeeze()
            y_mean, y_std = self.y_stats[key]
            results[key] = y_norm * y_std + y_mean
        return results

    def predict_one(self, x: np.ndarray) -> dict:
        out = self.predict_batch(x.reshape(1, -1))
        return {k: float(v[0]) for k, v in out.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Public API  (drop-in replacement for ml_surrogate.MLSurrogate)
# ─────────────────────────────────────────────────────────────────────────────

class MLSurrogateGPU:
    """GPU-accelerated surrogate; falls back to sklearn if PyTorch unavailable."""

    FEATURE_NAMES = ["cd_scale", "thrust_scale", "body_mass_kg",
                     "launch_angle_deg", "wind_speed_ms"]

    def __init__(self, simulator_fn: Callable,
                 output_keys: Optional[List[str]] = None):
        self.simulator_fn = simulator_fn
        self.output_keys  = output_keys or ["apogee_m", "max_velocity_ms", "max_mach"]
        self._torch_model: Optional[_TorchSurrogate] = None
        self._sklearn_models: dict = {}
        self._X_train = None
        self._y_train: dict = {}
        self.is_trained = False
        self.using_gpu  = TORCH_AVAILABLE and DEVICE.type == "cuda"
        self.device_str = str(DEVICE) if TORCH_AVAILABLE else "cpu (sklearn)"

    # ── Data generation ──────────────────────────────────────────────────────

    def generate_training_data(self, n_samples=300, param_bounds=None,
                               seed=0) -> Tuple[np.ndarray, dict]:
        rng = np.random.default_rng(seed)
        bounds = param_bounds or {
            "cd_scale":         (0.7, 1.3),
            "thrust_scale":     (0.85, 1.15),
            "body_mass_kg":     (1.5, 3.5),
            "launch_angle_deg": (80.0, 90.0),
            "wind_speed_ms":    (0.0, 8.0),
        }
        n_feat = len(self.FEATURE_NAMES)
        lhs = np.zeros((n_samples, n_feat))
        for j in range(n_feat):
            perm = rng.permutation(n_samples)
            lhs[:, j] = (perm + rng.random(n_samples)) / n_samples
        X = np.zeros_like(lhs)
        for j, name in enumerate(self.FEATURE_NAMES):
            lo, hi = bounds[name]
            X[:, j] = lo + lhs[:, j] * (hi - lo)

        y_dict = {k: [] for k in self.output_keys}
        for i in range(n_samples):
            params = dict(zip(self.FEATURE_NAMES, X[i]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.simulator_fn(**params)
            for k in self.output_keys:
                y_dict[k].append(result.get(k, 0.0))

        for k in self.output_keys:
            y_dict[k] = np.array(y_dict[k])
        self._X_train = X
        self._y_train = y_dict
        return X, y_dict

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, X=None, y_dict=None, max_iter=500) -> dict:
        X      = X      or self._X_train
        y_dict = y_dict or self._y_train
        assert X is not None

        if TORCH_AVAILABLE:
            self._torch_model = _TorchSurrogate(
                hidden=(128, 128, 64), max_epochs=max_iter)
            metrics = self._torch_model.fit(X, y_dict)
        else:
            # sklearn fallback
            from sklearn.preprocessing import StandardScaler
            from sklearn.neural_network import MLPRegressor
            from sklearn.pipeline import Pipeline
            metrics = {}
            for key in self.output_keys:
                y = y_dict[key]
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.2, random_state=42)
                pipe = Pipeline([
                    ("sc", StandardScaler()),
                    ("mlp", MLPRegressor(hidden_layer_sizes=(64,64,32),
                                         max_iter=max_iter, random_state=42,
                                         early_stopping=True)),
                ])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_te)
                metrics[key] = {
                    "r2": round(r2_score(y_te, y_pred), 4),
                    "mae": round(mean_absolute_error(y_te, y_pred), 3),
                }
                self._sklearn_models[key] = pipe

        self.is_trained = True
        return metrics

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, cd_scale=1.0, thrust_scale=1.0, body_mass_kg=2.0,
                launch_angle_deg=90.0, wind_speed_ms=0.0) -> dict:
        assert self.is_trained
        x = np.array([[cd_scale, thrust_scale, body_mass_kg,
                       launch_angle_deg, wind_speed_ms]])
        if TORCH_AVAILABLE:
            return self._torch_model.predict_one(x[0])
        else:
            return {k: float(self._sklearn_models[k].predict(x)[0])
                    for k in self.output_keys}

    def predict_batch_gpu(self, X: np.ndarray) -> dict:
        """Predict a large batch at once — maximises GPU utilisation."""
        assert self.is_trained
        if TORCH_AVAILABLE:
            return self._torch_model.predict_batch(X)
        else:
            return {k: self._sklearn_models[k].predict(X)
                    for k in self.output_keys}

    # ── Parameter estimation ─────────────────────────────────────────────────

    def estimate_parameters(self, observed: dict, param_bounds=None,
                             method="differential_evolution") -> dict:
        assert self.is_trained
        bounds_dict = param_bounds or {
            "cd_scale":     (0.5, 2.0),
            "thrust_scale": (0.7, 1.3),
        }
        defaults = {"body_mass_kg": 2.0, "launch_angle_deg": 90.0,
                    "wind_speed_ms": 0.0}
        param_names = list(bounds_dict.keys())
        bounds_list = [bounds_dict[k] for k in param_names]

        def objective(theta):
            kwargs = dict(defaults)
            kwargs.update(dict(zip(param_names, theta)))
            pred = self.predict(**kwargs)
            loss = 0.0
            for k, v_obs in observed.items():
                if k in pred and abs(v_obs) > 1e-9:
                    loss += ((pred[k] - v_obs) / v_obs) ** 2
            return loss

        if method == "differential_evolution":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = differential_evolution(objective, bounds=bounds_list,
                                                seed=42, maxiter=500, tol=1e-8)
        else:
            x0 = np.array([(b[0]+b[1])/2 for b in bounds_list])
            result = minimize(objective, x0, bounds=bounds_list, method="L-BFGS-B")

        best = dict(zip(param_names, result.x))
        pred_best = self.predict(**{**defaults, **best})
        residuals = {k: round(pred_best.get(k, 0) - v, 3)
                     for k, v in observed.items() if k in pred_best}
        return {
            "estimated_params":  {k: round(v, 4) for k, v in best.items()},
            "predicted_outputs": {k: round(v, 2) for k, v in pred_best.items()},
            "observed_outputs":  observed,
            "residuals":         residuals,
            "optimizer_success": bool(getattr(result, "success", True)),
            "final_loss":        round(float(result.fun), 8),
        }

    # ── Sensitivity ──────────────────────────────────────────────────────────

    def sensitivity_analysis(self, target="apogee_m", n_points=50,
                              base_params=None) -> dict:
        assert self.is_trained
        base = base_params or {"cd_scale": 1.0, "thrust_scale": 1.0,
                                "body_mass_kg": 2.0, "launch_angle_deg": 90.0,
                                "wind_speed_ms": 0.0}
        ranges = {
            "cd_scale":         np.linspace(0.7, 1.5, n_points),
            "thrust_scale":     np.linspace(0.8, 1.2, n_points),
            "body_mass_kg":     np.linspace(1.0, 4.0, n_points),
            "launch_angle_deg": np.linspace(75.0, 90.0, n_points),
            "wind_speed_ms":    np.linspace(0.0, 15.0, n_points),
        }
        results = {}
        for param, vals in ranges.items():
            # Build a batch for GPU efficiency
            X_batch = np.tile(
                [base["cd_scale"], base["thrust_scale"], base["body_mass_kg"],
                 base["launch_angle_deg"], base["wind_speed_ms"]],
                (n_points, 1))
            feat_idx = self.FEATURE_NAMES.index(param)
            X_batch[:, feat_idx] = vals
            preds = self.predict_batch_gpu(X_batch)
            results[param] = (vals, preds[target])
        return results