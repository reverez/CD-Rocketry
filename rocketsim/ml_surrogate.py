"""
Lightweight ML surrogate model for rocket flight outputs.

The surrogate intentionally depends only on NumPy so the core project can run
without optional scientific stacks such as SciPy, scikit-learn, or PyTorch.
It fits a ridge-regularized quadratic response surface for each output.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np


class MLSurrogate:
    """Fast surrogate model for apogee, max velocity, and max Mach."""

    FEATURE_NAMES = [
        "cd_scale",
        "thrust_scale",
        "body_mass_kg",
        "launch_angle_deg",
        "wind_speed_ms",
    ]

    def __init__(
        self,
        simulator_fn: Callable,
        output_keys: Optional[List[str]] = None,
        ridge_alpha: float = 1e-6,
    ):
        self.simulator_fn = simulator_fn
        self.output_keys = output_keys or ["apogee_m", "max_velocity_ms", "max_mach"]
        self.ridge_alpha = ridge_alpha
        self._X_train = None
        self._y_train = {}
        self._x_mean = None
        self._x_std = None
        self._weights = {}
        self.is_trained = False
        self.using_gpu = False
        self.device_str = "cpu (numpy)"

    def generate_training_data(
        self,
        n_samples: int = 300,
        param_bounds: Optional[dict] = None,
        seed: int = 0,
    ) -> Tuple[np.ndarray, dict]:
        """Generate Latin-hypercube samples and evaluate the simulator."""
        rng = np.random.default_rng(seed)
        bounds = param_bounds or {
            "cd_scale": (0.7, 1.3),
            "thrust_scale": (0.85, 1.15),
            "body_mass_kg": (1.5, 3.5),
            "launch_angle_deg": (80.0, 90.0),
            "wind_speed_ms": (0.0, 8.0),
        }

        lhs = np.zeros((n_samples, len(self.FEATURE_NAMES)))
        for col in range(lhs.shape[1]):
            order = rng.permutation(n_samples)
            lhs[:, col] = (order + rng.random(n_samples)) / n_samples

        X = np.zeros_like(lhs)
        for col, name in enumerate(self.FEATURE_NAMES):
            low, high = bounds[name]
            X[:, col] = low + lhs[:, col] * (high - low)

        y_dict = {key: [] for key in self.output_keys}
        for row in X:
            result = self.simulator_fn(**dict(zip(self.FEATURE_NAMES, row)))
            for key in self.output_keys:
                y_dict[key].append(float(result.get(key, 0.0)))

        self._X_train = X
        self._y_train = {key: np.array(values) for key, values in y_dict.items()}
        return self._X_train, self._y_train

    def train(self, X=None, y_dict=None, max_iter=None) -> dict:
        """Fit one quadratic ridge model per output and return validation metrics."""
        X = self._X_train if X is None else np.asarray(X, dtype=float)
        y_dict = self._y_train if y_dict is None else y_dict
        if X is None or not y_dict:
            raise ValueError("Call generate_training_data() or pass X and y_dict first.")

        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        design = self._design_matrix(X)

        rng = np.random.default_rng(42)
        indices = rng.permutation(len(X))
        split = max(1, int(0.8 * len(X)))
        train_idx = indices[:split]
        test_idx = indices[split:] if split < len(X) else indices[:split]

        metrics = {}
        for key in self.output_keys:
            y = np.asarray(y_dict[key], dtype=float)
            phi_train = design[train_idx]
            y_train = y[train_idx]

            penalty = self.ridge_alpha * np.eye(phi_train.shape[1])
            penalty[0, 0] = 0.0
            lhs = phi_train.T @ phi_train + penalty
            rhs = phi_train.T @ y_train
            weights = np.linalg.solve(lhs, rhs)
            self._weights[key] = weights

            y_true = y[test_idx]
            y_pred = design[test_idx] @ weights
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
            mae = float(np.mean(np.abs(y_true - y_pred)))
            metrics[key] = {"r2": round(r2, 4), "mae": round(mae, 3)}

        self.is_trained = True
        return metrics

    def predict(
        self,
        cd_scale=1.0,
        thrust_scale=1.0,
        body_mass_kg=2.0,
        launch_angle_deg=90.0,
        wind_speed_ms=0.0,
    ) -> dict:
        """Predict all configured outputs for one parameter set."""
        X = np.array(
            [[cd_scale, thrust_scale, body_mass_kg, launch_angle_deg, wind_speed_ms]],
            dtype=float,
        )
        return {key: float(values[0]) for key, values in self.predict_batch(X).items()}

    def predict_batch(self, X: np.ndarray) -> dict:
        """Predict all configured outputs for a batch of parameter rows."""
        if not self.is_trained:
            raise RuntimeError("Call train() before predict().")
        design = self._design_matrix(np.asarray(X, dtype=float))
        return {key: design @ weights for key, weights in self._weights.items()}

    def predict_batch_gpu(self, X: np.ndarray) -> dict:
        """Compatibility alias for the previous GPU-oriented API."""
        return self.predict_batch(X)

    def estimate_parameters(
        self,
        observed: dict,
        param_bounds: Optional[dict] = None,
        method: str = "random_search",
    ) -> dict:
        """Estimate free parameters by minimizing normalized output residuals."""
        if not self.is_trained:
            raise RuntimeError("Call train() before estimate_parameters().")

        bounds = param_bounds or {
            "cd_scale": (0.5, 2.0),
            "thrust_scale": (0.7, 1.3),
        }
        defaults = {
            "cd_scale": 1.0,
            "thrust_scale": 1.0,
            "body_mass_kg": 2.0,
            "launch_angle_deg": 90.0,
            "wind_speed_ms": 0.0,
        }
        names = list(bounds)
        rng = np.random.default_rng(42)

        candidates = []
        midpoint = {name: (low + high) / 2 for name, (low, high) in bounds.items()}
        candidates.append(midpoint)
        for _ in range(2000):
            candidates.append(
                {name: rng.uniform(low, high) for name, (low, high) in bounds.items()}
            )

        best_params = None
        best_loss = float("inf")
        for candidate in candidates:
            kwargs = {**defaults, **candidate}
            loss = self._loss(kwargs, observed)
            if loss < best_loss:
                best_loss = loss
                best_params = candidate

        # Coordinate refinement around the best random candidate.
        step_sizes = {
            name: (bounds[name][1] - bounds[name][0]) / 8.0 for name in names
        }
        for _ in range(40):
            improved = False
            for name in names:
                for direction in (-1.0, 1.0):
                    low, high = bounds[name]
                    trial = dict(best_params)
                    trial[name] = float(
                        np.clip(trial[name] + direction * step_sizes[name], low, high)
                    )
                    kwargs = {**defaults, **trial}
                    loss = self._loss(kwargs, observed)
                    if loss < best_loss:
                        best_loss = loss
                        best_params = trial
                        improved = True
            if not improved:
                step_sizes = {name: step / 2.0 for name, step in step_sizes.items()}

        final_kwargs = {**defaults, **best_params}
        prediction = self.predict(**final_kwargs)
        residuals = {
            key: round(prediction[key] - value, 3)
            for key, value in observed.items()
            if key in prediction
        }
        return {
            "estimated_params": {
                key: round(float(value), 4) for key, value in best_params.items()
            },
            "predicted_outputs": {
                key: round(float(value), 2) for key, value in prediction.items()
            },
            "observed_outputs": observed,
            "residuals": residuals,
            "optimizer_success": True,
            "final_loss": round(float(best_loss), 8),
        }

    def sensitivity_analysis(
        self,
        target: str = "apogee_m",
        n_points: int = 50,
        base_params: Optional[dict] = None,
    ) -> dict:
        """Sweep each feature independently and return target predictions."""
        if not self.is_trained:
            raise RuntimeError("Call train() before sensitivity_analysis().")

        base = base_params or {
            "cd_scale": 1.0,
            "thrust_scale": 1.0,
            "body_mass_kg": 2.0,
            "launch_angle_deg": 90.0,
            "wind_speed_ms": 0.0,
        }
        ranges = {
            "cd_scale": np.linspace(0.7, 1.5, n_points),
            "thrust_scale": np.linspace(0.8, 1.2, n_points),
            "body_mass_kg": np.linspace(1.0, 4.0, n_points),
            "launch_angle_deg": np.linspace(75.0, 90.0, n_points),
            "wind_speed_ms": np.linspace(0.0, 15.0, n_points),
        }

        results = {}
        base_row = np.array([base[name] for name in self.FEATURE_NAMES], dtype=float)
        for name, values in ranges.items():
            batch = np.tile(base_row, (n_points, 1))
            batch[:, self.FEATURE_NAMES.index(name)] = values
            results[name] = (values, self.predict_batch(batch)[target])
        return results

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        scaled = (X - self._x_mean) / self._x_std
        columns = [np.ones(len(scaled))]
        columns.extend(scaled[:, i] for i in range(scaled.shape[1]))
        columns.extend(scaled[:, i] ** 2 for i in range(scaled.shape[1]))
        for i in range(scaled.shape[1]):
            for j in range(i + 1, scaled.shape[1]):
                columns.append(scaled[:, i] * scaled[:, j])
        return np.column_stack(columns)

    def _loss(self, kwargs: dict, observed: dict) -> float:
        prediction = self.predict(**kwargs)
        loss = 0.0
        for key, observed_value in observed.items():
            if key in prediction and abs(observed_value) > 1e-12:
                loss += ((prediction[key] - observed_value) / observed_value) ** 2
        return float(loss)


MLSurrogateGPU = MLSurrogate
