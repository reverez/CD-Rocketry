"""
Monte Carlo Uncertainty Propagation

Runs an ensemble of flight simulations with randomised parameters drawn
from user-specified distributions, then computes statistics and
confidence intervals for key flight metrics.

Typical sources of uncertainty:
    - Drag coefficient  (manufacturing tolerances, surface finish)
    - Thrust scale      (motor-to-motor variation, temperature)
    - Launch angle      (rail misalignment, wind gusts)
    - Body mass         (payload/fuel measurement error)
    - Wind speed        (forecast uncertainty)
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor
import warnings

from .simulator import RocketSimulator, RocketConfig, FlightResult
from .atmosphere import StandardAtmosphere
from .motor import Motor
from .aerodynamics import Aerodynamics


@dataclass
class UncertaintySpec:
    """Defines the uncertainty distribution for one parameter.

    distribution : 'normal' | 'uniform' | 'lognormal'
    mean         : nominal value
    std          : standard deviation (normal) OR half-range (uniform)
    """
    distribution: str = "normal"
    mean: float = 1.0
    std: float = 0.05


class MonteCarlo:
    """
    Monte Carlo dispersion analysis for rocket flight.

    Parameters
    ----------
    base_config    : RocketConfig    — nominal rocket configuration
    base_atmosphere: StandardAtmosphere — nominal atmosphere
    n_samples      : int             — number of MC runs
    seed           : int             — random seed for reproducibility

    Uncertainty parameters (each is a UncertaintySpec or None to skip):
        cd_scale_unc       : drag coefficient multiplier
        thrust_scale_unc   : thrust multiplier
        body_mass_unc      : body mass additive variation [kg]
        launch_angle_unc   : launch angle additive variation [deg]
        wind_speed_unc     : wind speed [m/s]
    """

    def __init__(
        self,
        base_config: RocketConfig,
        base_atmosphere: Optional[StandardAtmosphere] = None,
        n_samples: int = 200,
        seed: int = 42,
        cd_scale_unc:     Optional[UncertaintySpec] = None,
        thrust_scale_unc: Optional[UncertaintySpec] = None,
        body_mass_unc:    Optional[UncertaintySpec] = None,
        launch_angle_unc: Optional[UncertaintySpec] = None,
        wind_speed_unc:   Optional[UncertaintySpec] = None,
        dt: float = 0.05,
    ):
        self.base_config = base_config
        self.base_atm    = base_atmosphere or StandardAtmosphere()
        self.n_samples   = n_samples
        self.rng         = np.random.default_rng(seed)
        self.dt          = dt

        # Defaults if not provided
        self.cd_unc     = cd_scale_unc     or UncertaintySpec("normal",  1.0, 0.05)
        self.T_unc      = thrust_scale_unc or UncertaintySpec("normal",  1.0, 0.03)
        self.mass_unc   = body_mass_unc    or UncertaintySpec("normal",  0.0, 0.05)
        self.angle_unc  = launch_angle_unc or UncertaintySpec("normal",  0.0, 0.5)
        self.wind_unc   = wind_speed_unc   or UncertaintySpec("uniform", 0.0, 3.0)

    # ------------------------------------------------------------------

    def _sample(self, spec: UncertaintySpec) -> float:
        d = spec.distribution.lower()
        if d == "normal":
            return float(self.rng.normal(spec.mean, spec.std))
        elif d == "uniform":
            return float(self.rng.uniform(spec.mean - spec.std, spec.mean + spec.std))
        elif d == "lognormal":
            mu  = np.log(spec.mean**2 / np.sqrt(spec.mean**2 + spec.std**2))
            sig = np.sqrt(np.log(1 + (spec.std / spec.mean)**2))
            return float(self.rng.lognormal(mu, sig))
        else:
            raise ValueError(f"Unknown distribution: {spec.distribution}")

    def _single_run(self, i: int) -> dict:
        """Execute one MC sample and return result dict."""
        cd_scale     = max(0.1, self._sample(self.cd_unc))
        thrust_scale = max(0.1, self._sample(self.T_unc))
        mass_delta   = self._sample(self.mass_unc)
        angle_delta  = self._sample(self.angle_unc)
        wind_speed   = max(0.0, abs(self._sample(self.wind_unc)))
        wind_dir     = float(self.rng.uniform(0, 360))

        # Build perturbed config
        import copy
        cfg = copy.deepcopy(self.base_config)
        cfg.body_mass_kg     = max(0.1, cfg.body_mass_kg + mass_delta)
        cfg.launch_angle_deg = np.clip(cfg.launch_angle_deg + angle_delta, 60, 90)

        atm = StandardAtmosphere(wind_speed=wind_speed, wind_direction_deg=wind_dir)

        sim = RocketSimulator(cfg, atm, dt=self.dt,
                              cd_scale=cd_scale, thrust_scale=thrust_scale)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sim.run()

        return {
            "run": i,
            "apogee_m":         result.apogee_m,
            "max_velocity_ms":  result.max_velocity,
            "max_mach":         result.max_mach,
            "max_accel_g":      result.max_accel / 9.80665,
            "flight_time_s":    result.flight_time,
            "cd_scale":         cd_scale,
            "thrust_scale":     thrust_scale,
            "body_mass_kg":     cfg.body_mass_kg,
            "launch_angle_deg": cfg.launch_angle_deg,
            "wind_speed_ms":    wind_speed,
        }

    def run(self) -> "MCResult":
        """Run all Monte Carlo samples. Returns MCResult."""
        runs = [self._single_run(i) for i in range(self.n_samples)]
        return MCResult(runs, self.base_config)


class MCResult:
    """Results from a Monte Carlo simulation ensemble."""

    def __init__(self, runs: List[dict], config: RocketConfig):
        self.runs   = runs
        self.config = config

        self.apogees     = np.array([r["apogee_m"]        for r in runs])
        self.velocities  = np.array([r["max_velocity_ms"] for r in runs])
        self.machs       = np.array([r["max_mach"]        for r in runs])
        self.accels_g    = np.array([r["max_accel_g"]     for r in runs])
        self.flight_times= np.array([r["flight_time_s"]   for r in runs])

    def statistics(self, metric: str = "apogee_m") -> dict:
        data = np.array([r[metric] for r in self.runs])
        return {
            "metric":   metric,
            "n":        len(data),
            "mean":     round(float(np.mean(data)), 2),
            "std":      round(float(np.std(data)),  2),
            "p05":      round(float(np.percentile(data,  5)), 2),
            "p25":      round(float(np.percentile(data, 25)), 2),
            "median":   round(float(np.median(data)),         2),
            "p75":      round(float(np.percentile(data, 75)), 2),
            "p95":      round(float(np.percentile(data, 95)), 2),
            "min":      round(float(np.min(data)), 2),
            "max":      round(float(np.max(data)), 2),
        }

    def summary(self) -> dict:
        return {
            "apogee_m":       self.statistics("apogee_m"),
            "max_velocity_ms":self.statistics("max_velocity_ms"),
            "max_mach":       self.statistics("max_mach"),
        }