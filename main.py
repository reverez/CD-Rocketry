"""
RocketSim — Main Entry Point
Run this file from the project root:  python main.py
"""

import sys
import os

# Make the rocketsim package importable
sys.path.insert(0, os.path.dirname(__file__))

from rocketsim import (
    StandardAtmosphere,
    Motor,
    Aerodynamics,
    RocketSimulator,
    MonteCarlo,
    MLSurrogate,
)
from rocketsim.simulator import RocketConfig
from rocketsim.monte_carlo import UncertaintySpec
import numpy as np

# ──────────────────────────────────────────────────────────────
# 1. BUILD THE ROCKET
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  RocketSim — Physics-First Rocket Flight Simulator")
print("=" * 60)

atmosphere = StandardAtmosphere(wind_speed=3.0, wind_direction_deg=45.0)

motor = Motor.example_k550()          # AeroTech K550W-class motor
aero  = Aerodynamics(diameter_m=0.075)  # 75 mm body tube

config = RocketConfig(
    name             = "Demo Rocket",
    body_mass_kg     = 2.0,
    motor            = motor,
    aero             = aero,
    launch_angle_deg = 88.0,      # slightly off-vertical
    parachute_cd_area= 1.2,
    drogue_cd_area   = 0.15,
)

print("\n[Motor]")
for k, v in motor.summary().items():
    print(f"  {k:<25} {v}")

print("\n[Aerodynamics]")
for k, v in aero.summary().items():
    print(f"  {k:<25} {v}")

# ──────────────────────────────────────────────────────────────
# 2. SINGLE SIMULATION
# ──────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  SECTION 1 — Single Flight Simulation")
print("─" * 60)

sim    = RocketSimulator(config, atmosphere, dt=0.05)
result = sim.run()

print("\n[Flight Summary]")
for k, v in result.summary().items():
    print(f"  {k:<25} {v}")

# ──────────────────────────────────────────────────────────────
# 3. PARAMETER SWEEPS
# ──────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  SECTION 2 — Parameter Sweeps")
print("─" * 60)

cd_values     = np.linspace(0.7, 1.3, 5)
apogees_cd    = sim.sweep_cd(cd_values)
print("\n  Cd scale → Apogee (m):")
for cd, ap in zip(cd_values, apogees_cd):
    print(f"    Cd×{cd:.2f}  →  {ap:.1f} m")

thrust_values  = np.linspace(0.85, 1.15, 5)
apogees_thrust = sim.sweep_thrust(thrust_values)
print("\n  Thrust scale → Apogee (m):")
for ts, ap in zip(thrust_values, apogees_thrust):
    print(f"    T×{ts:.2f}   →  {ap:.1f} m")

# ──────────────────────────────────────────────────────────────
# 4. MONTE CARLO ANALYSIS
# ──────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  SECTION 3 — Monte Carlo Dispersion (200 runs)")
print("─" * 60)

mc = MonteCarlo(
    base_config     = config,
    base_atmosphere = atmosphere,
    n_samples       = 200,
    seed            = 42,
    cd_scale_unc    = UncertaintySpec("normal",  1.0, 0.05),
    thrust_scale_unc= UncertaintySpec("normal",  1.0, 0.03),
    body_mass_unc   = UncertaintySpec("normal",  0.0, 0.05),
    launch_angle_unc= UncertaintySpec("normal",  0.0, 0.5),
    wind_speed_unc  = UncertaintySpec("uniform", 3.0, 3.0),
    dt              = 0.05,
)

mc_result = mc.run()
summary   = mc_result.summary()

print("\n  Apogee statistics:")
s = summary["apogee_m"]
print(f"    mean   = {s['mean']} m")
print(f"    std    = {s['std']} m")
print(f"    5th %  = {s['p05']} m")
print(f"    95th % = {s['p95']} m")
print(f"    range  = [{s['min']}, {s['max']}] m")

print("\n  Max-velocity statistics:")
v = summary["max_velocity_ms"]
print(f"    mean   = {v['mean']} m/s")
print(f"    std    = {v['std']} m/s")

# ──────────────────────────────────────────────────────────────
# 5. ML SURROGATE MODEL
# ──────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  SECTION 4 — ML Surrogate Training & Parameter Estimation")
print("─" * 60)

# Wrapper that matches the MLSurrogate's expected signature
def sim_fn(cd_scale, thrust_scale, body_mass_kg, launch_angle_deg, wind_speed_ms):
    cfg = RocketConfig(
        name             = "surrogate_rocket",
        body_mass_kg     = body_mass_kg,
        motor            = Motor.example_k550(),
        aero             = Aerodynamics(diameter_m=0.075),
        launch_angle_deg = launch_angle_deg,
        parachute_cd_area= 1.2,
        drogue_cd_area   = 0.15,
    )
    atm = StandardAtmosphere(wind_speed=wind_speed_ms)
    s   = RocketSimulator(cfg, atm, dt=0.05,
                          cd_scale=cd_scale, thrust_scale=thrust_scale)
    r   = s.run()
    return {
        "apogee_m":       r.apogee_m,
        "max_velocity_ms": r.max_velocity,
        "max_mach":        r.max_mach,
    }

surrogate = MLSurrogate(simulator_fn=sim_fn)

print("\n  Generating 150 training samples (LHS)…")
X, y = surrogate.generate_training_data(n_samples=150, seed=7)

print("\n  Training MLP surrogates…")
metrics = surrogate.train(X, y, max_iter=1000)
print("\n  Surrogate accuracy:")
for key, m in metrics.items():
    print(f"    {key:<22} R²={m['r2']:.4f}  MAE={m['mae']:.3f}")

# Fast surrogate prediction
pred = surrogate.predict(cd_scale=1.0, thrust_scale=1.0, body_mass_kg=2.0,
                         launch_angle_deg=88.0, wind_speed_ms=3.0)
print("\n  Surrogate prediction (nominal params):")
for k, v in pred.items():
    print(f"    {k:<22} {round(v, 2)}")

# Inverse problem: infer cd_scale + thrust_scale from observed apogee
print("\n  Parameter estimation from observed apogee…")
estimated = surrogate.estimate_parameters(
    observed={"apogee_m": result.apogee_m},
    method="differential_evolution",
)
print(f"    Estimated params : {estimated['estimated_params']}")
print(f"    Predicted apogee : {estimated['predicted_outputs'].get('apogee_m')} m")
print(f"    Observed apogee  : {result.apogee_m:.1f} m")
print(f"    Residuals        : {estimated['residuals']}")

# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Simulation complete.")
print("=" * 60)