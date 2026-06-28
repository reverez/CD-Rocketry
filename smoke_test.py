"""Fast project smoke test for simulation and ML surrogate wiring."""

from rocketsim import (
    Aerodynamics,
    MLSurrogate,
    Motor,
    RocketSimulator,
    StandardAtmosphere,
)
from rocketsim.simulator import RocketConfig


def make_config(
    body_mass_kg: float = 2.0,
    launch_angle_deg: float = 88.0,
) -> RocketConfig:
    return RocketConfig(
        name="Smoke Test Rocket",
        body_mass_kg=body_mass_kg,
        motor=Motor.example_k550(),
        aero=Aerodynamics(diameter_m=0.075),
        launch_angle_deg=launch_angle_deg,
        parachute_cd_area=1.2,
        drogue_cd_area=0.15,
    )


def simulator_fn(cd_scale, thrust_scale, body_mass_kg, launch_angle_deg, wind_speed_ms):
    sim = RocketSimulator(
        make_config(body_mass_kg, launch_angle_deg),
        StandardAtmosphere(wind_speed=wind_speed_ms),
        dt=0.1,
        cd_scale=cd_scale,
        thrust_scale=thrust_scale,
    )
    result = sim.run()
    return {
        "apogee_m": result.apogee_m,
        "max_velocity_ms": result.max_velocity,
        "max_mach": result.max_mach,
    }


def main() -> None:
    result = RocketSimulator(make_config(), StandardAtmosphere(), dt=0.1).run()
    print(f"Simulation OK: apogee={result.apogee_m:.1f} m")

    surrogate = MLSurrogate(simulator_fn)
    print(f"ML backend: {surrogate.device_str}, gpu={surrogate.using_gpu}")
    x_train, y_train = surrogate.generate_training_data(n_samples=16, seed=1)
    metrics = surrogate.train(x_train, y_train, max_iter=5)
    prediction = surrogate.predict(
        cd_scale=1.0,
        thrust_scale=1.0,
        body_mass_kg=2.0,
        launch_angle_deg=88.0,
        wind_speed_ms=3.0,
    )
    print(f"Training OK: outputs={list(metrics)}")
    print(f"Prediction OK: apogee={prediction['apogee_m']:.1f} m")


if __name__ == "__main__":
    main()
