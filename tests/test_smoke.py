import unittest

from rocketsim import Aerodynamics, MLSurrogate, Motor, RocketSimulator, StandardAtmosphere
from rocketsim.simulator import RocketConfig


def build_config():
    return RocketConfig(
        name="Smoke Test Rocket",
        body_mass_kg=2.0,
        motor=Motor.example_k550(),
        aero=Aerodynamics(diameter_m=0.075),
        launch_angle_deg=88.0,
        parachute_cd_area=1.2,
        drogue_cd_area=0.15,
    )


class RocketSimSmokeTests(unittest.TestCase):
    def test_single_flight_produces_sane_summary(self):
        result = RocketSimulator(build_config(), StandardAtmosphere(), dt=0.1).run()

        self.assertGreater(result.apogee_m, 1000.0)
        self.assertGreater(result.max_velocity, 100.0)
        self.assertGreater(result.flight_time, 10.0)
        self.assertGreater(result.max_mach, 0.1)

    def test_wind_changes_horizontal_flight_state(self):
        calm = RocketSimulator(build_config(), StandardAtmosphere(wind_speed=0.0), dt=0.1).run()
        windy = RocketSimulator(
            build_config(),
            StandardAtmosphere(wind_speed=8.0, wind_direction_deg=90.0),
            dt=0.1,
        ).run()

        self.assertNotAlmostEqual(float(calm.vx[-1]), float(windy.vx[-1]), places=3)

    def test_surrogate_trains_and_predicts(self):
        def sim_fn(cd_scale, thrust_scale, body_mass_kg, launch_angle_deg, wind_speed_ms):
            cfg = RocketConfig(
                name="Surrogate Smoke",
                body_mass_kg=body_mass_kg,
                motor=Motor.example_f39(),
                aero=Aerodynamics(diameter_m=0.05),
                launch_angle_deg=launch_angle_deg,
            )
            result = RocketSimulator(
                cfg,
                StandardAtmosphere(wind_speed=wind_speed_ms),
                dt=0.1,
                cd_scale=cd_scale,
                thrust_scale=thrust_scale,
            ).run()
            return {
                "apogee_m": result.apogee_m,
                "max_velocity_ms": result.max_velocity,
                "max_mach": result.max_mach,
            }

        surrogate = MLSurrogate(sim_fn)
        X, y = surrogate.generate_training_data(n_samples=30, seed=1)
        metrics = surrogate.train(X, y)
        prediction = surrogate.predict()

        self.assertIn("apogee_m", metrics)
        self.assertGreater(prediction["apogee_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
