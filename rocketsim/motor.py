"""
Motor Model

Handles motor thrust curve, propellant mass flow, total impulse, and burn time.
Supports loading from CSV (time, thrust) or from a built-in lookup of common
RASP/ThrustCurve.org format data.

Equations
---------
Mass flow (kg/s):
    dm/dt = -thrust(t) / (I_sp * g0)       [approximate; perfect for ideal nozzle]

where I_sp is the specific impulse in seconds.
"""

import numpy as np
import csv
import os


class Motor:
    """Rocket motor model built from a thrust-time curve.

    Parameters
    ----------
    thrust_data : list of (time, thrust) tuples  [s, N]
    dry_mass_kg : float     — motor casing mass after burn
    propellant_mass_kg : float — initial propellant mass
    name : str              — motor designation (e.g. 'AeroTech K550')
    isp_s : float           — specific impulse [s]; used to derive mass flow
    """

    g0 = 9.80665  # m/s²

    def __init__(
        self,
        thrust_data: list,
        dry_mass_kg: float,
        propellant_mass_kg: float,
        name: str = "Unknown Motor",
        isp_s: float = 200.0,
    ):
        self.name = name
        self.dry_mass_kg = dry_mass_kg
        self.propellant_mass_kg = propellant_mass_kg
        self.isp_s = isp_s

        # Sort and unpack thrust data
        thrust_data = sorted(thrust_data, key=lambda x: x[0])
        if not thrust_data:
            raise ValueError("thrust_data must contain at least one time/thrust point")

        times  = np.array([p[0] for p in thrust_data])
        thrust = np.array([p[1] for p in thrust_data])

        # Ensure thrust starts and ends at zero
        if times[0] > 0:
            times  = np.concatenate([[0.0], times])
            thrust = np.concatenate([[0.0], thrust])
        if thrust[-1] != 0:
            times  = np.concatenate([times,  [times[-1] + 1e-6]])
            thrust = np.concatenate([thrust, [0.0]])

        self._times  = times
        self._thrust = thrust

        self.burn_time = times[-1]
        self.total_impulse = np.trapezoid(thrust, times)  # N·s

    # ------------------------------------------------------------------

    def thrust(self, t: float) -> float:
        """Return thrust [N] at time t [s]."""
        return float(np.interp(t, self._times, self._thrust, left=0.0, right=0.0))

    def propellant_remaining(self, t: float) -> float:
        """Approximate propellant remaining [kg] at time t via cumulative impulse."""
        if t <= 0:
            return self.propellant_mass_kg
        if t >= self.burn_time:
            return 0.0
        # Fraction of total impulse consumed
        t_clipped = np.linspace(0, min(t, self.burn_time), 500)
        thrust = np.interp(t_clipped, self._times, self._thrust, left=0.0, right=0.0)
        impulse_consumed = np.trapezoid(thrust, t_clipped)
        fraction_consumed = impulse_consumed / (self.total_impulse + 1e-12)
        return self.propellant_mass_kg * (1.0 - fraction_consumed)

    def mass(self, t: float) -> float:
        """Total motor mass [kg] = dry casing + remaining propellant."""
        return self.dry_mass_kg + self.propellant_remaining(t)

    def is_burning(self, t: float) -> bool:
        return 0 <= t < self.burn_time

    def summary(self) -> dict:
        return {
            "name":            self.name,
            "burn_time_s":     round(self.burn_time, 3),
            "total_impulse_Ns":round(self.total_impulse, 1),
            "avg_thrust_N":    round(self.total_impulse / self.burn_time, 1),
            "peak_thrust_N":   round(float(np.max(self._thrust)), 1),
            "propellant_kg":   self.propellant_mass_kg,
            "dry_mass_kg":     self.dry_mass_kg,
            "isp_s":           self.isp_s,
        }

    # ------------------------------------------------------------------
    # Factory / class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, filepath: str, dry_mass_kg: float,
                 propellant_mass_kg: float, **kwargs) -> "Motor":
        """Load thrust curve from a two-column CSV (time_s, thrust_N)."""
        data = []
        with open(filepath, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                try:
                    data.append((float(row[0]), float(row[1])))
                except (ValueError, IndexError):
                    continue
        name = kwargs.pop("name", os.path.basename(filepath).replace(".csv", ""))
        return cls(data, dry_mass_kg, propellant_mass_kg, name=name, **kwargs)

    @classmethod
    def example_k550(cls) -> "Motor":
        """AeroTech K550W-ish motor — representative high-power amateur motor."""
        # Simplified thrust curve (time_s, thrust_N) for a K550 class motor
        thrust_data = [
            (0.000,   0.0),
            (0.020, 680.0),
            (0.100, 620.0),
            (0.500, 570.0),
            (1.000, 550.0),
            (1.500, 545.0),
            (2.000, 540.0),
            (2.500, 530.0),
            (3.000, 510.0),
            (3.200, 480.0),
            (3.400, 380.0),
            (3.500, 200.0),
            (3.600,  50.0),
            (3.650,   0.0),
        ]
        return cls(
            thrust_data,
            dry_mass_kg=0.72,
            propellant_mass_kg=0.97,
            name="AeroTech K550W (approx)",
            isp_s=198,
        )

    @classmethod
    def example_f39(cls) -> "Motor":
        """Estes F39T — small F-class motor for lighter rockets."""
        thrust_data = [
            (0.00,  0.0),
            (0.02, 60.0),
            (0.10, 45.0),
            (0.50, 40.0),
            (1.00, 38.0),
            (1.50, 35.0),
            (1.80, 20.0),
            (1.90,  5.0),
            (1.95,  0.0),
        ]
        return cls(
            thrust_data,
            dry_mass_kg=0.035,
            propellant_mass_kg=0.045,
            name="Estes F39T (approx)",
            isp_s=96,
        )
