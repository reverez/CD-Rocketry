"""
Standard Atmosphere Model (ISA — International Standard Atmosphere)

Implements the US Standard Atmosphere 1976 with accurate layer-by-layer
lapse rates. Provides air density, pressure, temperature, and speed of sound
as a function of altitude.

Layers:
  0–11 km  : Troposphere      lapse rate = -6.5 K/km
  11–20 km : Lower Stratosphere  isothermal at 216.65 K
  20–32 km : Upper Stratosphere  lapse rate = +1.0 K/km
"""

import numpy as np


class StandardAtmosphere:
    """US Standard Atmosphere 1976.

    All altitudes in metres, temperatures in Kelvin, pressures in Pa,
    densities in kg/m³.
    """

    # Physical constants
    R = 287.058      # specific gas constant for dry air  [J/(kg·K)]
    g0 = 9.80665     # standard gravity                   [m/s²]
    gamma = 1.4      # ratio of specific heats (dry air)

    # Atmosphere layer table: (base_altitude_m, base_temp_K, base_pressure_Pa, lapse_rate_K/m)
    _LAYERS = [
        (0,      288.15, 101325.0,  -0.0065),
        (11000,  216.65,  22632.1,   0.0),
        (20000,  216.65,   5474.89,  0.001),
        (32000,  228.65,    868.02,  0.0028),
        (47000,  270.65,    110.91,  0.0),
        (51000,  270.65,     66.94, -0.0028),
        (71000,  214.65,      3.96, -0.002),
    ]

    def __init__(self, wind_speed: float = 0.0, wind_direction_deg: float = 0.0):
        """
        Parameters
        ----------
        wind_speed : float
            Horizontal wind speed in m/s (constant with altitude for simplicity).
        wind_direction_deg : float
            Wind direction in degrees (0 = north, 90 = east).
        """
        self.wind_speed = wind_speed
        self.wind_direction_deg = wind_direction_deg

    # ------------------------------------------------------------------
    # Core atmospheric state
    # ------------------------------------------------------------------

    def temperature(self, altitude_m: float) -> float:
        """Return air temperature [K] at given geometric altitude [m]."""
        altitude_m = max(0.0, altitude_m)
        T, _ = self._layer_state(altitude_m)
        return T

    def pressure(self, altitude_m: float) -> float:
        """Return static pressure [Pa] at given geometric altitude [m]."""
        altitude_m = max(0.0, altitude_m)
        _, P = self._layer_state(altitude_m)
        return P

    def density(self, altitude_m: float) -> float:
        """Return air density [kg/m³] at given geometric altitude [m]."""
        T = self.temperature(altitude_m)
        P = self.pressure(altitude_m)
        return P / (self.R * T)

    def speed_of_sound(self, altitude_m: float) -> float:
        """Return speed of sound [m/s] at given geometric altitude [m]."""
        T = self.temperature(altitude_m)
        return np.sqrt(self.gamma * self.R * T)

    def mach_number(self, altitude_m: float, velocity_ms: float) -> float:
        """Return Mach number given altitude and airspeed."""
        a = self.speed_of_sound(altitude_m)
        return abs(velocity_ms) / a if a > 0 else 0.0

    def wind_vector(self) -> np.ndarray:
        """Return 2D wind velocity vector [vx, vy] in m/s (East, North)."""
        angle_rad = np.radians(self.wind_direction_deg)
        vx = self.wind_speed * np.sin(angle_rad)   # East component
        vy = self.wind_speed * np.cos(angle_rad)   # North component
        return np.array([vx, vy])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _layer_state(self, altitude_m: float):
        """Compute (temperature, pressure) for the layer containing altitude_m."""
        # Find the correct layer
        layer = self._LAYERS[0]
        for L in self._LAYERS:
            if altitude_m >= L[0]:
                layer = L
            else:
                break

        h_b, T_b, P_b, L_r = layer
        dh = altitude_m - h_b

        T = T_b + L_r * dh

        if abs(L_r) < 1e-12:
            # Isothermal layer
            P = P_b * np.exp(-self.g0 * dh / (self.R * T_b))
        else:
            # Gradient layer
            P = P_b * (T / T_b) ** (-self.g0 / (L_r * self.R))

        return T, P

    # ------------------------------------------------------------------
    # Convenience: profile over an altitude array
    # ------------------------------------------------------------------

    def profile(self, altitudes_m: np.ndarray) -> dict:
        """Return a dict of atmosphere arrays over a range of altitudes."""
        T = np.array([self.temperature(h) for h in altitudes_m])
        P = np.array([self.pressure(h)    for h in altitudes_m])
        rho = np.array([self.density(h)   for h in altitudes_m])
        a   = np.array([self.speed_of_sound(h) for h in altitudes_m])
        return {"altitude": altitudes_m, "temperature": T,
                "pressure": P, "density": rho, "speed_of_sound": a}