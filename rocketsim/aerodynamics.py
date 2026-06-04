"""
Aerodynamics Model

Computes drag, lift, and aerodynamic moments acting on the rocket.

Drag equation:
    D = 0.5 * rho * v² * Cd * A_ref

where:
    rho   — air density [kg/m³]
    v     — airspeed [m/s]
    Cd    — drag coefficient (may vary with Mach)
    A_ref — reference cross-sectional area [m²] = pi * (d/2)²

Mach-dependent Cd follows a simple transonic correction:
    Near Mach 1, drag rises significantly (wave drag).
    This is modelled with a smooth bump around M = 0.9–1.2.
"""

import numpy as np
from typing import Optional


class Aerodynamics:
    """Aerodynamic force model for a slender, axisymmetric rocket.

    Parameters
    ----------
    diameter_m     : body tube outer diameter [m]
    cd_subsonic    : drag coefficient at subsonic speeds (M < 0.8)
    cd_supersonic  : drag coefficient at supersonic speeds (M > 1.5)
    cd_transonic   : peak drag coefficient at M ≈ 1.0 (wave drag)
    cl_alpha       : lift-curve slope per radian (for 2D/3D extensions)
    """

    def __init__(
        self,
        diameter_m: float,
        cd_subsonic: float = 0.45,
        cd_supersonic: float = 0.30,
        cd_transonic: float = 0.65,
        cl_alpha: float = 2.0,
    ):
        self.diameter_m    = diameter_m
        self.cd_subsonic   = cd_subsonic
        self.cd_supersonic = cd_supersonic
        self.cd_transonic  = cd_transonic
        self.cl_alpha      = cl_alpha

        self.ref_area = np.pi * (diameter_m / 2.0) ** 2   # m²

    # ------------------------------------------------------------------

    def drag_coefficient(self, mach: float) -> float:
        """Return Cd as a function of Mach number using a smooth model.

        Uses a logistic blend between subsonic, transonic, and supersonic values.
        """
        M = abs(mach)

        if M < 0.8:
            return self.cd_subsonic
        elif M > 1.5:
            return self.cd_supersonic
        else:
            # Smooth transonic peak using a Gaussian bump centred at M=1.0
            peak = self.cd_transonic
            sigma = 0.20
            M_peak = 1.0
            bump = (peak - self.cd_subsonic) * np.exp(-((M - M_peak)**2) / (2 * sigma**2))
            # Blend toward supersonic past M=1
            if M <= 1.0:
                base = self.cd_subsonic
            else:
                t = (M - 1.0) / 0.5   # 0→1 as M: 1.0→1.5
                base = self.cd_subsonic + t * (self.cd_supersonic - self.cd_subsonic)
            return base + bump

    def drag_force(self, altitude_m: float, velocity_ms: float,
                   atmosphere) -> float:
        """Compute drag force magnitude [N].

        Parameters
        ----------
        altitude_m  : current altitude [m]
        velocity_ms : airspeed magnitude [m/s]
        atmosphere  : StandardAtmosphere instance
        """
        rho  = atmosphere.density(altitude_m)
        mach = atmosphere.mach_number(altitude_m, velocity_ms)
        cd   = self.drag_coefficient(mach)
        return 0.5 * rho * velocity_ms**2 * cd * self.ref_area

    def dynamic_pressure(self, altitude_m: float, velocity_ms: float,
                         atmosphere) -> float:
        """Dynamic pressure q = 0.5 * rho * v² [Pa]."""
        rho = atmosphere.density(altitude_m)
        return 0.5 * rho * velocity_ms**2

    def lift_force(self, altitude_m: float, velocity_ms: float,
                   angle_of_attack_rad: float, atmosphere) -> float:
        """Compute lift force [N] for a given angle of attack [rad].

        L = 0.5 * rho * v² * Cl * A_ref
        where Cl = cl_alpha * alpha
        """
        rho = atmosphere.density(altitude_m)
        Cl  = self.cl_alpha * angle_of_attack_rad
        return 0.5 * rho * velocity_ms**2 * Cl * self.ref_area

    def summary(self) -> dict:
        return {
            "diameter_m":     self.diameter_m,
            "ref_area_m2":    round(self.ref_area, 6),
            "cd_subsonic":    self.cd_subsonic,
            "cd_transonic":   self.cd_transonic,
            "cd_supersonic":  self.cd_supersonic,
        }