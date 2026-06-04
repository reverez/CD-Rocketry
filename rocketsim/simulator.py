"""
RocketSimulator — Core 1D/2D Flight Dynamics Engine

Integrates the equations of motion for a rocket in flight using the
classical 4th-order Runge-Kutta (RK4) method.

Equations of motion (1D vertical):
    dz/dt  = v                               [position]
    dv/dt  = (T - D - m*g) / m              [Newton's second law]
    dm/dt  ≈ -T / (Isp * g0)               [propellant burn]

where:
    T  = thrust(t)     [N]
    D  = drag(z, v)    [N] — always opposes velocity
    m  = total mass    [kg] (structure + motor)
    g  = gravity       [m/s²] — slight variation with altitude
    z  = altitude      [m]
    v  = velocity      [m/s]

The integrator can be extended to 2D/3D by passing a launch_angle_deg
parameter, whereupon horizontal and vertical velocity components are
tracked separately.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from .atmosphere import StandardAtmosphere
from .motor import Motor
from .aerodynamics import Aerodynamics


@dataclass
class RocketConfig:
    """Rocket physical configuration."""
    name: str = "My Rocket"
    body_mass_kg: float = 2.0         # structural mass (airframe + payload)
    motor: Optional[Motor] = None
    aero: Optional[Aerodynamics] = None
    launch_angle_deg: float = 90.0    # 90° = vertical
    parachute_cd_area: float = 1.0    # Cd * A for main chute [m²]
    drogue_cd_area:    float = 0.1    # Cd * A for drogue [m²]
    deploy_apogee: bool = True        # deploy drogue at apogee


@dataclass
class FlightState:
    """Snapshot of rocket state at one timestep."""
    time:     float = 0.0
    altitude: float = 0.0
    velocity: float = 0.0            # vertical velocity [m/s]
    vx:       float = 0.0            # horizontal velocity [m/s]
    mass:     float = 0.0
    thrust:   float = 0.0
    drag:     float = 0.0
    accel:    float = 0.0
    mach:     float = 0.0
    dyn_pressure: float = 0.0
    phase:    str = "pad"


class FlightResult:
    """Complete flight simulation result."""

    def __init__(self, states: List[FlightState], config: RocketConfig):
        self.states = states
        self.config = config
        self._build_arrays()

    def _build_arrays(self):
        s = self.states
        self.time         = np.array([st.time     for st in s])
        self.altitude     = np.array([st.altitude for st in s])
        self.velocity     = np.array([st.velocity for st in s])
        self.vx           = np.array([st.vx       for st in s])
        self.mass         = np.array([st.mass     for st in s])
        self.thrust       = np.array([st.thrust   for st in s])
        self.drag         = np.array([st.drag     for st in s])
        self.accel        = np.array([st.accel    for st in s])
        self.mach         = np.array([st.mach     for st in s])
        self.dyn_pressure = np.array([st.dyn_pressure for st in s])

        idx_apogee = int(np.argmax(self.altitude))
        self.apogee_m     = float(self.altitude[idx_apogee])
        self.apogee_time  = float(self.time[idx_apogee])
        self.max_velocity = float(np.max(np.abs(self.velocity)))
        self.max_mach     = float(np.max(self.mach))
        self.max_accel    = float(np.max(self.accel))
        self.max_q        = float(np.max(self.dyn_pressure))
        self.flight_time  = float(self.time[-1])

    def summary(self) -> dict:
        return {
            "rocket":           self.config.name,
            "apogee_m":         round(self.apogee_m, 1),
            "apogee_ft":        round(self.apogee_m * 3.28084, 1),
            "apogee_time_s":    round(self.apogee_time, 2),
            "max_velocity_ms":  round(self.max_velocity, 2),
            "max_mach":         round(self.max_mach, 3),
            "max_accel_ms2":    round(self.max_accel, 2),
            "max_accel_g":      round(self.max_accel / 9.80665, 2),
            "max_dyn_press_Pa": round(self.max_q, 1),
            "flight_time_s":    round(self.flight_time, 2),
        }


class RocketSimulator:
    """
    Physics-first rocket flight simulator.

    Integrates 1D (vertical) or 2D (planar) equations of motion
    using 4th-order Runge-Kutta with adaptive timestep safeguards.

    Parameters
    ----------
    config     : RocketConfig  — rocket geometry/mass/motor/aero
    atmosphere : StandardAtmosphere — atmospheric model
    dt         : float         — base timestep [s]
    cd_scale   : float         — multiplicative correction on Cd (ML hook)
    thrust_scale : float       — multiplicative correction on thrust (ML hook)
    """

    G0 = 9.80665   # standard gravity m/s²
    RE = 6371000.0 # Earth radius m (for altitude-dependent gravity)

    def __init__(
        self,
        config: RocketConfig,
        atmosphere: Optional[StandardAtmosphere] = None,
        dt: float = 0.01,
        cd_scale: float = 1.0,
        thrust_scale: float = 1.0,
    ):
        self.config       = config
        self.atm          = atmosphere or StandardAtmosphere()
        self.dt           = dt
        self.cd_scale     = cd_scale
        self.thrust_scale = thrust_scale

        assert config.motor is not None, "RocketConfig must have a Motor."
        assert config.aero  is not None, "RocketConfig must have Aerodynamics."

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, max_time: float = 600.0) -> FlightResult:
        """Run a complete flight simulation from launch to landing.

        Returns a FlightResult containing time-series arrays and summary stats.
        """
        states = []
        t = 0.0
        dt = self.dt

        launch_angle_rad = np.radians(self.config.launch_angle_deg)
        sin_a = np.sin(launch_angle_rad)
        cos_a = np.cos(launch_angle_rad)  # horizontal component

        # Initial state: [altitude, vertical_vel, horizontal_vel]
        z  = 0.0
        vz = 0.0
        vx = 0.0

        phase = "boost"
        apogee_detected = False
        drogue_deployed = False
        main_deployed   = False
        parachute_cd_area = 0.0

        motor = self.config.motor
        aero  = self.config.aero

        while t <= max_time:
            m = self.config.body_mass_kg + motor.mass(t)
            v_total = np.sqrt(vz**2 + vx**2)

            T = motor.thrust(t) * self.thrust_scale

            # Gravity (slightly altitude-dependent)
            g = self.G0 * (self.RE / (self.RE + max(0, z)))**2

            # Drag — acts along velocity vector
            rho  = self.atm.density(z)
            mach = self.atm.mach_number(z, v_total)
            Cd   = aero.drag_coefficient(mach) * self.cd_scale
            D    = 0.5 * rho * v_total**2 * Cd * aero.ref_area

            # Parachute drag
            if drogue_deployed or main_deployed:
                D_chute = 0.5 * rho * v_total**2 * parachute_cd_area
                D += D_chute

            # Net force components along trajectory and horizontal
            if v_total > 0.01:
                drag_z = D * (vz / v_total)
                drag_x = D * (vx / v_total)
            else:
                drag_z = 0.0
                drag_x = 0.0

            # Thrust components (along launch direction during boost)
            if motor.is_burning(t):
                T_z = T * sin_a
                T_x = T * cos_a
            else:
                T_z = 0.0
                T_x = 0.0

            # Accelerations
            az = (T_z - drag_z) / m - g
            ax = (T_x - drag_x) / m

            dyn_pressure = 0.5 * rho * v_total**2

            # Record state
            st = FlightState(
                time=t, altitude=z, velocity=vz, vx=vx,
                mass=m, thrust=T, drag=D, accel=abs(az),
                mach=mach, dyn_pressure=dyn_pressure, phase=phase,
            )
            states.append(st)

            # -- Phase transitions --
            if not motor.is_burning(t) and phase == "boost":
                phase = "coast"

            if not apogee_detected and phase == "coast" and vz < 0:
                apogee_detected = True
                phase = "drogue"
                if self.config.deploy_apogee:
                    drogue_deployed = True
                    parachute_cd_area = self.config.drogue_cd_area

            if apogee_detected and z < 300 and not main_deployed:
                main_deployed = True
                drogue_deployed = False
                parachute_cd_area = self.config.parachute_cd_area
                phase = "main"

            if z <= 0.0 and t > 0.5:
                phase = "landed"
                break

            # RK4 integration of [z, vz, vx]
            def derivatives(state, time):
                sz, svz, svx = state
                sv_total = np.sqrt(svz**2 + svx**2)
                sm = self.config.body_mass_kg + motor.mass(time)
                sg = self.G0 * (self.RE / (self.RE + max(0, sz)))**2
                srho = self.atm.density(max(0, sz))
                smach = self.atm.mach_number(max(0, sz), sv_total)
                sCd   = aero.drag_coefficient(smach) * self.cd_scale
                sD    = 0.5 * srho * sv_total**2 * sCd * aero.ref_area
                if drogue_deployed or main_deployed:
                    sD += 0.5 * srho * sv_total**2 * parachute_cd_area
                if sv_total > 0.01:
                    sdrag_z = sD * (svz / sv_total)
                    sdrag_x = sD * (svx / sv_total)
                else:
                    sdrag_z = sdrag_x = 0.0
                if motor.is_burning(time):
                    sT_z = motor.thrust(time) * self.thrust_scale * sin_a
                    sT_x = motor.thrust(time) * self.thrust_scale * cos_a
                else:
                    sT_z = sT_x = 0.0
                saz = (sT_z - sdrag_z) / sm - sg
                sax = (sT_x - sdrag_x) / sm
                return np.array([svz, saz, sax])

            state = np.array([z, vz, vx])
            k1 = derivatives(state,            t)
            k2 = derivatives(state + dt/2*k1,  t + dt/2)
            k3 = derivatives(state + dt/2*k2,  t + dt/2)
            k4 = derivatives(state + dt*k3,    t + dt)
            state_new = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            z  = max(0.0, state_new[0])
            vz = state_new[1]
            vx = state_new[2]
            t += dt

        return FlightResult(states, self.config)

    # ------------------------------------------------------------------
    # Sensitivity / parameter sweep
    # ------------------------------------------------------------------

    def sweep_cd(self, cd_values: np.ndarray) -> List[float]:
        """Run multiple simulations with different Cd scale factors.
        Returns list of apogee values [m]."""
        apogees = []
        for cd in cd_values:
            sim = RocketSimulator(self.config, self.atm, self.dt,
                                  cd_scale=cd, thrust_scale=self.thrust_scale)
            result = sim.run()
            apogees.append(result.apogee_m)
        return apogees

    def sweep_thrust(self, thrust_values: np.ndarray) -> List[float]:
        """Run multiple simulations with different thrust scale factors."""
        apogees = []
        for ts in thrust_values:
            sim = RocketSimulator(self.config, self.atm, self.dt,
                                  cd_scale=self.cd_scale, thrust_scale=ts)
            result = sim.run()
            apogees.append(result.apogee_m)
        return apogees