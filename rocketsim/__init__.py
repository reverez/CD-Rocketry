"""
RocketSim — Physics-First Rocket Flight Simulator with ML Parameter Estimation
"""

from .atmosphere import StandardAtmosphere
from .motor import Motor
from .aerodynamics import Aerodynamics
from .simulator import RocketSimulator
from .monte_carlo import MonteCarlo
from .ml_surrogate import MLSurrogate

__version__ = "1.0.0"
__all__ = [
    "StandardAtmosphere",
    "Motor",
    "Aerodynamics",
    "RocketSimulator",
    "MonteCarlo",
    "MLSurrogate",
]

