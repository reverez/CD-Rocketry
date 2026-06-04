
# CD Rocketry: Advanced Flight Simulation Framework

CD Rocketry is a modular Python framework designed to simulate and analyze model rocket flight trajectories. The system integrates traditional aerodynamic physics with stochastic analysis and machine learning to predict flight performance, apogee, and landing zones.

## System Architecture and Workflow

The framework operates by splitting the physics and analysis into distinct, independent modules under the `rocketsim` package. 

```text
CD-Rocketry/
├── main.py                 # Simulation execution and configuration entry point
└── rocketsim/
    ├── __init__.py         # Package initialization
    ├── simulator.py        # Core numerical integration engine
    ├── aerodynamics.py     # Lift, drag, and stability calculations
    ├── atmosphere.py       # Barometric models and wind vector simulation
    ├── motor.py            # Solid motor thrust curves and mass depletion
    ├── monte_carlo.py      # Statistical dispersion and uncertainty wrapper
    └── ml_surrogate.py     # Machine learning acceleration layer

```

### Functional Overview

1. **State Initialization (`main.py`):** The user defines the vehicle geometry, dry mass, motor choice, and launch rail configurations.
2. **Environmental Modeling (`atmosphere.py`):** Calculates localized air density, temperature, and pressure based on current altitude. It simultaneously applies a wind vector model to simulate lateral forces and weather-cocking during the ascent.
3. **Propulsion Dynamics (`motor.py`):** Evaluates time-dependent thrust using the motor's specific curve. It dynamically subtracts the burned propellant weight from the total vehicle mass at every time step, updating the vehicle's center of gravity.
4. **Aerodynamic Forces (`aerodynamics.py`):** Computes instantaneous drag forces using the standard drag equation. It scales the drag coefficient ($C_d$) based on velocity variations.
5. **Numerical Solver (`simulator.py`):** Acts as the core integration loop. It aggregates thrust, gravity, and aerodynamic vectors to update acceleration, velocity, and position coordinates at high temporal resolution until touchdown.
6. **Advanced Predictive Analytics (`monte_carlo.py` & `ml_surrogate.py`):** * The **Monte Carlo** module runs thousands of parallel simulations, injecting Gaussian noise into variables like ignition wind angle and total motor impulse to map out a probabilistic landing zone dispersion.
* The **ML Surrogate** module uses a trained regression model to bypass the physics loop entirely, outputting instant apogee and safety margin predictions based on inputs.

---

## Technical Specifications

### Aerodynamic Drag Core

The atmospheric module dynamically recalculates fluid density ($\rho$) as a function of altitude to resolve the drag equation across the flight envelope:

$$F_d = \frac{1}{2} \rho v^2 C_d A$$

Where:

* $F_d$ = Drag force vector
* $\rho$ = Altitude-dependent air density
* $v$ = Relative velocity vector (including wind components)
* $C_d$ = Drag coefficient
* $A$ = Cross-sectional area of the airframe

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

```
