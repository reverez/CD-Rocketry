# CD Rocketry: Advanced Flight Simulation Framework

CD Rocketry is a modular Python framework for simulating and analyzing model
rocket flight trajectories. It combines aerodynamic physics, standard
atmosphere calculations, motor thrust curves, Monte Carlo uncertainty analysis,
and a lightweight surrogate model for fast performance estimates.

## Quick Start

Create a virtual environment and install the runtime dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

For the optional GPU environment check:

```bash
.venv/bin/python -m pip install -r requirements-gpu.txt
.venv/bin/python check_gpu.py
```

For an NVIDIA RTX 5070, `check_gpu.py` should report `CUDA available: True` and
list the GPU name. If it reports `False`, the project can still run on CPU, but
the operating system, WSL, driver, or container is not exposing the GPU to
PyTorch.

Run a fast simulation plus ML smoke test:

```bash
.venv/bin/python smoke_test.py
```

Run the full demo:

```bash
.venv/bin/python main.py
```

## Project Layout

```text
CD-Rocketry/
|-- main.py                 # End-to-end demo and configuration entry point
|-- pyproject.toml          # Package metadata
|-- requirements.txt        # Runtime dependency list
|-- requirements-gpu.txt    # Optional PyTorch dependency for GPU checks
|-- check_gpu.py            # PyTorch/CUDA visibility check
|-- smoke_test.py           # Fast local smoke test
|-- rocketsim/
|   |-- __init__.py         # Public package exports
|   |-- simulator.py        # Core numerical integration engine
|   |-- aerodynamics.py     # Drag, lift, and dynamic pressure calculations
|   |-- atmosphere.py       # Standard atmosphere and wind model
|   |-- motor.py            # Motor thrust curves and mass depletion
|   |-- monte_carlo.py      # Uncertainty propagation
|   `-- ml_surrogate.py     # NumPy surrogate model
`-- tests/
    `-- test_smoke.py       # Fast smoke tests
```

## Installation

```bash
python -m pip install -r requirements.txt
```

For editable local development:

```bash
python -m pip install -e .
```

## Usage

Run the full demo:

```bash
python main.py
```

The demo builds a sample rocket, runs a single flight simulation, performs drag
and thrust parameter sweeps, runs a 50-sample Monte Carlo analysis, trains the
surrogate model, and estimates parameters from an observed apogee.

## Testing

```bash
python -m unittest discover -s tests
```

The smoke tests verify that the package imports, a flight produces sane summary
values, wind affects the horizontal state, and the surrogate model trains and
predicts.

## Functional Overview

1. **State initialization (`main.py`)**: defines vehicle mass, motor choice,
   geometry, launch angle, and recovery settings.
2. **Environmental modeling (`atmosphere.py`)**: calculates temperature,
   pressure, density, speed of sound, Mach number, and wind vector.
3. **Propulsion dynamics (`motor.py`)**: evaluates a time-dependent thrust curve
   and subtracts burned propellant mass during flight.
4. **Aerodynamic forces (`aerodynamics.py`)**: computes drag, lift, dynamic
   pressure, and Mach-dependent drag coefficients.
5. **Numerical solver (`simulator.py`)**: integrates the flight state with RK4
   until landing and reports apogee, max velocity, max Mach, max acceleration,
   max dynamic pressure, and total flight time.
6. **Predictive analytics (`monte_carlo.py`, `ml_surrogate.py`)**: runs
   uncertainty sweeps and trains a fast quadratic surrogate model for repeated
   prediction and parameter estimation.

## License

Distributed under the MIT License. See `LICENSE.md` for details.
