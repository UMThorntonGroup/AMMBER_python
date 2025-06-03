# What is AMMBER?
The AI-assisted Microstructure Model BuildER (AMMBER) is an ongoing project at the University of Michigan.

AMMBER_python is a utility for extracting free-energy data and formatting it for use in phase-field simulation codes.

Phase-field models, which incorporate thermodynamic and kinetic data from atomistic calculations and experiments, have become a key computational tool for understanding microstructural evolution and providing a path to control and optimize morphologies and topologies of structures from nanoscale to microscales. However, due to the complexity of interactions between multiple species, these models are difficult to parameterize. In this project, we developed algorithms and software that automate and optimize the selection of thermodynamic and kinetic parameters for phase-field simulations of microstructure evolution in multicomponent systems.

Presently, the framework consists of two modules: [AMMBER_python](https://github.com/UMThorntonGroup/AMMBER_python), which is used to extract phase-field usable free energies from general data sources, and [AMMBER-PRISMS-PF](https://github.com/UMThorntonGroup/AMMBER-PRISMS-PF), which provides an open-source suite of multi-component, multi-phase-field model implementations with a simple, flexible interface for defining a system of thermodynamic and kinetic parameters.

# Quick Start Guide

### Install:

To install the AMMBER Python package, you can use `pip`. First, ensure your `pip` is up to date:

```bash
python -m pip install --upgrade pip
pip install ammber
```

#### Installing from Source:
If you want to install the package directly from the source code, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/UMThorntonGroup/AMMBER_python.git
   cd AMMBER_python
   ```

2. Upgrade `pip` and install the package:
   ```bash
   python -m pip install --upgrade pip
   pip install .
   ```

#### Development Installation:
For development purposes, you can install the package in editable mode:

1. Clone the repository:
   ```bash
   git clone https://github.com/UMThorntonGroup/AMMBER_python.git
   cd AMMBER_python
   ```

2. Upgrade `pip` and install in editable mode:
   ```bash
   python -m pip install --upgrade pip
   pip install -e .
   ```

#### Dependencies:
AMMBER requires the following Python packages:
- `numpy`
- `scipy`
- `pycalphad`

These dependencies will be installed automatically when using `pip install ammber`. If you encounter issues, you can manually install them using:
```bash
pip install numpy scipy pycalphad
```

# License:
MIT License. Please see [LICENSE](LICENSE.md) for details.

# Links
[AMMBER_python Repository](https://github.com/UMThorntonGroup/AMMBER_python) <br>
[AMMBER-PRISMS-PF Repository](https://github.com/UMThorntonGroup/AMMBER-PRISMS-PF) <br>
