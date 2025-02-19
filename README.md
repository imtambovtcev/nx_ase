# nx_ase

A Python library that extends ASE (Atomic Simulation Environment) with advanced molecular manipulation and visualization capabilities using NetworkX.

## Features

- **Enhanced Molecular Representation**: Combines ASE's atomic handling with NetworkX's graph capabilities
- **Smart Bond Detection**: Automatic detection and classification of chemical bonds
- **Molecular Visualization**: Advanced 3D visualization using PyVista with support for:
  - Multiple bond types (single, double, triple)
  - Customizable atom and bond appearances
  - Scalar field visualization
  - Interactive viewing options
- **Molecular Motors**: Specialized support for molecular motor analysis and manipulation
- **Fragment Handling**: Tools for working with molecular fragments
- **Path Analysis**: Tools for analyzing molecular motion paths
- **Scalar Field Analysis**: Support for electron density and other scalar field visualizations

## Installation

```bash
git clone https://github.com/imtambovtcev/nx_ase.git
cd nx_ase
pip install .
```

## Basic Usage

```python
from nx_ase import Molecule, Motor, Path

# Load a molecule from a file
molecule = Molecule.load("molecule.xyz")

# Visualize the molecule
molecule.render()

# Work with molecular motors
motor = Motor.load("motor.xyz")
stator, rotor = motor.get_stator_rotor()

# Create and analyze paths
path = Path([motor1, motor2, motor3])
path.reorder_atoms_of_intermediate_images()

# Work with scalar fields
molecule = Molecule.load_from_cube("density.cube")
molecule.scalar_fields["Electron density"].render()
```

## Key Classes

- **Molecule**: Enhanced version of ASE's Atoms with graph-based analysis
- **Motor**: Specialized class for molecular motor analysis
- **Fragment**: Tools for working with molecular fragments
- **Path**: Analysis of sequences of molecular configurations
- **ScalarField**: Handling and visualization of scalar fields

## Advanced Features

### Visualization Options

```python
molecule.render(
    show_hydrogens=True,
    show_numbers=True,
    valency=True,
    resolution=20,
    background_color='black'
)
```

### Motor Analysis

```python
motor = Motor.load("motor.xyz")
bond_info = motor.get_stator_rotor_bond()
break_bonds = motor.get_break_bonds()
```

### Path Manipulation

```python
path = Path.load("path.xyz")
path.rotate()
path.fix_bonds_breaks()
path.reorder_atoms_of_intermediate_images()
```

## Testing

```bash
pytest tests/
```
