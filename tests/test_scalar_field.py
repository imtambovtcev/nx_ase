import pathlib
import numpy as np
import pytest

from nx_ase import ScalarField, Molecule


@pytest.fixture
def cube_file(get_test_file_path):
    return get_test_file_path('C2H4.eldens.cube')


@pytest.fixture
def scalar_field(cube_file):
    return ScalarField.load_cube(cube_file)


def test_scalar_field_load(scalar_field):
    assert isinstance(scalar_field, ScalarField)
    assert scalar_field.scalar_field is not None
    assert scalar_field.scalar_field.ndim == 3


def test_scalar_field_properties(scalar_field):
    assert hasattr(scalar_field, 'points')
    assert hasattr(scalar_field, 'dimensions')
    assert hasattr(scalar_field, 'volume_element')


def test_scalar_field_copy(scalar_field):
    copy = scalar_field.copy()
    assert np.allclose(copy.scalar_field, scalar_field.scalar_field)
    assert np.allclose(copy.org, scalar_field.org)


def test_scalar_field_transformations(scalar_field):
    original_org = scalar_field.org.copy()
    original_field = scalar_field.scalar_field.copy()

    # Test translation
    translation = np.array([1.0, 0.0, 0.0])
    scalar_field.translate(translation)
    assert np.allclose(scalar_field.org, original_org + translation)

    # Test rotation
    rotation = np.eye(3)  # Identity rotation
    scalar_field.rotate(rotation)
    assert np.allclose(scalar_field.scalar_field, original_field)


def test_molecule_from_cube(cube_file):
    molecule = Molecule.load_from_cube(cube_file)
    assert isinstance(molecule, Molecule)
    assert len(molecule.scalar_fields) > 0
