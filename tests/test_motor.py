import numpy as np
import pytest
from ase.collections import g2
from nx_ase import Motor


@pytest.fixture
def mpf_motor(get_test_file_path):
    return Motor.load(get_test_file_path('mpf_motor.xyz'))


def test_motor_creation(mpf_motor):
    assert mpf_motor.get_chemical_formula() == 'C27H20'
    assert isinstance(mpf_motor, Motor)


def test_motor_stator_rotor(mpf_motor):
    bond_info = mpf_motor.get_stator_rotor_bond()
    assert 'bond' in bond_info
    assert 'bond_stator_node' in bond_info
    assert 'bond_rotor_node' in bond_info


def test_motor_copy(mpf_motor):
    copy = mpf_motor.copy()
    assert isinstance(copy, Motor)
    assert copy.get_chemical_formula() == mpf_motor.get_chemical_formula()
    assert np.allclose(copy.positions, mpf_motor.positions)


def test_motor_rotation(mpf_motor):
    original_pos = mpf_motor.positions.copy()
    rotation_atoms = mpf_motor.find_rotation_atoms()
    mpf_motor.to_origin(*rotation_atoms)
    assert not np.allclose(mpf_motor.positions, original_pos)


def test_motor_get_break_bonds(mpf_motor):
    break_bonds = mpf_motor.get_break_bonds()
    assert isinstance(break_bonds, list)
    assert all(len(bond) == 2 for bond in break_bonds)
