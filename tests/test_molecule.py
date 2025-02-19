import os
import numpy as np
import pytest
from ase.collections import g2

from nx_ase import Molecule


@pytest.fixture
def benzene():
    return Molecule(g2['C6H6'])


@pytest.fixture
def ethylene():
    return Molecule(g2['C2H4'])


@pytest.fixture
def acetylene():
    return Molecule(g2['C2H2'])


def test_molecule_creation(benzene):
    assert benzene.get_chemical_formula() == 'C6H6'
    assert len(benzene.get_all_bonds()) > 0


def test_molecule_copy(benzene):
    copy = benzene.copy()
    assert copy.get_chemical_formula() == benzene.get_chemical_formula()
    assert np.allclose(copy.positions, benzene.positions)


def test_molecule_reorder(benzene):
    new_m = benzene.get_standard_order()
    mapping = new_m.best_mapping(benzene)
    assert mapping is not None
    assert len(mapping) == len(benzene)


def test_molecule_render(benzene):
    plotter = benzene.render(show=False)
    assert plotter is not None


def test_molecule_save_load(benzene, get_test_file_path):
    test_file = get_test_file_path('test_benzene.xyz')
    if os.path.exists(test_file):
        os.remove(test_file)

    benzene.save(test_file)
    loaded = Molecule.load(test_file)

    assert loaded.get_chemical_formula() == benzene.get_chemical_formula()
    assert np.allclose(loaded.positions, benzene.positions)

    os.remove(test_file)


def test_molecule_rotation(ethylene):
    original_pos = ethylene.positions.copy()
    rotated = ethylene.rotate_part(0, [0, 1], 90)

    assert not np.allclose(rotated.positions, original_pos)
    assert np.allclose(rotated.positions[0], [0., 0., 0.66748], atol=1e-5)


def test_simple_bonds(acetylene):
    acetylene.update_bond_labels()
    assert acetylene.G.edges[0, 1]['bond_type'] == 3


def test_ring_bonds(benzene):
    benzene.update_bond_labels()
    ring_bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    total_bond_order = sum(
        benzene.G.edges[i, j]['bond_type'] for i, j in ring_bonds)
    assert total_bond_order == 9  # Should be equivalent to three double bonds


def test_molecule_extend(benzene, ethylene):
    original_len = len(benzene)
    benzene.extend(ethylene)
    assert len(benzene) == original_len + len(ethylene)


def test_molecule_divide_in_two(ethylene):
    part1, part2 = ethylene.divide_in_two([0, 1])
    assert len(part1) + len(part2) == len(ethylene)
    assert 0 in part1
    assert 1 in part2
