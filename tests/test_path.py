import numpy as np
import pytest
from ase.collections import g2

from nx_ase import Motor, Path


@pytest.fixture
def simple_path():
    m1 = Motor(g2['C6H6'])
    m2 = Motor(g2['C6H6'])
    m2.translate(np.array([1.0, 0.0, 0.0]))  # Move second molecule
    return Path([m1, m2])


def test_path_creation(simple_path):
    assert len(simple_path) == 2
    assert isinstance(simple_path[0], Motor)
    assert isinstance(simple_path[1], Motor)


def test_path_copy(simple_path):
    path_copy = simple_path.copy()
    assert len(path_copy) == len(simple_path)
    assert np.allclose(path_copy[0].positions, simple_path[0].positions)


def test_path_extend(simple_path):
    m3 = Motor(g2['C6H6'])
    original_len = len(simple_path)
    simple_path.extend([m3])
    assert len(simple_path) == original_len + 1


def test_path_append(simple_path):
    m3 = Motor(g2['C6H6'])
    original_len = len(simple_path)
    simple_path.append(m3)
    assert len(simple_path) == original_len + 1


def test_path_iteration(simple_path):
    molecules = list(simple_path)
    assert len(molecules) == 2
    assert all(isinstance(m, Motor) for m in molecules)


def test_path_indexing(simple_path):
    assert isinstance(simple_path[0], Motor)
    assert isinstance(simple_path[-1], Motor)
