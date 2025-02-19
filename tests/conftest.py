import pathlib
import pytest
from ase.collections import g2
from nx_ase import Molecule, Motor


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return pathlib.Path(__file__).resolve().parent / 'data'


@pytest.fixture
def get_test_file_path(test_data_dir):
    """Return a function that resolves filenames relative to the test data directory."""
    def _get_path(filename):
        return test_data_dir / filename
    return _get_path


# Common molecule fixtures that can be shared across tests
@pytest.fixture
def benzene():
    return Molecule(g2['C6H6'])


@pytest.fixture
def ethylene():
    return Molecule(g2['C2H4'])


@pytest.fixture
def acetylene():
    return Molecule(g2['C2H2'])
