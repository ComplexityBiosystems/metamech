# Codes used for the paper:
# "Automatic Design of Mechanical Metamaterial Actuators"
# by S. Bonfanti, R. Guerra, F. Font-Clos, R. Rayneau-Kirkhope, S. Zapperi
# Center for Complexity and Biosystems, University of Milan
# (c) University of Milan
#
#
######################################################################
#
# End User License Agreement (EULA)
# Your access to and use of the downloadable code (the "Code") is subject
# to a non-exclusive,  revocable, non-transferable,  and limited right to
# use the Code for the exclusive purpose of undertaking academic,
# governmental, or not-for-profit research. Use of the Code or any part
# thereof for commercial purposes is strictly prohibited in the absence
# of a Commercial License Agreement from the University of Milan. For
# information contact the Technology Transfer Office of the university
# of Milan (email: tto@unimi.it)
#
#######################################################################

"""
Tests for Input/Output functions (io.py)

io.py contains functions to read/write in some formats.
so far:
+ LAMMPS

"""
import pytest
from distutils import dir_util
import os

from metamech.io import read_lammps
import numpy as np


@pytest.fixture
def datadir(tmpdir, request):
    """
    Create a tmpdir with data.

    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.

    taken from:
    https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))
    return tmpdir


def test_io_read_lammps(datadir):
    params = read_lammps(datadir / "input.data")
    assert np.all(params['nodes_positions'] == np.array([
        [1.,  1.,  0.],
        [-1.,  1.,  0.],
        [-1., -1.,  0.],
        [1., -1.,  0.]]))
    assert np.all(params['edges_indices'] == np.array([
        [0, 1],
        [0, 2],
        [1, 2],
        [0, 3],
        [2, 3]]))
    assert params['input_nodes'] == [2]
    assert params['output_nodes'] == [3]
    assert params['frozen_nodes'] == [1]
