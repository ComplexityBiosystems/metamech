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
Tests for the base Metropolis class.

"""
import numpy as np
from metamech.metropolis import Metropolis
from metamech.actuator import Actuator
from metamech.lattice import Lattice
from metamech.spring import LinearSpring
import pytest
from pathlib import Path
from distutils import dir_util
import os

from typing import List
from typing import Any

import subprocess


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


@pytest.fixture
def square_lattice():
    """a simple square with one diagonal"""
    nodes_positions = np.array([
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1]
    ]).astype(float)
    edges_indices = np.array([
        [0, 1],
        [1, 2],
        [0, 2],
        [2, 3],
        [3, 0]
    ]).astype(int)

    # create lattice
    lattice = Lattice(
        nodes_positions=nodes_positions,
        edges_indices=edges_indices,
        linear_stiffness=1,
        angular_stiffness=1
    )
    # activate all edges
    for edge in lattice._possible_edges:
        lattice.flip_edge(edge)

    return lattice


@pytest.fixture
def square_actuator(square_lattice):
    """actuator for the simple square"""
    lattice = square_lattice
    actuator = Actuator(
        lattice=lattice,
        frozen_nodes=[],
        input_nodes=[0],
        input_vectors=np.array([
            [0.1, 0.2]
        ]),
        output_nodes=[1],
        output_vectors=np.array([
            [0.3, 0.4]
        ]),
    )
    return actuator


@pytest.fixture
def square_metropolis(square_actuator):
    """metropolis for the simple square"""
    actuator = square_actuator
    metropolis = Metropolis(
        actuator=actuator
    )
    return metropolis


def test_metropolis_init(square_metropolis: Metropolis):
    """test that metropolis objects can be created"""
    metropolis = square_metropolis
    assert isinstance(metropolis, Metropolis)


def test_metropolis_removable_edges(square_metropolis: Metropolis):
    """test that we compute the removable edges correctly"""
    metropolis = square_metropolis
    assert len(metropolis._removable_edges) == 1
    edge: LinearSpring = metropolis._removable_edges.pop()
    node_u, node_v = edge._nodes
    assert [2, 3] == sorted([node_u.label, node_v.label])


def test_metropolis_(square_lattice: Lattice):
    """test that metropolis raises error if no edges can be removed"""
    lattice = square_lattice
    actuator = Actuator(
        lattice=lattice,
        input_nodes=[0],
        output_nodes=[1],
        frozen_nodes=[2, 3],
        input_vectors=np.array([
            [0.1, 0.2]
        ]),
        output_vectors=np.array([
            [0.3, 0.4]
        ]),
    )
    with pytest.raises(RuntimeError):
        metropolis = Metropolis(actuator)


def test_metropolis_from_pickle(datadir):
    """make sure we can read a saved metropolis object from disc"""
    metropolis = Metropolis.from_pickle(datadir / "square_metropolis.p")
    assert isinstance(metropolis, Metropolis)


def test_metropolis_to_pickle(datadir, square_metropolis: Metropolis):
    """make sure we can write a metropolis object to disc"""
    metropolis = square_metropolis
    path_to_pickle: Path = datadir / "square_metropolis_fromtest.p"
    metropolis.to_pickle(str(path_to_pickle))
    assert Path(path_to_pickle).is_file()


def test_metropolis_to_pickle_no_overwrite(datadir, square_metropolis: Metropolis):
    """make sure the code refuses to overwrite existing files"""
    metropolis = square_metropolis
    path_to_pickle: Path = datadir / "square_metropolis.p"
    with pytest.raises(RuntimeError):
        metropolis.to_pickle(str(path_to_pickle))


def test_metropolis_to_pickle_unexistent_dirs(datadir, square_metropolis: Metropolis):
    """make sure we realize the output dir does not exist"""
    metropolis = square_metropolis
    path_to_pickle: Path = datadir / "nonexistent_dir" / "square_metropolis.p"
    with pytest.raises(RuntimeError):
        metropolis.to_pickle(str(path_to_pickle))


def test_metropolis_run_runs(square_metropolis: Metropolis):
    """test that the run function can be executed"""
    metropolis = square_metropolis
    metropolis.run(
        temperature=1,
        num_steps=10
    )
    assert len(metropolis.history["current_efficiency"]) > 0


def test_metropolis_temperature_mode_fails(square_metropolis: Metropolis):
    """check that cannot use unknown temperature modes"""
    metropolis = square_metropolis
    with pytest.raises(RuntimeError):
        metropolis.run(
            num_steps=10,
            temperature=1,
            temperature_change_mode="unknown"
        )


def test_metropolis_temperature_advance_fails(square_metropolis: Metropolis):
    """check that cannot use unknown temperature advance scheme"""
    metropolis = square_metropolis
    with pytest.raises(RuntimeError):
        metropolis.run(
            num_steps=10,
            temperature=1,
            temperature_advance_on="unknown"
        )


def test_metropolis_max_consequtive_rejects_fails_negative(square_metropolis: Metropolis):
    """make sure we cannot set max consecutive rejects to negative values"""
    metropolis = square_metropolis
    with pytest.raises(RuntimeError):
        metropolis.run(
            num_steps=10,
            temperature=1,
            max_consecutive_rejects=-1
        )


def test_metropolis_max_consequtive_rejects_fails_toosmall(square_metropolis: Metropolis):
    """make sure we cannot set max conseq rejects to a small value"""
    metropolis = square_metropolis
    with pytest.raises(RuntimeError):
        metropolis.run(
            num_steps=10,
            temperature=1,
            max_consecutive_rejects=3
        )


def test_metropolis_temperature_init_combinations(square_metropolis: Metropolis):
    """check that if no constant temperature is given then
    both initial and final are given
    """
    metropolis = square_metropolis
    # only initial
    with pytest.raises(AssertionError):
        metropolis.run(initial_temperature=1, num_steps=100)
    # only final
    with pytest.raises(AssertionError):
        metropolis.run(final_temperature=1, num_steps=100)


def test_metropolis_temperature_init_limits_1(square_metropolis: Metropolis):
    """check that when givin initial and final temeprature
    the initial must be positive and final must be non-negative
    """
    metropolis = square_metropolis
    # init = 0
    with pytest.raises(AssertionError):
        metropolis.run(
            initial_temperature=0,
            final_temperature=1,
            num_steps=100
        )


def test_metropolis_temperature_init_limits_2(square_metropolis: Metropolis):
    """check that when givin initial and final temeprature
    the initial must be positive and final must be non-negative
    """
    metropolis = square_metropolis
    # init < 0
    with pytest.raises(AssertionError):
        metropolis.run(
            initial_temperature=-1,
            final_temperature=1,
            num_steps=100
        )


def test_metropolis_temperature_init_limits_3(square_metropolis: Metropolis):
    """check that when givin initial and final temeprature
    the initial must be positive and final must be non-negative
    """
    metropolis = square_metropolis
    # final < 0
    with pytest.raises(AssertionError):
        metropolis.run(
            initial_temperature=1,
            final_temperature=-1,
            num_steps=100
        )


def test_metropolis_run_varying_temperature(square_metropolis: Metropolis):
    """check that we can run at varying temperature"""
    metropolis = square_metropolis
    metropolis.run(
        initial_temperature=1,
        final_temperature=0,
        temperature_advance_on="all",
        num_steps=5
    )
    assert len(set(metropolis.history["temperature"])) > 1


def test_metropolis_run_constant_temperature(square_metropolis: Metropolis):
    """check that we can run at constant temperature"""
    metropolis = square_metropolis
    metropolis.run(
        temperature=0.4,
        num_steps=5
    )
    assert len(set(metropolis.history["temperature"])) == 1


def test_metropolis_temperature_abs_mode(square_metropolis: Metropolis):
    """check that we change the temperature linearly
    notice the way it works:

    t0 = 2, tf = 0, num_steps = 2
    (2)     1      0
     -------+------+
    the first point t0 = 2 is not recorded in the history

    This way if you got from n to 0 in n steps you have n datapoints,
    not n+1 which is annoying.
    """
    metropolis = square_metropolis
    metropolis.run(
        initial_temperature=2,
        final_temperature=0,
        num_steps=2,
        temperature_change_mode="absolute",
        temperature_advance_on="all"
    )
    assert metropolis.history["temperature"] == [1, 0]


def test_metropolis_temperature_rel_mode(square_metropolis: Metropolis):
    """check that we change the temperature linearly
    notice the way it works:

    t0 = 4, tf = 1, num_steps = 2
    (4)     2      1
     -------+------+
    the first point t0 = 4 is not recorded in the history

    This way if you got from n to 0 in n steps you have n datapoints,
    not n+1 which is annoying.
    """
    metropolis = square_metropolis
    metropolis.run(
        initial_temperature=4,
        final_temperature=1,
        num_steps=2,
        temperature_change_mode="relative",
        temperature_advance_on="all"
    )
    assert metropolis.history["temperature"] == [2, 1]


def test_metropolis_nonexistent_change_mode(square_metropolis: Metropolis):
    """check that we cannot use nonexistent temperature change modes"""
    metropolis = square_metropolis
    with pytest.raises(RuntimeError):
        metropolis.run(
            initial_temperature=2,
            final_temperature=1,
            temperature_change_mode="nonexistent",
            num_steps=10
        )


def test_metropolis_get_ML_dataset_shape(square_metropolis: Metropolis):
    """make sure we are generating images of the correct size,
    which is 2 x 2 inches x the dpi. try several dpis.
    """
    metropolis = square_metropolis
    for dpi in [10, 27, 130]:
        metropolis.run(
            initial_temperature=1,
            final_temperature=0,
            num_steps=13,
            temperature_advance_on="all"
        )
        x, y = metropolis.get_ML_dataset(dpi=dpi)
        assert x[0].shape == (2 * dpi, 2 * dpi, 1)


def test_metropolis_get_ML_dataset_len(square_metropolis: Metropolis):
    """make sure we get the right number of images"""
    metropolis = square_metropolis
    metropolis.run(
        initial_temperature=1,
        final_temperature=0,
        num_steps=13,
        temperature_advance_on="all"
    )
    dpi = 100

    for subset in [
        [2, 3, 9],
        [0, 1],
        [12, 12, 10]
    ]:
        x, y = metropolis.get_ML_dataset(dpi=dpi, subset=subset)
        assert len(x) == len(subset)
        assert len(y) == len(subset)


def test_metropolis_get_ML_dataset_repeated(square_metropolis: Metropolis):
    """make sure we can ask for the same image more than once"""
    metropolis = square_metropolis
    metropolis.run(
        initial_temperature=1,
        final_temperature=0,
        num_steps=13,
        temperature_advance_on="all"
    )
    dpi = 100
    subset = [4, 4]
    x, y = metropolis.get_ML_dataset(dpi=dpi, subset=subset)
    x0, x1 = x
    y0, y1 = y
    assert np.all(x0 == x1)
    assert y0 == y1


def test_metropolis_history_lengths(square_metropolis: Metropolis):
    """make sure history is as long as it should"""
    metropolis = square_metropolis
    metropolis.run(
        temperature=0,
        num_steps=10,
        temperature_advance_on="all",
        max_consecutive_rejects=100
    )
    # the columns in history are not stable
    for col_name, col_values in metropolis.history.items():
        assert isinstance(col_values, list)
        assert len(col_values) == 10


def test_metropolis_reproducibility():
    """make sure that simulations can be reconstructed"""
    result = subprocess.run(
        ["python", "tests/test_metropolis/manual_reproducibility_test.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # decode result.stdout because its a binary string
    print('{}'.format(result.stdout.decode()))
    assert result.returncode == 0


def test_metropolis_is_valid_flag(square_lattice):
    """test that if max_iter is reached then is_valid
    is set to False"""
    lattice = square_lattice
    actuator = Actuator(
        lattice=lattice,
        frozen_nodes=[],
        input_nodes=[0],
        input_vectors=np.array([
            [0.1, 0.2]
        ]),
        output_nodes=[1],
        output_vectors=np.array([
            [0.3, 0.4]
        ]),
        max_iter=10
    )
    metropolis = Metropolis(actuator=actuator)
    metropolis.run(
        temperature=1,
        num_steps=100
    )
    assert not metropolis.is_valid


@pytest.mark.parametrize("max_iter,stop_at", [(10, 11), (20, 21), (50, 51)])
def test_metropolis_max_iter_early_stop(max_iter, stop_at, square_lattice):
    """test that we stop at the correct value of maxiter"""
    lattice = square_lattice
    actuator = Actuator(
        lattice=lattice,
        frozen_nodes=[],
        input_nodes=[0],
        input_vectors=np.array([
            [0.1, 0.2]
        ]),
        output_nodes=[1],
        output_vectors=np.array([
            [0.3, 0.4]
        ]),
        max_iter=max_iter
    )
    metropolis = Metropolis(actuator=actuator)
    metropolis.run(
        temperature=1,
        num_steps=100
    )
    assert metropolis.history["relax_internal_steps"][-1] == stop_at
