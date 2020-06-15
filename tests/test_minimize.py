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
Unit tests for the minimize.py module.
We test energy minimization which is done
via a C implementation of the FIRE algorithm [REF]
We use ctypes to import the code as a shared library. 
"""
from metamech.minimize import relax
from metamech.minimize import _get_params_from_lattice
from metamech.lattice import Lattice
import numpy as np
import pytest


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
        linear_stiffness=10,
        angular_stiffness=0.2
    )
    # activate all edges
    for edge in lattice._possible_edges:
        lattice.flip_edge(edge)

    return lattice


# test different conditions
# carefull, the total number
# of cases tested grows rapidly
xy_pert_list = [
    (x_pert, y_pert, klin, kang)
    for x_pert in np.linspace(0.05, 1.95, num=6)  # request.param[0]
    for y_pert in np.linspace(0.05, 1.95, num=6)  # request.param[1]
    for klin in np.geomspace(0.1, 10, num=4)       # request.param[2]
    for kang in np.geomspace(0.1, 10, num=4)       # request.param[3]
]


@pytest.fixture(params=xy_pert_list)
def square_lattice_to_relax(request):
    """
    Provide a square with its top-right vertex
    perturbed from its resting position.
    """
    # create lattice
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
    lattice = Lattice(
        nodes_positions=nodes_positions,
        edges_indices=edges_indices,
        linear_stiffness=request.param[2],
        angular_stiffness=request.param[3]
    )
    # activate all edges
    for edge in lattice._possible_edges:
        lattice.flip_edge(edge)

    ##
    # parameters for relax
    ##
    num_nodes = 4
    frozen = np.array([0, 1, 1, 1]).astype(np.intc)
    # input
    target_in = np.array([1]).astype(np.intc)
    Xin = np.array([0.])
    Yin = np.array([0.])
    Zin = np.array([0.])
    Nin = np.intc(1)
    # output
    target_out = np.array([1]).astype(np.intc)
    Xout = np.array([1.])
    Yout = np.array([1.])
    Zout = np.array([0.])
    kout = 1.
    Nout = np.intc(1)
    # other
    epot = np.array([0])
    fmax = np.array([1e-5])
    method = np.intc(0)
    enforce2D = np.intc(1)
    error = np.array([0]).astype(np.intc)
    step_relax = np.array([10000]).astype(np.intc)

    # parameters that depend on the lattice
    params_internal = _get_params_from_lattice(lattice)
    # parameters that depend on the actuator
    params_external = {"frozen": frozen,
                       "target_in": target_in, "Xin": Xin, "Yin": Yin, "Zin": Zin, "Nin": Nin,
                       "target_out": target_out, "Xout": Xout, "Yout": Yout, "Zout": Zout, "Nout": Nout, "kout": kout,
                       "epot": epot, "fmax": fmax, "method": method, "enforce2D": enforce2D, "error": error, "step_relax": step_relax
                       }
    # all params
    params = {**params_internal, **params_external}

    # displace the top-right node
    params["X"][0] = request.param[0]
    params["Y"][0] = request.param[1]

    return params


def test_minimize_relax_square_back_to_resting(square_lattice_to_relax):
    """test that a simple square goes back to rest position
    after being perturbed"""
    params = square_lattice_to_relax
    # relax
    relax(**params)
    # this is the real test: we make sure that the node went back to position (1, 1)
    assert np.isclose(params["X"][0], 1, rtol=1e-4)
    assert np.isclose(params["Y"][0], 1, rtol=1e-4)


def test_minimize__get_params_from_lattice(square_lattice):
    """make sure we can extract the parameters need for relax funciton from a simple lattice"""
    lattice = square_lattice
    params = _get_params_from_lattice(lattice)

    # expected parameters
    eparams = {
        'X': np.array([1., -1., -1.,  1.]),
        'Y': np.array([1.,  1., -1., -1.]),
        'Z': np.array([0., 0., 0., 0.]),
        'nnodes': 4,
        'ipart2': np.array([3, 0, 1, 0, 2]),
        'jpart2': np.array([0, 2, 2, 1, 3]),
        'r0': np.array([2., 2.82842712, 2., 2., 2.]),
        'kbond': np.array([10., 10., 10., 10., 10.]),
        'nbonds': 5,
        'ipart3': np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3]),
        'jpart3': np.array([3, 1, 2, 0, 2, 1, 0, 3, 2, 0]),
        'kpart3': np.array([1, 2, 3, 2, 0, 3, 1, 0, 0, 2]),
        'a0': np.array([45, 45, 45, 45, 90, 90, 270, 270, 270, 270]) / 180 * np.math.pi,
        'kangle': np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
        'nangles': 10
    }
    # check keys
    for key in eparams.keys():
        assert key in params.keys()
    for key in eparams.keys():
        assert key in eparams.keys()
    # check values
    for k in params.keys():
        try:
            # this will check the ints and floats
            assert params[k] == eparams[k]
        except:
            # this will check the arrays. since order is arbitrary
            # we need to sort
            for a, b in zip(sorted(params[k]), sorted(eparams[k])):
                assert np.isclose(a, b)


def test_minimize__get_params_from_lattice_types(square_lattice):
    "make sure everything has the expected type"
    lattice = square_lattice
    params = _get_params_from_lattice(lattice)
    # integer scalars
    for k in ["nnodes", "nbonds", "nangles"]:
        assert isinstance(params[k], int)
    # float arrays
    for k in ["X", "Y", "Z", "r0", "a0", "kangle"]:
        for x in params[k]:
            assert isinstance(x, float)
    # int arrays
    # IMPORTANT: these must be np.int32 due to ctypes compatibility
    # do not change to other int types!!
    for k in ["ipart2", "jpart2", "ipart3", "jpart3", "kpart3"]:
        for d in params[k]:
            assert isinstance(d, np.int32)
