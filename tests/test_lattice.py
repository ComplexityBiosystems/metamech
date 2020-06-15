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
Tests for metamech.lattice

Unit testing the base Lattice class.
"""
import pytest
import numpy as np
from metamech.lattice import Lattice


@pytest.fixture
def lattice() -> Lattice:
    """fixture to create a simple dummy square"""
    nodes_positions = np.array([
        [1,  1],
        [-1,  1],
        [-1, -1],
        [1, -1],
    ])
    edges_indices = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 2]
    ])
    lattice = Lattice(
        nodes_positions=nodes_positions,
        edges_indices=edges_indices,
        linear_stiffness=0.1,
        angular_stiffness=0.1
    )
    return lattice


def test_lattice_init(lattice: Lattice):
    """test that we can instantiate a lattice"""
    assert isinstance(lattice, Lattice)
    assert isinstance(lattice._nodes_positions, np.ndarray)
    assert isinstance(lattice._edges_indices, np.ndarray)
    assert isinstance(lattice._linear_stiffness, float)
    assert isinstance(lattice._angular_stiffness, float)


def test_lattice_add_edge(lattice: Lattice):
    """test that we can add edges"""
    for edge in lattice._possible_edges:
        lattice._add_edge(edge)
        assert edge in lattice.edges


def test_lattice_remove_edge(lattice: Lattice):
    """test that we can remove edges"""
    for edge in lattice._possible_edges:
        lattice._add_edge(edge)
    for edge in lattice._possible_edges:
        lattice._remove_edge(edge)
        assert edge not in lattice.edges


def test_lattice_flip_edge(lattice: Lattice):
    """test that we can flip edges

    Flipping an edge means adding it if it is not present
    and removing it if it is present
    """
    for edge in lattice.edges:
        lattice.flip_edge(edge)
        assert edge in lattice.edges
    for edge in lattice.edges:
        lattice.flip_edge(edge)
        assert edge not in lattice.edges


def test_lattice__add_nodes():
    """make sure nodes are correctly added at init"""
    nodes_positions = np.array([
        [1,  1],
        [-1,  1],
        [-1, -1],
        [1, -1],
    ])
    edges_indices = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 2]
    ])
    lattice = Lattice(
        nodes_positions=nodes_positions,
        edges_indices=edges_indices,
        linear_stiffness=0.1,
        angular_stiffness=0.1
    )
    assert len(lattice.nodes) == len(nodes_positions)
    for node, (x, y) in zip(lattice.nodes, nodes_positions):
        assert x == node.x
        assert y == node.y


def test_lattice__generate_possible_edges(lattice: Lattice):
    """test that edges are added correctly at init"""
    nodes_positions = np.array([
        [1,  1],
        [-1,  1],
        [-1, -1],
        [1, -1],
    ])
    edges_indices = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 2]
    ])
    lattice = Lattice(
        nodes_positions=nodes_positions,
        edges_indices=edges_indices,
        linear_stiffness=0.1,
        angular_stiffness=0.1
    )
    for edge, (i, j) in zip(lattice.edges, edges_indices):
        u, v = edge._nodes
        assert i == u.label
        assert j == v.label


def test_lattice_fails_edges_not_in_range_up():
    """make sure we raise an error when trying
    to instantiate with indices outside range"""
    nodes_positions = np.array([
        [1,  1],
        [-1,  1],
        [-1, -1],
        [1, -1],
    ])
    edges_indices = np.array([
        [0, 1],
        [1, 2],
        [2, 33],  # wrong index here
        [3, 0],
        [0, 2]
    ])
    with pytest.raises(RuntimeError):
        lattice = Lattice(
            nodes_positions=nodes_positions,
            edges_indices=edges_indices,
            linear_stiffness=0.1,
            angular_stiffness=0.1
        )


def test_lattice_fails_edges_not_in_range_down():
    """make sure we raise an error when trying
    to instantiate with negative indices"""
    nodes_positions = np.array([
        [1,  1],
        [-1,  1],
        [-1, -1],
        [1, -1],
    ])
    edges_indices = np.array([
        [0, 1],
        [1, 2],
        [2, -5],  # wrong index here
        [3, 0],
        [0, 2]
    ])
    with pytest.raises(RuntimeError):
        lattice = Lattice(
            nodes_positions=nodes_positions,
            edges_indices=edges_indices,
            linear_stiffness=0.1,
            angular_stiffness=0.1
        )


def test_lattice_fails_nonunique_nodes():
    """make sure we raise an error when passing nonunique nodes"""
    nodes_positions = np.array([
        [1,  1],
        [1,  1],  # repeated node here
        [-1, -1],
        [1, -1],
    ])
    edges_indices = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 2]
    ])
    with pytest.raises(RuntimeError):
        lattice = Lattice(
            nodes_positions=nodes_positions,
            edges_indices=edges_indices,
            linear_stiffness=0.1,
            angular_stiffness=0.1
        )


def test_lattice_fails_nonunique_edges():
    """make sure we raise an error when passing nonunique edges"""
    nodes_positions = np.array([
        [1,  1],
        [-1,  1],
        [-1, -1],
        [1, -1],
    ])
    edges_indices = np.array([
        [0, 1],
        [1, 2],
        [2, 1],  # repeated edge
        [2, 3],
        [3, 0],
        [0, 2]
    ])
    with pytest.raises(RuntimeError):
        lattice = Lattice(
            nodes_positions=nodes_positions,
            edges_indices=edges_indices,
            linear_stiffness=0.1,
            angular_stiffness=0.1
        )


def test_lattice_fails_wrong_dimension_nodes():
    """make sure we raise an error when passing nodes with wrong number of dimensions"""
    nodes_positions = np.array([
        [0, 0, 0, 1,  1],
        [0, 0, 0, -1,  1],
        [0, 0, 0, -1, -1],
        [0, 0, 0, 1, -1],
    ])
    edges_indices = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 2]
    ])
    with pytest.raises(RuntimeError):
        lattice = Lattice(
            nodes_positions=nodes_positions,
            edges_indices=edges_indices,
            linear_stiffness=0.1,
            angular_stiffness=0.1
        )


def test_lattice_fails_wrong_z():
    """make sure we raise an error when passing wrong z coordinates"""
    nodes_positions = np.array([
        [1,  1, 0],
        [-1,  1, 0],
        [-1, -1, 0],
        [1, -1, .2],  # z is not constant
    ])
    edges_indices = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 2]
    ])
    with pytest.raises(NotImplementedError):
        lattice = Lattice(
            nodes_positions=nodes_positions,
            edges_indices=edges_indices,
            linear_stiffness=0.1,
            angular_stiffness=0.1
        )


def test_lattice__store_edges_neighbours(lattice: Lattice):
    """make sure we correctly compute the neighbours of edges"""
    e01, e12, e23, e30, e02 = lattice._possible_edges
    assert e01._neighbouring_linear_springs == set([e12, e30, e02])
    assert e12._neighbouring_linear_springs == set([e01, e23, e02])
    assert e23._neighbouring_linear_springs == set([e12, e30, e02])
    assert e30._neighbouring_linear_springs == set([e01, e23, e02])
    assert e02._neighbouring_linear_springs == set([e01, e12, e23, e30])
