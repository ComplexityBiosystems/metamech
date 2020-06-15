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
Tests for metamech.node library.

Mainly testing Node class and its methods.
"""
from metamech.node import Node
import pytest
from numpy import deg2rad


@pytest.fixture
def test_node():
    node = Node(
        x=1.2,
        y=-3.4,
        clamped=True,
        label="test_node"
    )
    return node


def test_node_init(test_node: Node):
    """test that we can initialize a node"""
    assert isinstance(test_node, Node)
    assert test_node.x == 1.2
    assert test_node.y == -3.4
    assert test_node.clamped is True
    assert test_node.label == "test_node"


def test_node_add_neighbour(test_node: Node):
    """test that we add nodes"""
    other = Node(x=1.5, y=0)
    test_node.add_neighbour(other)
    assert other in test_node.neighbours
    assert other in test_node._ordered_neighbours


def test_node_order_neighbours():
    """test that we order neighbours correctly"""
    node0 = Node(x=0, y=0, label="node0")
    node1 = Node(x=1, y=0, label="node1")
    node2 = Node(x=1, y=1, label="node2")
    node3 = Node(x=0, y=1, label="node3")
    # add the nodes out of order
    node0.add_neighbour(node3)
    node0.add_neighbour(node1)
    node0.add_neighbour(node2)
    # check that it keeps them ordered
    assert ["node1", "node2", "node3"] == [
        node.label for node in node0._ordered_neighbours]


def test_node_remove_neighbour(test_node: Node):
    """test that we correctly remove nodes from neighbours"""
    other = Node(x=1.5, y=0)
    test_node.add_neighbour(other)
    test_node.remove_neighbour(other)
    assert other not in test_node.neighbours
    assert other not in test_node._ordered_neighbours


def test_node_get_adjacent_neighbours():
    """test that we retrieve the right neighbours"""
    node0 = Node(x=0, y=0, label="node0")
    node1 = Node(x=1, y=0, label="node1")
    node2 = Node(x=1, y=1, label="node2")
    node3 = Node(x=0, y=1, label="node3")
    # add a single node
    node0.add_neighbour(node1)
    before, after = node0._get_adjacent_neighbours(node1)
    assert before == node1
    assert after == node1
    # add a second node
    node0.add_neighbour(node2)
    before, after = node0._get_adjacent_neighbours(node2)
    assert before == node1
    assert after == node1
    # add a third node
    node0.add_neighbour(node3)
    before, after = node0._get_adjacent_neighbours(node2)
    assert before == node1
    assert after == node3


def test_get_angle():
    """test that we can measure angles between nodes"""
    node0 = Node(x=0, y=0, label="node0")
    node1 = Node(x=1, y=0, label="node1")
    node2 = Node(x=1, y=1, label="node2")
    # test a few angles
    assert node0._get_angle(node1) == deg2rad(0)
    assert node0._get_angle(node2) == deg2rad(45)
    assert node1._get_angle(node0) == deg2rad(180)
    assert node2._get_angle(node0) == deg2rad(180 + 45)
