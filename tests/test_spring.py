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
Tests for metamech.spring

Unit tests for LinearSpring and AngularSpring base classes
"""
import pytest
from metamech.spring import LinearSpring
from metamech.spring import AngularSpring
from metamech.node import Node
from typing import Tuple
import numpy as np


def test_linear_spring_init():
    """test that we can initialize a node"""
    node0 = Node(x=0, y=0)
    node1 = Node(x=1, y=1)
    linear_spring = LinearSpring(
        nodes=(node0, node1),
        stiffness=0.1,
    )
    # check types
    assert isinstance(linear_spring, LinearSpring)
    assert isinstance(linear_spring._nodes, Tuple)
    assert isinstance(linear_spring.stiffness, float)
    # check storing
    _node0, _node1 = linear_spring._nodes
    _stiffness = linear_spring.stiffness
    assert node0 == _node0
    assert node1 == _node1
    assert _stiffness == 0.1


def test_linear_spring_get_resting_length():
    """test that we compute the correct resting length"""
    for (x0, y0, x1, y1) in [
        (0, 0, 0, 0),
        (0, 0, 1, 1),
        (1, 1, 1, 1,),
        (1.5, 2.3, 0, 6.7),
        (-1, 0, -1, 0),
        (-1, -4.5, 6.7, 100)
    ]:
        node0 = Node(x=x0, y=y0, label="node0")
        node1 = Node(x=x1, y=y1, label="node1")
        linear_spring = LinearSpring(
            nodes=(node0, node1),
            stiffness=0.1,
        )
        resting_length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        assert resting_length == linear_spring._get_resting_length()


def test_angular_spring_init():
    """test that we can instantiate an angular spring"""
    node0 = Node(x=0, y=0)
    node1 = Node(x=1, y=0)
    node2 = Node(x=1, y=1)
    angular_spring = AngularSpring(
        node_origin=node0,
        node_start=node1,
        node_end=node2,
        stiffness=0.15
    )
    # check types
    assert isinstance(angular_spring, AngularSpring)
    assert isinstance(angular_spring._node_origin, Node)
    assert isinstance(angular_spring._node_start, Node)
    assert isinstance(angular_spring._node_end, Node)
    assert isinstance(angular_spring.stiffness, float)
    # check storage
    assert angular_spring._node_origin == node0
    assert angular_spring._node_start == node1
    assert angular_spring._node_end == node2
    assert angular_spring.stiffness == 0.15


def test_angular_spring_get_resting_angle():
    """check that we compute the correct resting angle"""
    node0 = Node(x=0, y=0)
    node1 = Node(x=1, y=0)
    node2 = Node(x=1, y=1)
    # test spring at 45 deg
    angular_spring = AngularSpring(
        node_origin=node0,
        node_start=node1,
        node_end=node2,
        stiffness=0.15
    )
    assert angular_spring._get_resting_angle() == np.deg2rad(45)
    # same spring, other direction
    # angles are signed, anti-clockwise
    angular_spring = AngularSpring(
        node_origin=node0,
        node_start=node2,
        node_end=node1,
        stiffness=0.15
    )
    assert angular_spring._get_resting_angle() == np.deg2rad(360 - 45)
