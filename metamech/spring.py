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
Base classes for linear and angular springs
"""
from .utils import three_points_to_angle

import numpy as np

from typing import Tuple
from typing import Set
from typing import Union
from typing import TYPE_CHECKING

from ordered_set import OrderedSet

if TYPE_CHECKING:
    from .node import Node


class LinearSpring:
    def __init__(
        self,
        nodes: Tuple["Node", "Node"],
        stiffness: float,
    ) -> None:
        """Linear spring between two nodes"""
        self._nodes = nodes
        self.stiffness = stiffness
        self.resting_length = self._get_resting_length()
        self._neighbouring_linear_springs: Set["LinearSpring"] = OrderedSet()

    def _get_resting_length(self) -> float:
        node_u, node_v = self._nodes
        ux, vx = node_u.x, node_v.x
        uy, vy = node_u.y, node_v.y
        r0 = np.sqrt((ux - vx) ** 2 + (uy - vy) ** 2)
        return r0


class AngularSpring:
    def __init__(
        self,
        node_origin: "Node",
        node_start: "Node",
        node_end: "Node",
        stiffness: float
    ) -> None:
        self._node_origin = node_origin
        self._node_start = node_start
        self._node_end = node_end
        self.stiffness = stiffness
        self.resting_angle = self._get_resting_angle()

    def _get_resting_angle(self) -> float:
        angle = three_points_to_angle(
            point_origin=np.array([self._node_origin.x, self._node_origin.y]),
            point_start=np.array([self._node_start.x, self._node_start.y]),
            point_end=np.array([self._node_end.x, self._node_end.y]),
        )
        return angle
