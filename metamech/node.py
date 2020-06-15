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
Base classes for nodes.

"""
from .utils import vectors_to_signed_angle
from .utils import circular_slice

from .spring import AngularSpring

from typing import Optional
from typing import List
from typing import Set
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np

from ordered_set import OrderedSet


class Node:
    def __init__(
            self,
            x: float,
            y: float,
            clamped: bool = False,
            label: Optional[Union[int, str]] = None
    ) -> None:
        """Initialize a node.

        Parameters
        ----------
        x : float
            Initial x-coordinate of the node.
        y : float
            Initial y-coordinate of the node.
        clamped : bool
            Set to True if the node cannot move, by default False.
        label : Union[str, None], optional
            Label to identify the node, by default None
        """
        # save info
        self.x = x
        self.y = y
        self.clamped = clamped
        if label is not None:
            self.label = label

        # empty set of neighbours
        self.neighbours: Set["Node"] = OrderedSet()

        # empty list of anti-clock-wise ordered neighbours
        self._ordered_neighbours: List["Node"] = []
        self._ordered_neighbours_angles: List[float] = []

        # empty set and dict for angular springs
        self.angular_springs: Set[AngularSpring] = OrderedSet()
        self._angular_springs_dict: Dict[Tuple["Node",
                                               "Node"], AngularSpring] = dict()

    def add_neighbour(self, node: "Node") -> None:
        """Add a neighbour to the node.

        Adds the node to the set of neighbours 
        and reorders the list of ordered neighbours.

        Parameters
        ----------
        node : Node
            The neighbour to add.
        """
        assert node not in self.neighbours
        self.neighbours.add(node)
        self._add_to_ordered_neighbours(node)

    def remove_neighbour(self, node: "Node") -> None:
        """Remove a neighbour from the node.

        Removes the node from the set of neighbours
        and from the list of ordered neighbours.

        Parameters
        ----------
        node : Node
            The neighbour to remove.
        """
        assert node in self.neighbours
        self.neighbours.remove(node)
        self._remove_from_ordered_neighbours(node)

    def split_angular_spring(self, node: "Node", angular_stiffness: float) -> None:
        """
        Split an angular spring into two via a new node.

        For instance, the angle BOA would split into BON and NOA


        Parameters
        ----------
        node : Node
            New node used to split the angular spring.
        angular_stiffness : float
            Stiffness of the two angular springs being created.
        """
        # remove the spring to be split
        before, after = self._get_adjacent_neighbours(node)
        # if after equals before, there is no spring to remove
        if after != before:
            angular_spring_to_remove = self._angular_springs_dict[(
                before, after)]
            self.angular_springs.remove(angular_spring_to_remove)
            del self._angular_springs_dict[(before, after)]
        # add two new angular springs
        # if node = before or node = after, nothing to add
        if node != before:
            angular_spring_to_add = AngularSpring(
                node_origin=self,
                node_start=before,
                node_end=node,
                stiffness=angular_stiffness
            )
            self._angular_springs_dict[(before, node)] = angular_spring_to_add
            self.angular_springs.add(angular_spring_to_add)
        if node != after:
            angular_spring_to_add = AngularSpring(
                node_origin=self,
                node_start=node,
                node_end=after,
                stiffness=angular_stiffness
            )
            self._angular_springs_dict[(node, after)] = angular_spring_to_add
            self.angular_springs.add(angular_spring_to_add)

    def join_angular_springs(self, node: "Node", angular_stiffness: float):
        # remove the two springs to be joined
        before, after = self._get_adjacent_neighbours(node)
        # from before to node
        if node != before:
            angular_spring_to_remove = self._angular_springs_dict[(
                before, node)]
            self.angular_springs.remove(angular_spring_to_remove)
            del self._angular_springs_dict[(before, node)]
        # from node to after
        if node != after:
            angular_spring_to_remove = self._angular_springs_dict[(
                node, after)]
            self.angular_springs.remove(angular_spring_to_remove)
            del self._angular_springs_dict[(node, after)]
        # add a new spring
        if after != before:
            angular_spring_to_add = AngularSpring(
                node_origin=self,
                node_start=before,
                node_end=after,
                stiffness=angular_stiffness
            )
            self._angular_springs_dict[(before, after)] = angular_spring_to_add
            self.angular_springs.add(angular_spring_to_add)

    # private methods
    def _add_to_ordered_neighbours(self, node):
        """
        Add the node to the list of ordered neighbours.

        To do so, compute the angle it forms wrt x-axis, 
        then resort all neighbours (whose angles we keep).
        """
        # get the angle of the new neighbour
        angle = self._get_angle(node)
        # add thhe node and the angle to the sorted lists
        # thus losing the correct ordering
        self._ordered_neighbours_angles.append(angle)
        self._ordered_neighbours.append(node)
        # reorder both lists, sorting via angles
        neighbours_order = np.argsort(self._ordered_neighbours_angles)
        self._ordered_neighbours_angles = [
            self._ordered_neighbours_angles[i]
            for i in neighbours_order
        ]
        self._ordered_neighbours = [
            self._ordered_neighbours[i]
            for i in neighbours_order
        ]

    def _remove_from_ordered_neighbours(self, node):
        """remove a node from the lists of ordered neighbours and their angles"""
        # get the index of the neiughbour we are removing
        idx = self._ordered_neighbours.index(node)
        # remove it from both lists
        del self._ordered_neighbours_angles[idx]
        del self._ordered_neighbours[idx]

    def _get_adjacent_neighbours(self, node):
        """get the previous and the next nodes in the list of ordered neighbours"""
        # make sure node is a neighbour of self
        assert node in self.neighbours
        # slice the list of ordered neighbours
        before, _, after = circular_slice(
            mylist=self._ordered_neighbours,
            element=node,
            count=1
        )
        return before, after

    def _get_angle(self, other: "Node"):
        """compute the angle that two nodes form
        with the x-axis. It is measured with
        me in the origin and other forming the angle.
             other
            /
          me -- x-axis
        """
        point_origin = np.array([self.x, self.y])
        point_end = np.array([other.x, other.y])
        v0 = np.array([1, 0])
        v1 = point_end - point_origin
        angle = vectors_to_signed_angle(v0, v1)
        return angle
