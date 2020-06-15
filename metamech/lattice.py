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
Classes to hold regular and irregular configurations
"""
from .node import Node
from .spring import LinearSpring
from .io import _get_lammps_string

from typing import Optional
from typing import List
from typing import Tuple
from typing import Dict
from typing import Set
from typing import Union

import numpy as np
from itertools import combinations
from pathlib import Path
import sys

from ordered_set import OrderedSet


class Lattice:

    def __init__(
        self,
        nodes_positions: Dict[Tuple[int, int], float],
        edges_indices: Dict[Tuple[int, int], int],
        linear_stiffness: float = 10,
        angular_stiffness: float = 0.2
    ) -> None:
        """
        Create a lattice object.

        Create a lattice object from a list of nodes and
        a list of edges.

        Parameters
        ----------
        nodes_positions : np.ndarray
            Positions of nodes.
        edges_indices : np.ndarray, (n_edges, 2)
            Pairs of indices where edges can be placed.
        list_of_clamped_nodes : Optional[List], optional
            List of indices of nodes that cannot move, by default None
        linear_stiffness : float, optional
            Stiffness of linear springs, by default 0.1
        angular_stiffness : float, optional
            Stiffness of angular springs, by default 0.1
        """
        # type checking
        assert isinstance(nodes_positions, np.ndarray)
        assert isinstance(edges_indices, np.ndarray)
        # make sure there are no edge indices outside the range of nodes
        if not max(edges_indices.reshape(-1)) <= len(nodes_positions):
            raise RuntimeError(
                f"Array of edges indices contains {max(edges_indices.reshape(-1))}, which is more than the number of nodes {len(nodes_positions)}")
        # make sure all edges indices are positive
        if not min(edges_indices.reshape(-1)) >= 0:
            raise RuntimeError(
                f"Array of edges indices contains negative values"
            )
        # make sure all nodes are different
        if not len(np.unique(nodes_positions, axis=0)) == len(nodes_positions):
            raise RuntimeError(
                f"You are creating two nodes in the same position."
            )
        # make sure all edges are different
        if not len(np.unique(np.sort(edges_indices), axis=0)) == len(edges_indices):
            raise RuntimeError(
                f"You are creating two or more identical edges"
            )
        # store the input
        if not len(nodes_positions.T) in [2, 3]:
            raise RuntimeError(
                f"Nodes must be placed in 2 or 3 dimensional space. Check your node_positions array!"
            )
        # so far 3D not implemented, z coord is ignored (but we check it is constant)
        if len(nodes_positions.T) == 3:
            self._nodes_positions = nodes_positions[:, :2]
            z = nodes_positions[:, 2]
            if not len(set(z)) == 1:
                raise NotImplementedError(
                    f"You passed a non-constant 3rd coordinate. 3D not implementdd"
                )
        elif len(nodes_positions.T) == 2:
            self._nodes_positions = nodes_positions
        self._edges_indices = edges_indices
        self._linear_stiffness = linear_stiffness
        self._angular_stiffness = angular_stiffness

        # store basic info
        self.num_nodes = len(nodes_positions)
        self.num_possible_edges = len(edges_indices)

        # add the nodes
        self._add_nodes()

        # for the moment, stiffness is constant
        self._edge_stiffness = self._linear_stiffness * \
            np.ones(self.num_possible_edges)

        # generate bag of possible edges
        self._generate_possible_edges()

        # store neighbours of edges
        self._store_edges_neighbours()

        # empty set of edges
        self.edges: Set[LinearSpring] = OrderedSet()

        # set recursion limit large enough
        # to allow for deepcopy operation
        self._set_recursion_limit()

    def _set_recursion_limit(self):
        old_reclimit = sys.getrecursionlimit()
        new_reclimit = 10 * len(self._possible_edges)
        if new_reclimit > old_reclimit:
            sys.setrecursionlimit(new_reclimit)

    def to_lammps(self, path: Union[str, Path]) -> None:
        """
        Save Lattice instance as text LAMMPS file.

        Parameters
        ----------
        path : Union[str, Path]
            File path where to save the actuator.

        Raises
        ------
        RuntimeError
            If file already exists
        RuntimeError
            If parent dir does not exist
        """
        assert isinstance(path, (str, Path))
        assert isinstance(path, (str, Path))
        # make sure we are not overwriting
        path = Path(path)
        if path.is_file():
            raise RuntimeError(
                f"File {path} already exists, I refuse to overwrite!")
        # make sure directory exists
        if not path.parent.is_dir():
            raise RuntimeError(
                f"Parent of {path} does not exist. Create the directory!")

        # get the string representations
        lammps_string = _get_lammps_string(lattice=self)
        # write it to the file
        path.write_text(lammps_string)

    def flip_edge(self, edge):
        """conveniently add/remove an edge"""
        if edge in self.edges:
            self._remove_edge(edge)
        else:
            self._add_edge(edge)

    def _add_edge(self, edge):
        """add an edge to the configuration """
        assert edge not in self.edges
        self.edges.add(edge)
        n0, n1 = edge._nodes

        n0.add_neighbour(n1)
        n0.split_angular_spring(n1, angular_stiffness=self._angular_stiffness)

        n1.add_neighbour(n0)
        n1.split_angular_spring(n0, angular_stiffness=self._angular_stiffness)

    def _remove_edge(self, edge):
        """remove an edge from the configuration"""
        assert edge in self.edges
        self.edges.remove(edge)
        n0, n1 = edge._nodes

        n0.join_angular_springs(n1, angular_stiffness=self._angular_stiffness)
        n0.remove_neighbour(n1)

        n1.join_angular_springs(n0, angular_stiffness=self._angular_stiffness)
        n1.remove_neighbour(n0)

    def _add_nodes(self):
        """add the nodes to the configuration"""
        self.nodes = tuple(
            Node(x=x, y=y, clamped=False, label=idx)
            for idx, (x, y) in enumerate(self._nodes_positions)
        )

    def _generate_possible_edges(self):
        """create a bag of possible edges"""
        ii = self._edges_indices.T[0]
        jj = self._edges_indices.T[1]
        self._possible_edges = [
            LinearSpring(
                nodes=(self.nodes[i], self.nodes[j]), stiffness=stiffness)
            for i, j, stiffness
            in zip(ii, jj, self._edge_stiffness)
        ]

    def _store_edges_neighbours(self):
        """store the edge neighbours of each edge"""
        for edge1, edge2 in combinations(self._possible_edges, 2):
            a, b = edge1._nodes
            c, d = edge2._nodes
            if a == c or a == d or b == c or b == d:
                edge1._neighbouring_linear_springs.add(edge2)
                edge2._neighbouring_linear_springs.add(edge1)
