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
Actuator class, which holds togehter a lattice, the set of
input/output nodes and vectors, and the set of frozen nodes.
"""
from .lattice import Lattice
from .minimize import relax
from .minimize import _get_params_from_lattice
from .io import _get_lammps_string

from typing import Union
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from typing import Callable

import numpy as np
import copy
from pathlib import Path
import itertools


class Actuator:
    def __init__(self,
                 lattice: Lattice,
                 frozen_nodes: List[int],
                 input_nodes: List[int],
                 input_vectors: np.ndarray,
                 output_nodes: List[int],
                 output_vectors: np.ndarray,
                 method: str = "displacement",
                 output_spring_stiffness: float = .1,
                 max_iter: int = 1_000_000_000,
                 max_force: float = 1e-5,
                 efficiency_agg_fun: Callable = np.mean,
                 rescale_efficiency: bool = True,
                 frozen_x_nodes: List[int] = [],
                 frozen_y_nodes: List[int] = [],
                 ) -> None:
        # check that all labels passed in the lists of nodes exist
        existing_nodes = [
            node.label
            for node
            in lattice.nodes
        ]
        for node_idx in frozen_nodes + input_nodes + output_nodes:
            if node_idx not in existing_nodes:
                raise RuntimeError(
                    f"Node {node_idx} is not found in the lattice")

        # check that input, output and frozen do not have repeated elements
        if len(set(frozen_nodes)) != len(frozen_nodes):
            raise RuntimeError("The list of frozen nodes contains duplicates")
        if len(set(frozen_x_nodes)) != len(frozen_x_nodes):
            raise RuntimeError("The list of frozen nodes contains duplicates")
        if len(set(frozen_y_nodes)) != len(frozen_y_nodes):
            raise RuntimeError("The list of frozen nodes contains duplicates")
        if len(set(input_nodes)) != len(input_nodes):
            raise RuntimeError("The list of input nodes contains duplicates")
        if len(set(output_nodes)) != len(output_nodes):
            raise RuntimeError("The list of output nodes contains duplicates")

        # check that there is no overlap between in, out, frozen nodes,
        # frozen_y_nodes, frozen_x_nodes
        _str_to_set = {
            "input": set(input_nodes),
            "output": set(output_nodes),
            "frozen": set(frozen_nodes),
            "frozen_x_nodes": set(frozen_x_nodes),
            "frozen_y_nodes": set(frozen_y_nodes)
        }
        _sets_labels = list(_str_to_set.keys())
        for _set1_label, _set2_label in itertools.combinations(_sets_labels, 2):
            _set1 = _str_to_set[_set1_label]
            _set2 = _str_to_set[_set2_label]
            if not _set1.isdisjoint(_set2):
                raise RuntimeError(
                    f"Overlap between lists of {_set1_label} and {_set2_label} nodes")

        # make sure sizes match
        if len(input_vectors) != len(input_nodes):
            raise RuntimeError(
                "Inconsistent number of input_vectors and input_nodes")
        if len(output_vectors) != len(output_nodes):
            raise RuntimeError(
                "Inconsistent number of output_vectors and output_nodes")

        # if rescale efficiency is True,
        # make sure all input vectors are same length
        if rescale_efficiency:
            input_norms = np.sqrt(np.sum(input_vectors ** 2, axis=1))
            if not len(set(input_norms)) == 1:
                raise RuntimeError("Input vectors are not all of same length, ",
                                   "so I cannot compute an aggregated efficiency."
                                   "To proceed anyway, set rescale_efficiency to False"
                                   )
            else:
                # they are all equal at this point
                self._input_norm = input_norms[0]

        # store args
        self.lattice = lattice
        self.frozen_nodes = frozen_nodes
        self.frozen_y_nodes = frozen_y_nodes
        self.input_nodes = input_nodes
        self.input_vectors = input_vectors
        self.output_nodes = output_nodes
        self.output_vectors = output_vectors
        self.method = method
        self.output_spring_stiffness = output_spring_stiffness
        self.max_iter = max_iter
        self.max_force = max_force
        self._agg_fun = efficiency_agg_fun
        self._rescale_efficiency = rescale_efficiency

        # get ready to quickly call relax when needed
        # FROZEN CODES
        # 0 -- not frozen
        # 1 -- frozen both x and y
        # 2 -- frozen y only
        # 3 -- frozen x only
        frozen = []
        for d in range(lattice.num_nodes):
            if d in frozen_nodes:
                frozen.append(1)
            elif d in frozen_y_nodes:
                frozen.append(2)
            elif d in frozen_x_nodes:
                frozen.append(3)
            else:
                frozen.append(0)
        frozen = np.array(frozen).astype(np.intc)

        target_in = np.array(input_nodes).astype(np.intc)
        Nin = np.intc(len(target_in))
        # do not unpack, make copies to avoid mem problems
        Xin = np.array(input_vectors).astype(float).T[0].copy()
        Yin = np.array(input_vectors).astype(float).T[1].copy()
        Zin = np.zeros(Nin).astype(float)

        target_out = np.array(output_nodes).astype(np.intc)
        Nout = np.intc(len(target_out))
        # do not unpack, make copies to avoid mem problems
        Xout = np.array(output_vectors).astype(float).T[0].copy()
        Yout = np.array(output_vectors).astype(float).T[1].copy()

        Zout = np.zeros(Nout).astype(float)
        kout = output_spring_stiffness

        epot = np.array([0.])
        fmax = np.array([max_force])

        if self.method == "displacement":
            method = np.intc(0)
        elif self.method == "force":
            method = np.intc(1)
        else:
            raise RuntimeError(
                f"Method {method} not known. Available methods: 'displacement', 'force'")

        enforce2D = np.intc(1)
        error = np.array([0]).astype(np.intc)
        step_relax = np.array([max_iter]).astype(np.intc)

        self._actuator_params = {
            "frozen": frozen,
            "target_in": target_in, "Xin": Xin, "Yin": Yin, "Zin": Zin, "Nin": Nin,
            "target_out": target_out, "Xout": Xout, "Yout": Yout, "Zout": Zout, "Nout": Nout, "kout": kout,
            "epot": epot, "fmax": fmax, "method": method, "enforce2D": enforce2D, "error": error, "step_relax": step_relax
        }

        # act once at least, so that relaxed parameters exist
        self.act()

    def to_lammps(self, path: Union[str, Path]) -> None:
        """
        Save Actuator instance as text LAMMPS file.

        Uses custom convention to mark nodes (atoms in LAMMPS)
        as standard (1), frozen (2), input (3) or output (4)
        via atom type attributes.


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
        lammps_string = _get_lammps_string(actuator=self)
        # write it to the file
        path.write_text(lammps_string)

    def act(self) -> None:
        """
        Computed displaced configuration under certain force/displacement.

        Uses FIRE algorithm to find minimum of potential. Force/displacement
        are set at object instantiation.
        """
        # get the parameters that depend on the lattice
        self._lattice_params = _get_params_from_lattice(self.lattice)
        # put them together with those that depend on the actuator
        actuator_params = copy.deepcopy(self._actuator_params)
        self._relax_params = {**self._lattice_params, **actuator_params}
        # call relax
        relax(**self._relax_params)
        # update efficiency
        self.efficiency = self._get_efficiency()

    def _get_efficiency(self) -> float:
        if self.method == "displacement":
            efficiency = self._get_efficiency_displacement()
        elif self.method == "force":
            efficiency = self._get_efficiency_force()
        return efficiency

    def _get_efficiency_displacement(self) -> float:
        """
        Compute the efficiency of the actuator

        If more than one output vector was specified, aggregates
        efficiencies using a function (average by default)

        Parameters
        ----------
        agg_fun : Callable, optional
            Function to aggregate efficiency of different output vectors, by default np.mean
        """
        X0, Y0 = self.lattice._nodes_positions[self.output_nodes].T
        X1 = self._relax_params["X"][self.output_nodes]
        Y1 = self._relax_params["Y"][self.output_nodes]
        displacement = np.array([X1 - X0, Y1 - Y0])
        # projection of displacements onto output vectors
        _num = np.diag(np.dot(self.output_vectors, displacement))
        _denom = np.sqrt(np.sum(self.output_vectors ** 2, axis=1))
        displacement_proj = _num / _denom
        # divide by input vectors norm only if we know
        # that all input vectors are same length
        if self._rescale_efficiency:
            efficiencies = displacement_proj / self._input_norm
        else:
            efficiencies = displacement_proj
        # aggregate the efficiencies
        return self._agg_fun(efficiencies)

    def _get_efficiency_force(self) -> float:
        # displacement vectors
        X0, Y0 = self.lattice._nodes_positions[self.output_nodes].T
        X1 = self._relax_params["X"][self.output_nodes]
        Y1 = self._relax_params["Y"][self.output_nodes]
        displacement = np.array([X1 - X0, Y1 - Y0])
        # projection along output direction
        displacement_projected = np.diag(
            np.dot(self.output_vectors, displacement))
        # output_force
        spring_output_forces = -1 * displacement_projected * self.output_spring_stiffness
        output_forces = -1 * spring_output_forces
        # input_force
        input_forces = np.sqrt(np.sum(self.input_vectors ** 2, axis=1))
        return self._agg_fun(output_forces) / self._agg_fun(input_forces)

    def _get_displaced_actuator(self) -> "Actuator":
        # get the displaced node positions
        nodes_positions = np.array(
            [self._relax_params["X"], self._relax_params["Y"]]
        ).T
        # make a copy of the whole actuator
        displaced_actuator = copy.deepcopy(self)
        # move the nodes
        displaced_actuator.lattice._nodes_positions = nodes_positions
        for node, (x, y) in zip(displaced_actuator.lattice.nodes, nodes_positions):
            node.x = x
            node.y = y
        return displaced_actuator
