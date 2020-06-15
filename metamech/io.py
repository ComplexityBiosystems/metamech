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
Functions to read/write data
"""
from typing import Optional
from typing import Dict
from typing import List
from typing import Union
from pathlib import Path
import numpy as np
from .utils import reindex_nodes
from .spring import LinearSpring
from .spring import AngularSpring
from .node import Node
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .lattice import Lattice
    from .actuator import Actuator


KNOWN_KEYWORDS = ["Masses", "Bond Coeffs",
                  "Angle Coeffs", "Atoms", "Bonds", "Angles"]


def _get_lammps_string(
    actuator: Optional["Actuator"] = None,
    lattice: Optional["Lattice"] = None
) -> str:
    if actuator is None and lattice is None:
        raise RuntimeError("You must either pass an actuator or a lattice")
    elif actuator is not None and lattice is not None:
        raise RuntimeError("You cannot pass both the actuator and the lattice")
    # correct situation here
    elif lattice is None and actuator is not None:
        lattice = actuator.lattice
    elif actuator is None and lattice is not None:
        pass
    # prepare lists, avoid using sets!
    nodes = list(lattice.nodes)
    node_types: Optional[List[int]]
    # prepare list of node types
    if actuator is not None:
        node_types = []
        for node in nodes:
            node_idx = node.label
            if node_idx in actuator.frozen_nodes:
                node_types.append(2)
            elif node_idx in actuator.input_nodes:
                node_types.append(3)
            elif node_idx in actuator.output_nodes:
                node_types.append(4)
            else:
                node_types.append(1)
        num_atom_types = len(set(node_types))
    else:
        node_types = None
        num_atom_types = 1
    edges = list(lattice.edges)
    angles = [
        angular_spring
        for node in lattice.nodes
        for angular_spring in node.angular_springs
    ]
    # compose lammps string
    lammps_string = \
        _write_header(lattice, num_atom_types=num_atom_types) + \
        _write_atoms_section(nodes=nodes, node_types=node_types) + \
        _write_masses_section(num_atom_types=num_atom_types) +  \
        _write_bonds_section(edges) + \
        _write_bond_coeffs_section(edges) + \
        _write_angles_section(angles) +  \
        _write_angle_coeffs_section(angles)
    return lammps_string


def _write_atoms_section(nodes: List[Node], node_types: Optional[List[int]] = None) -> str:
    """Create string representation of nodes of a lattice, LAMMPS syntax."""
    header = """
Atoms

"""
    # if don't know type of node use type
    # one for all nodes
    if node_types is None:
        node_types = np.ones(len(nodes))
    # otherwise check that things make sense
    # so far we implement 4 types of nodes only
    else:
        assert len(node_types) == len(nodes)
        assert min(node_types) >= 1 and max(node_types) <= 4
    lines = [
        f"{node.label + 1} {node_type} {node.x} {node.y} 0"
        for node, node_type in zip(nodes, node_types)
    ]
    section = header + "\n".join(lines) + "\n"
    return section


def _write_masses_section(num_atom_types: int = 1) -> str:
    """create a string represetnation of the masses info in LAMMPS syntax"""
    header = """
Masses

"""
    lines = [
        f"{atom_type} 1.0"
        for atom_type in range(1, num_atom_types + 1)
    ]
    section = header + "\n".join(lines) + "\n"
    return section


def _write_bonds_section(edges: List[LinearSpring]) -> str:
    """Create string representation of bonds of a lattice using LAMMPS syntax."""
    header = """
Bonds

"""
    lines = [
        f"{idx + 1} {idx + 1} {edge._nodes[0].label + 1} {edge._nodes[1].label + 1}"
        for idx, edge in enumerate(edges)
    ]
    section = header + "\n".join(lines) + "\n"
    return section


def _write_bond_coeffs_section(edges: List[LinearSpring]) -> str:
    """Create string representation of bonds resting legnths of a lattice using LAMMPS syntax."""
    header = """
Bond Coeffs

"""
    lines = [
        f"{idx + 1} {edge.stiffness} {edge.resting_length}"
        for idx, edge in enumerate(edges)
    ]
    section = header + "\n".join(lines) + "\n"
    return section


def _write_angles_section(angles: List[AngularSpring]) -> str:
    """Create string representation of angles of a lattice using LAMMPS syntax."""
    header = """
Angles

"""
    lines = [
        f"{idx + 1} {idx + 1} {angle._node_start.label + 1} {angle._node_origin.label + 1} {angle._node_end.label + 1}"
        for idx, angle in enumerate(angles)
    ]
    section = header + "\n".join(lines) + "\n"
    return section


def _write_angle_coeffs_section(angles: List[AngularSpring]) -> str:
    """Create string representation of angles coefficients of a lattice using LAMMPS syntax."""
    header = """
Angle Coeffs

"""
    lines = [
        f"{idx + 1} {angle.stiffness} {np.rad2deg(angle.resting_angle)}"
        for idx, angle in enumerate(angles)
    ]
    section = header + "\n".join(lines) + "\n"
    return section


def _write_header(lattice: "Lattice", num_atom_types: int = 1) -> str:
    """create string for the header in lammps format"""
    num_atoms = lattice.num_nodes
    num_bonds = len(lattice.edges)
    num_angles = sum(len(node.angular_springs) for node in lattice.nodes)
    # average bond length to get an idea of the scale of the lattice
    unit_length = np.mean([edge.resting_length for edge in lattice.edges])
    X, Y = np.array([
        [node.x, node.y]
        for node in lattice.nodes
    ]).T
    xlo, xhi = min(X) - unit_length, max(X) + unit_length
    ylo, yhi = min(Y) - unit_length, max(Y) + unit_length
    header = f"""
{num_atoms} atoms
{num_bonds} bonds
{num_angles} angles

{num_atom_types} atom types
{num_bonds} bond types
{num_angles} angle types

{xlo} {xhi} xlo xhi
{ylo} {yhi} ylo yhi
-1 1 zlo zhi
"""
    return header


def read_lammps(path: Union[str, Path]) -> Dict:
    if isinstance(path, str):
        path = Path(path)
    lines = path.read_text(encoding="utf8").split("\n")

    # read the atoms
    lines_atoms = _get_section(lines, "Atoms")
    _nodes_positions = []
    _nodes_indices = []
    input_nodes = []
    output_nodes = []
    frozen_nodes = []
    for line in lines_atoms:
        try:
            idx, node_type, x, y, z = line.split()
        except:
            idx, node_type, _, x, y, z = line.split()
        if node_type == "2":
            frozen_nodes.append(int(idx) - 1)
        elif node_type == "3":
            input_nodes.append(int(idx) - 1)
        elif node_type == "4":
            output_nodes.append(int(idx) - 1)
        else:
            assert node_type == "1"
        _nodes_positions.append([x, y, z])
        _nodes_indices.append(int(idx) - 1)
    nodes_positions = np.array(_nodes_positions).astype(float)
    nodes_indices = np.array(_nodes_indices)

    # read the bonds
    lines_bonds = _get_section(lines, "Bonds")
    _edges_indices = []
    for line in lines_bonds:
        _, _, i, j = line.split()
        _edges_indices.append([int(i) - 1, int(j) - 1])
    edges_indices = np.array(_edges_indices)

    reindexed_nodes_indices, reindexed_edges_indices = reindex_nodes(
        nodes_indices=nodes_indices,
        edges_indices=edges_indices
    )

    params = {
        "nodes_positions": nodes_positions,
        "edges_indices": reindexed_edges_indices,
        "input_nodes": input_nodes,
        "output_nodes": output_nodes,
        "frozen_nodes": frozen_nodes
    }
    return params


def _get_section(lines: List[str], keyword) -> List[str]:
    """select a subset of lines based on a header keyword"""
    if keyword not in KNOWN_KEYWORDS:
        raise RuntimeError(
            f"Keyword {keyword} not known. Should be one of {KNOWN_KEYWORDS}")
    num_lines = len(lines)
    keyword_length = len(keyword)
    for i in range(num_lines):
        if lines[i].strip()[:keyword_length] == keyword:
            assert lines[i+1].strip() == ""
            assert lines[i+2].strip() != ""
            break
    if i == num_lines - 1:
        raise RuntimeError(f"Keyword '{keyword}'' not found in input file")
    for j in range(i+3, num_lines):
        if lines[j].strip() == "":
            break

    return lines[i + 2:j]
