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
Python wrapper for the C shared library mylib
"""
import numpy as np
import sys
import platform
import ctypes
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from .lattice import Lattice


if platform.system() == "Darwin":
    mylib_path = Path(__file__).parent / "./minimize.dylib"
elif platform.system() == "Linux":
    mylib_path = Path(__file__).parent / "./minimize.sso"

try:
    mylib = ctypes.CDLL(str(mylib_path))
except OSError:
    print("Unable to load the system C library")
    sys.exit()


# pointer to double
c_double_p = ctypes.POINTER(ctypes.c_double)
# pointer to int
c_int_p = ctypes.POINTER(ctypes.c_int)
# int
c_int = ctypes.c_int
# double
c_double = ctypes.c_double

# load shared library
_relax = mylib.relax
_relax.argtypes = [
    # nodes
    c_double_p,                    # double *X
    c_double_p,                    # double *Y
    c_double_p,                    # double *Z
    c_int_p,                       # int *frozen
    c_int,                         # int nnodes
    # bonds
    c_int_p,                       # int *ipart2
    c_int_p,                       # int *jpart2
    c_double_p,                    # double *r0
    c_double_p,                    # double *kbond
    c_int,                         # int nbonds
    # angles
    c_int_p,                       # int *ipart3
    c_int_p,                       # int *jpart3
    c_int_p,                       # int *kpart3
    c_double_p,                    # double *a0
    c_double_p,                    # double *kangle
    c_int,                         # int nangles
    # input perturbation
    c_int_p,                       # int *target_in
    c_double_p,                    # double *Xin
    c_double_p,                    # double *Yin
    c_double_p,                    # double *Zin
    c_int,                         # int Nin
    # desired output
    c_int_p,                       # int *target_out
    c_double_p,                    # double *Xout
    c_double_p,                    # double *Yout
    c_double_p,                    # double *Zout
    c_int,                         # int Nout
    c_double,                      # double kout
    # other info
    c_double_p,                    # double *epot
    c_double_p,                    # double *fmax
    c_int,                         # int method
    c_int,                         # int enforce2D
    c_int_p,                       # int *error
    c_int_p,                       # int *step_relax
]
_relax.restype = None


def relax(
    X, Y, Z, frozen, nnodes,
    ipart2, jpart2, r0, kbond, nbonds,
    ipart3, jpart3, kpart3, a0, kangle, nangles,
    target_in, Xin, Yin, Zin, Nin,
    target_out, Xout, Yout, Zout, Nout, kout,
    epot, fmax, method, enforce2D, error, step_relax
) -> None:
    _relax(
        X.ctypes.data_as(c_double_p),
        Y.ctypes.data_as(c_double_p),
        Z.ctypes.data_as(c_double_p),
        frozen.ctypes.data_as(c_int_p),
        c_int(nnodes),
        ipart2.ctypes.data_as(c_int_p),
        jpart2.ctypes.data_as(c_int_p),
        r0.ctypes.data_as(c_double_p),
        kbond.ctypes.data_as(c_double_p),
        c_int(nbonds),
        ipart3.ctypes.data_as(c_int_p),
        jpart3.ctypes.data_as(c_int_p),
        kpart3.ctypes.data_as(c_int_p),
        a0.ctypes.data_as(c_double_p),
        kangle.ctypes.data_as(c_double_p),
        c_int(nangles),
        target_in.ctypes.data_as(c_int_p),
        Xin.ctypes.data_as(c_double_p),
        Yin.ctypes.data_as(c_double_p),
        Zin.ctypes.data_as(c_double_p),
        c_int(Nin),
        target_out.ctypes.data_as(c_int_p),
        Xout.ctypes.data_as(c_double_p),
        Yout.ctypes.data_as(c_double_p),
        Zout.ctypes.data_as(c_double_p),
        c_int(Nout),
        c_double(kout),
        epot.ctypes.data_as(c_double_p),
        fmax.ctypes.data_as(c_double_p),
        c_int(method),
        c_int(enforce2D),
        error.ctypes.data_as(c_int_p),
        step_relax.ctypes.data_as(c_int_p)
    )

# some utility functions to call relax


def _get_params_from_lattice(lattice: Lattice) -> Dict:
    """
    Get relaxation parameters from a lattice, and cast 
    them to ctype-ready types.

    Convenience function to extract the parameters
    needed by relax function from a lattice instance.
    currently for 2D lattices only.

    Parameters
    ----------
    lattice : Lattice
        Instance of lattice class.

    Returns
    -------
    Dict
        Parameters used by relax function.
    """
    # node-related parameters
    X = lattice._nodes_positions.T[0].copy()
    Y = lattice._nodes_positions.T[1].copy()
    Z = np.zeros(len(X)).astype(float)
    nnodes = lattice.num_nodes

    # linear-spring related parameters
    _ipart2, _jpart2, _r0 = np.array([[
        edge._nodes[0].label,
        edge._nodes[1].label,
        edge.resting_length
    ] for edge in lattice.edges]).T
    # notice we DO need to convert to intc type
    ipart2 = _ipart2.astype(np.intc)
    jpart2 = _jpart2.astype(np.intc)  # same
    r0 = _r0.astype(float)
    kbond = np.array([edge.stiffness for edge in lattice.edges])
    nbonds = len(lattice.edges)

    # angular-spring related parameters
    _ipart3, _jpart3, _kpart3, _a0, _kangle = np.array([
        [
            spring._node_origin.label,
            spring._node_start.label,
            spring._node_end.label,
            spring.resting_angle,
            spring.stiffness,
        ]
        for node in lattice.nodes
        for spring in node.angular_springs
    ]).T
    ipart3 = _ipart3.astype(np.intc)
    jpart3 = _jpart3.astype(np.intc)
    kpart3 = _kpart3.astype(np.intc)
    a0 = _a0.astype(float)

    kangle = _kangle.astype(float)
    nangles = len([
        ""
        for node in lattice.nodes
        for spring in node.angular_springs
    ])

    # put all parameters in a dict
    params = {
        "X": X,
        "Y": Y,
        "Z": Z,
        "nnodes": nnodes,
        "ipart2": ipart2,
        "jpart2": jpart2,
        "r0": r0,
        "kbond": kbond,
        "nbonds": nbonds,
        "ipart3": ipart3,
        "jpart3": jpart3,
        "kpart3": kpart3,
        "a0": a0,
        "kangle": kangle,
        "nangles": nangles
    }
    return params
