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
import numpy as np
import numpy.linalg as la

from typing import List
from itertools import cycle, islice


def coupon_collector_bound(num_different_coupons: int, sureness: float = 1e-3) -> int:
    r"""a lower bound to how many coupons you need to buy to
    to be sure you will see all the coupon collection.

    bound derived from
    https://en.wikipedia.org/wiki/Coupon_collector%27s_problem

    $$
    P\left [ T > \beta n \log n \right ] = P \left [ 	\bigcup_i {Z}_i^{\beta n \log n} \right ] \le n \cdot P [ {Z}_1^{\beta n \log n} ] \le n^{-\beta + 1}
    $$
    """
    float_bound = num_different_coupons * \
        (np.log(num_different_coupons) - np.log(sureness))
    int_bound = int(float_bound) + 1
    return int_bound


def reindex_nodes(nodes_indices, edges_indices):
    """make indexes to be consecutive"""
    nodes_indices = np.array(nodes_indices)
    edges_indices = np.array(edges_indices)
    translator = {
        v: k
        for k, v
        in dict(enumerate(nodes_indices)).items()
    }
    translated_edges_indices = np.vectorize(translator.get)(edges_indices)
    translated_nodes_indices = np.vectorize(translator.get)(nodes_indices)
    return translated_nodes_indices, translated_edges_indices


def three_points_to_angle(point_origin, point_start, point_end) -> float:
    """
    Signed angle between three points.

    Computes the signed angle between 3 points, with the result
    being always between 0 and 2 pi.

    Parameters
    ----------
    point_origin : [type]
        Central point of the angle.
    point_start : [type]
        Point where the angle starts.
    point_end : [type]
        Point where the angle ends.

    Returns
    -------
    float
        signed angle, in radians
    """
    return vectors_to_signed_angle(
        point_start - point_origin,
        point_end - point_origin
    )


def vectors_to_signed_angle(v_0, v_1) -> float:
    """
    Computes the signed angle from v0 to v1.
    Always between 0 and 2*pi

    Notes
    -----
    https://stackoverflow.com/questions/2150050/finding-signed-angle-between-vectors
    https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors/16544330#16544330
    """
    ax, ay = v_0
    bx, by = v_1
    # this angle is always between -pi and pi
    angle = np.arctan2(ax * by - ay * bx, ax * bx + ay * by)

    # we want an angle between 0 and 2pi
    if angle < 0:
        angle = angle + 2 * np.math.pi

    return angle


def circular_slice(mylist, element, count=1):
    """
    Slice a list circularly.

    It interprets the input list as a circle
    and slices it count elements to the right
    and count elements to the left.


    Parameters
    ----------
    mylist : list
        Input list, considered circular.
    element :
        One element of list around to which to slice.
    count : int, optional
        how many elements to slice around it.

    Returns
    -------
    list
        A sliced sublist of mylist.

    Example
    -------
    In[0]   circular_slice(
                mylist=['a', 'b', 'c'],
                element='a',
                count=1)
    Out[0]  ['d', 'a', 'b']
    """
    index = mylist.index(element) + len(mylist)
    aux = islice(cycle(mylist), index - count, index + count + 1)
    return list(aux)
