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
Tests for metamech.utils library.
"""
from metamech.utils import three_points_to_angle
from metamech.utils import vectors_to_signed_angle
from metamech.utils import circular_slice
from metamech.utils import reindex_nodes
from metamech.utils import coupon_collector_bound
import numpy as np


def test_coupon_collector_bound():
    """test the coupon collector formula"""
    for n in [2, 5, 12]:
        for s in [1e-2, 1e-3, 1e-5]:
            assert coupon_collector_bound(
                num_different_coupons=n,
                sureness=s
            ) == int((np.log(n) - np.log(s)) * n) + 1


def test_utils_reindex_nodes():
    nodes_indices = np.array([3, 5, 4, 9])
    edges_indices = np.array([[3, 5], [5, 9]])
    expected_nodes_indices = np.array([0, 1, 2, 3])
    expected_edges_indices = np.array([[0, 1], [1, 3]])
    re_nodes_indices, re_edges_indices = reindex_nodes(
        nodes_indices=nodes_indices,
        edges_indices=edges_indices
    )
    assert np.all(re_nodes_indices == expected_nodes_indices)
    assert np.all(re_edges_indices == expected_edges_indices)


def test_utils_three_points_to_angle():
    """test that we correctly compute some angles"""

    # fourty-five degrees
    p_origin = np.array([0, 0])
    p_start = np.array([1, 1])
    p_end = np.array([0, 1])
    angle_rad = three_points_to_angle(
        point_origin=p_origin,
        point_start=p_start,
        point_end=p_end
    )
    angle_deg = np.rad2deg(angle_rad)
    assert angle_deg == 45

    # ninety degrees
    p_origin = np.array([0, 0])
    p_start = np.array([1, 0])
    p_end = np.array([0, 1])
    angle_rad = three_points_to_angle(
        point_origin=p_origin,
        point_start=p_start,
        point_end=p_end
    )
    angle_deg = np.rad2deg(angle_rad)
    assert angle_deg == 90

    # 135 degrees
    p_origin = np.array([0, 0])
    p_start = np.array([1, 0])
    p_end = np.array([-1, 1])
    angle_rad = three_points_to_angle(
        point_origin=p_origin,
        point_start=p_start,
        point_end=p_end
    )
    angle_deg = np.rad2deg(angle_rad)
    assert angle_deg == 135

    # 180 degrees
    p_origin = np.array([0, 0])
    p_start = np.array([1, 0])
    p_end = np.array([-1, 0])
    angle_rad = three_points_to_angle(
        point_origin=p_origin,
        point_start=p_start,
        point_end=p_end
    )
    angle_deg = np.rad2deg(angle_rad)
    assert angle_deg == 180

    # 270 degrees
    p_origin = np.array([0, 0])
    p_start = np.array([1, 0])
    p_end = np.array([0, -1])
    angle_rad = three_points_to_angle(
        point_origin=p_origin,
        point_start=p_start,
        point_end=p_end
    )
    angle_deg = np.rad2deg(angle_rad)
    assert angle_deg == 270


def test_utils_vectors_to_signed_angle():
    """test some angles, make sure they are positive"""
    v0 = np.array([1, 0])
    v1 = np.array([1, 1])

    # 45 degrees, v0 to v1
    angle_rad = vectors_to_signed_angle(v0, v1)
    angle_deg = np.rad2deg(angle_rad)
    assert angle_deg == 45

    # 315 degrees, v1 to v0
    angle_rad = vectors_to_signed_angle(v1, v0)
    angle_deg = np.rad2deg(angle_rad)
    assert angle_deg == 315


def test_utils_vectors_to_signed_angle_no_twopi_equal_vectors():
    """make sure that the angle betwene a vector an itself is 0 and not twopi"""
    for vx, vy in [(1, 1), (2, 3), (-1, 1), (2.33, 5.6), (-1.5, 9), (0.001, -0.4)]:
        vector = np.array([vx, vy])
        angle = vectors_to_signed_angle(vector, vector)
        assert np.isclose(angle, 0)


def test_utils_vectors_to_signed_angle_no_twopi_similar_vectors():
    """
    make sure that we get a small angle and not twopi between
    two vectors that form a small angle.
    """
    for small_angle in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        # prepare two vectors u, v
        # such that they form a very small
        # angle from u to v
        cos, sin = np.cos(small_angle), np.sin(small_angle)
        ux, uy = (1, 1)
        vx = ux * cos - uy * sin
        vy = ux * sin + uy * cos
        u = np.array([ux, uy])
        v = np.array([vx, vy])
        # recover the angle
        angle = vectors_to_signed_angle(v_0=u, v_1=v)
        # make sure its very small, and not twopi
        assert np.isclose(angle, small_angle)


def test_circular_slice():
    """test circular slicing with count=1 only"""
    mylist = ["a", "b", "c", "d"]
    assert circular_slice(mylist, "a", count=1) == ["d", "a", "b"]
    assert circular_slice(mylist, "b", count=1) == ["a", "b", "c"]
    assert circular_slice(mylist, "c", count=1) == ["b", "c", "d"]
    assert circular_slice(mylist, "d", count=1) == ["c", "d", "a"]
