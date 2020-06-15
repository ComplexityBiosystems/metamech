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
Tests for the base Actuator class.

"""
from metamech.actuator import Actuator
from metamech.lattice import Lattice

import numpy as np
import pytest
import copy


@pytest.fixture
def square_lattice():
    """a simple square with one diagonal"""
    nodes_positions = np.array([
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1]
    ]).astype(float)
    edges_indices = np.array([
        [0, 1],
        [1, 2],
        [0, 2],
        [2, 3],
        [3, 0]
    ]).astype(int)

    # create lattice
    lattice = Lattice(
        nodes_positions=nodes_positions,
        edges_indices=edges_indices,
        linear_stiffness=1,
        angular_stiffness=1
    )
    # activate all edges
    for edge in lattice._possible_edges:
        lattice.flip_edge(edge)

    return lattice


@pytest.fixture
def square_actuator(square_lattice: Lattice):
    """a actuator for the simple square"""
    lattice = square_lattice
    actuator = Actuator(
        lattice=lattice,
        frozen_nodes=[2, 3],
        input_nodes=[0],
        input_vectors=np.array([
            [0.1, 0.2]
        ]),
        output_nodes=[1],
        output_vectors=np.array([
            [0.3, 0.4]
        ]),
    )
    return actuator


"""
little trick: make a copy of the fixture to 
get non-shared instnaces
"""
square_lattice_clone = square_lattice


@pytest.fixture
def square_actuator_clone(square_lattice_clone: Lattice):
    """a actuator for the simple square"""
    lattice = square_lattice_clone
    actuator = Actuator(
        lattice=lattice,
        frozen_nodes=[2, 3],
        input_nodes=[0],
        input_vectors=np.array([
            [0.1, 0.2]
        ]),
        output_nodes=[1],
        output_vectors=np.array([
            [0.3, 0.4]
        ]),
    )
    return actuator


@pytest.fixture
def square_actuator_force(square_lattice: Lattice):
    """a actuator for the simple square"""
    lattice = square_lattice
    actuator = Actuator(
        lattice=lattice,
        frozen_nodes=[2, 3],
        input_nodes=[0],
        input_vectors=np.array([
            [0.1, 0.2]
        ]),
        output_nodes=[1],
        output_vectors=np.array([
            [0.3, 0.4]
        ]),
        method="force"
    )
    return actuator


def test_actuator_init(square_actuator: Actuator):
    """make sure we can instantiate a actuator"""
    actuator = square_actuator
    assert isinstance(actuator, Actuator)


def test_actuator_store(square_actuator: Actuator):
    """make sure we are storing things properly at init"""
    actuator = square_actuator
    assert isinstance(actuator.lattice, Lattice)
    assert actuator.frozen_nodes == [2, 3]
    assert actuator.input_nodes == [0]
    assert actuator.output_nodes == [1]
    assert np.all(actuator.input_vectors == [0.1, 0.2])
    assert np.all(actuator.output_vectors == [0.3, 0.4])


def test_actuator_nodes_exist(square_lattice: Lattice):
    """make sure we cannot pass inexistent nodes"""
    with pytest.raises(RuntimeError):
        actuator = Actuator(
            lattice=square_lattice,
            frozen_nodes=[0, 5, 8],
            input_nodes=[1],
            output_nodes=[2, 4],
            input_vectors=np.array([
                [1, 1]
            ]),
            output_vectors=np.array([
                [1, 1]
            ]),
        )


def test_actuator_nodes_unique(square_lattice: Lattice):
    """make sure we cannot pass non-unique lists of nodes"""
    with pytest.raises(RuntimeError):
        actuator = Actuator(
            lattice=square_lattice,
            frozen_nodes=[0, 1],
            input_nodes=[1, 2, 1],
            output_nodes=[2, 0],
            input_vectors=np.array([
                [1, 1]
            ]),
            output_vectors=np.array([
                [1, 1]
            ]),
        )


def test_actuator_nodes_overlap_frozen_input(square_lattice: Lattice):
    """make sure we cannot pass overlapping lists between frozen and input"""
    with pytest.raises(RuntimeError):
        actuator = Actuator(
            lattice=square_lattice,
            frozen_nodes=[2, 3],
            input_nodes=[2],
            input_vectors=np.array([
                [0.1, 0.2]
            ]),
            output_nodes=[1],
            output_vectors=np.array([
                [0.3, 0.4]
            ]),
        )


def test_actuator_nodes_overlap_frozen_output(square_lattice: Lattice):
    """make sure we cannot pass overlapping lists between frozen and output"""
    with pytest.raises(RuntimeError):
        actuator = Actuator(
            lattice=square_lattice,
            frozen_nodes=[2, 3],
            input_nodes=[0],
            input_vectors=np.array([
                [0.1, 0.2]
            ]),
            output_nodes=[2],
            output_vectors=np.array([
                [0.3, 0.4]
            ]),
        )


def test_actuator_nodes_overlap_input_output(square_lattice: Lattice):
    """make sure we cannot pass overlapping lists between input output"""
    with pytest.raises(RuntimeError):
        actuator = Actuator(
            lattice=square_lattice,
            frozen_nodes=[2, 3],
            input_nodes=[0],
            input_vectors=np.array([
                [0.1, 0.2]
            ]),
            output_nodes=[0],
            output_vectors=np.array([
                [0.3, 0.4]
            ]),
        )


def test_actuator_nodes_sizes_input(square_lattice: Lattice):
    """make sure we raise error when input sizes dont match"""
    with pytest.raises(RuntimeError):
        actuator = Actuator(
            lattice=square_lattice,
            frozen_nodes=[0, 1],
            input_nodes=[2],
            output_nodes=[3],
            input_vectors=np.array([
                [1, 1],
                [1, 1]
            ]),
            output_vectors=np.array([
                [1, 1]
            ]),
        )


def test_actuator_nodes_sizes_output(square_lattice: Lattice):
    """make sure we raise error when output sizes dont match"""
    with pytest.raises(RuntimeError):
        actuator = Actuator(
            lattice=square_lattice,
            frozen_nodes=[0, 1],
            input_nodes=[2],
            output_nodes=[3],
            input_vectors=np.array([
                [1, 1]
            ]),
            output_vectors=np.array([
                [1, 1],
                [1, 1],
            ]),
        )


def test_actuator_no_duplicates_in_frozen(square_lattice: Lattice):
    """make sure we cannot pass duplicated nodes to frozen"""
    with pytest.raises(RuntimeError):
        actuator = Actuator(
            lattice=square_lattice,
            frozen_nodes=[2, 2],
            input_nodes=[0],
            input_vectors=np.array([
                [0.1, 0.2]
            ]),
            output_nodes=[1],
            output_vectors=np.array([
                [0.3, 0.4]
            ]),
        )


def test_actuator_no_duplicates_in_input(square_lattice: Lattice):
    """make sure we cannot pass duplicated nodes to input"""
    with pytest.raises(RuntimeError):
        actuator = Actuator(
            lattice=square_lattice,
            frozen_nodes=[2, 3],
            input_nodes=[0, 0],
            input_vectors=np.array([
                [0.1, 0.2],
                [0.1, 0.2]
            ]),
            output_nodes=[1],
            output_vectors=np.array([
                [0.3, 0.4]
            ]),
        )


def test_actuator_no_duplicates_in_output(square_lattice: Lattice):
    """make sure we cannot pass duplicated nodes to output"""
    with pytest.raises(RuntimeError):
        actuator = Actuator(
            lattice=square_lattice,
            frozen_nodes=[2, 3],
            input_nodes=[0],
            input_vectors=np.array([
                [0.1, 0.2]
            ]),
            output_nodes=[1, 1],
            output_vectors=np.array([
                [0.3, 0.4],
                [0.3, 0.4],
            ]),
        )


def test_actuator_get_efficiency_displacement(square_lattice: Lattice):
    """make sure we correctly compute efficiencies
    for the displacement case (with just one vector)
    """
    for _ in range(100):
        for rescale_efficiency in [True, False]:
            # set a specific displacement
            disp_x = np.random.uniform(-2, 2)
            disp_y = np.random.uniform(-2, 2)
            # set a specific output
            out_x = np.random.uniform(-2, 2)
            out_y = np.random.uniform(-2, 2)
            # set a specific input
            inp_x = np.random.uniform(-2, 2)
            inp_y = np.random.uniform(-2, 2)
            # setup the actuator
            actuator = Actuator(
                lattice=square_lattice,
                frozen_nodes=[2, 3],
                input_nodes=[0],
                input_vectors=np.array([
                    [inp_x, inp_y]
                ]),
                output_nodes=[1],
                output_vectors=np.array([
                    [out_x, out_y]
                ]),
                rescale_efficiency=rescale_efficiency
            )
            # move the output node by hand
            actuator._relax_params["X"][actuator.output_nodes] = -1 + disp_x
            actuator._relax_params["Y"][actuator.output_nodes] = 1 + disp_y
            # compute the efficiency by hand
            numerator = disp_x * out_x + disp_y * out_y
            denominator = np.sqrt(out_x ** 2 + out_y ** 2)
            projection_length = numerator / denominator
            input_length = np.sqrt(inp_x ** 2 + inp_y ** 2)
            if rescale_efficiency:
                efficiency_by_hand = projection_length / input_length
            else:
                efficiency_by_hand = projection_length
            # compute with the actuators method
            efficiency_by_code = actuator._get_efficiency()
            # compare
            assert np.isclose(efficiency_by_code, efficiency_by_hand)


def test_actuator_deterministic_efficiency_displacement(square_actuator: Actuator):
    """test that several calls to .act() give the exact same efficiency
    in displacement mode
    """
    actuator = square_actuator
    efficiencies = []
    for _ in range(10):
        actuator.act()
        efficiencies.append(actuator.efficiency)
    assert len(set(efficiencies)) == 1


def test_actuator_deterministic_efficiency_force(square_actuator_force: Actuator):
    """test that several calls to .act() give the exact same efficiency
    in force mode
    """
    actuator = square_actuator_force
    efficiencies = []
    for _ in range(10):
        actuator.act()
        efficiencies.append(actuator.efficiency)
    assert len(set(efficiencies)) == 1


def test_actuator_rescale_efficiency_fails(square_lattice: Lattice):
    """make sure we can't use rescale efficiency when norms are not constant"""
    with pytest.raises(RuntimeError):
        actuator = Actuator(
            lattice=square_lattice,
            frozen_nodes=[3],
            input_nodes=[0, 2],
            input_vectors=np.array([
                [0.1, 0.1],
                [0.2, 0.2]
            ]),
            output_nodes=[1],
            output_vectors=np.array([
                [0.3, 0.4]
            ]),
            rescale_efficiency=True
        )

# tests for force-based protocol


def test_actuator_force_can_be_used(square_lattice: Lattice):
    """make sure force-based protocol can be called"""
    actuator = Actuator(
        lattice=square_lattice,
        frozen_nodes=[2, 3],
        input_nodes=[0],
        input_vectors=np.array([
            [0.1, 0.2]
        ]),
        output_nodes=[1],
        output_vectors=np.array([
            [0.3, 0.4],
        ]),
        method="force"
    )
    actuator.act()


def test_actuator_unkown_protocols(square_lattice: Lattice):
    """make sure cannot use unkown protocols"""
    with pytest.raises(RuntimeError):
        actuator = Actuator(
            lattice=square_lattice,
            frozen_nodes=[2, 3],
            input_nodes=[0],
            input_vectors=np.array([
                [0.1, 0.2]
            ]),
            output_nodes=[1],
            output_vectors=np.array([
                [0.3, 0.4],
            ]),
            method="unknown"
        )


@pytest.mark.parametrize("dummy", range(100))
def test_actuator_square_deterministic_act(square_actuator: Actuator, square_actuator_clone: Actuator, dummy: int):
    """test that potential energy is deterministically calculated

    Note
    ----
    This test is related to a bug found on May 2020 where the order by
    which teh angles are passed to relax can have a tiny but noticeable
    effect on the potential energy calculation (surprisingly).
    The bug led to non-deterministic computation of epot, and was
    fixed by substituting sets with ordered sets throught the whole module.

    Note 2
    ------
    Using a trick with dummy parameter to run the test many times. 
    This is to make sure that the test fails at least once, if angles order
    is not deterministic. 
    """
    actuator1 = square_actuator
    actuator1.act()
    epot1 = actuator1._relax_params["epot"]

    actuator2 = square_actuator_clone
    actuator2.act()
    epot2 = actuator2._relax_params["epot"]

    assert epot1 == epot2


def test_actuator_frozen_y_nodes(square_lattice):
    """make sure that we can add y-only contraints"""
    actuator = Actuator(
        lattice=square_lattice,
        frozen_nodes=[3],
        frozen_y_nodes=[2],
        input_nodes=[0],
        input_vectors=np.array([
            [-0.5, 0]
        ]),
        output_nodes=[1],
        output_vectors=np.array([
            [-1, 0]
        ]),
    )
    actuator.act()
    # grab x and y of semifrozen nodes
    x_displaced = actuator._relax_params["X"][2]
    y_displaced = actuator._relax_params["Y"][2]
    assert x_displaced != -1 and y_displaced == -1


def test_actuator_frozen_overlaps(square_lattice):
    """make sure that we cannot pass overlapping lists of frozen nodes"""
    with pytest.raises(RuntimeError):
        actuator = Actuator(
            lattice=square_lattice,
            frozen_nodes=[3, 2],
            frozen_x_nodes=[2],
            input_nodes=[0],
            input_vectors=np.array([
                [-0.5, 0]
            ]),
            output_nodes=[1],
            output_vectors=np.array([
                [-1, 0]
            ]),
        )


def test_actuator_frozen_x_nodes(square_lattice):
    """make sure that we can add x-only contraints"""
    actuator = Actuator(
        lattice=square_lattice,
        frozen_nodes=[3],
        frozen_x_nodes=[2],
        input_nodes=[0],
        input_vectors=np.array([
            [-0.5, 0]
        ]),
        output_nodes=[1],
        output_vectors=np.array([
            [-1, 0]
        ]),
    )
    actuator.act()
    # grab x and y of semifrozen nodes
    x_displaced = actuator._relax_params["X"][2]
    y_displaced = actuator._relax_params["Y"][2]
    assert x_displaced == -1 and y_displaced != -1
