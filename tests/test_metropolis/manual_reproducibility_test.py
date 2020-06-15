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

from metamech.metropolis import Metropolis
import numpy as np
from pathlib import Path


def replicate_run():
    # load the pickled output
    metro_old = Metropolis.from_pickle(
        Path(__file__).parent / "run_output" / "amorphous_1000_to_reproduce.p")
    # load what we need from the run to
    # be reproduced
    np_random_state = metro_old._initial_rng_state
    run_calls = metro_old._run_calls_history
    initial_actuator = metro_old._initial_actuator
    expected_efficiency = metro_old.actuator.efficiency
    # create a new instance of Metropolis
    metro_new = Metropolis(
        actuator=initial_actuator
    )
    # set the numpy RNG state
    np.random.set_state(np_random_state)
    # call run just like it was logged
    for run_call in run_calls:
        del run_call["self"]
        metro_new.run(**run_call)
    return metro_new


if __name__ == "__main__":
    """
    Replicate a run.

    Read a pickled metropolis file and replicate it from scratch,
    several times, using the same initial conditions and run calls.

    Test that the result is exact for this machine and np.isclose
    wrt to the original file, which might have originated in a different
    machine.
    """
    metro_old = Metropolis.from_pickle(
        Path(__file__).parent / "run_output" / "amorphous_1000_to_reproduce.p")
    original_efficiency = metro_old.actuator.efficiency
    new_efficiencies = [
        replicate_run().actuator.efficiency
        for _ in range(10)
    ]

    # all recreated runs in this computer are exactly equal
    pass_condition_1 = len(set(new_efficiencies)) == 1
    # they are very very close to the original run
    pass_condition_2 = np.isclose(
        new_efficiencies[0],
        original_efficiency,
        rtol=1e-16
    )
    pass_condition = (pass_condition_1) and (pass_condition_2)
    # exit with code 1 if it didn't pass the test
    # so that subprocess can catch in from within pytest
    fail_condition = not pass_condition
    if fail_condition:
        print("# efficiency in this machine (10 times):")
        for eta in new_efficiencies:
            print("%.32G" % eta)
        print("")
        print("# efficiency in original machine:")
        print("%.32G" % original_efficiency)
        exit(code=1)
