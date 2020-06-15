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
from .actuator import Actuator
from .spring import LinearSpring
from .node import Node
from .utils import coupon_collector_bound

from typing import Optional, Tuple, List, Union, Dict, Any
from typing import List, Set, Deque

import numpy as np
from tqdm import tqdm
import copy
from pathlib import Path
from time import process_time

from itertools import chain
from itertools import cycle
from collections import deque
from ordered_set import OrderedSet

from pandas.io.pickle import to_pickle
from pandas.io.pickle import read_pickle
from pickle import HIGHEST_PROTOCOL

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_agg import FigureCanvas

from io import BytesIO


class Metropolis:

    def __init__(
        self,
        actuator: Actuator,
    ) -> None:
        # store inputs
        self.actuator = actuator
        self.lattice = self.actuator.lattice

        # initial efficiency
        self.best_efficiency = -np.inf
        self.best_actuator = copy.deepcopy(actuator)

        # acceptance rate tracker
        self._last_n_accepted: Deque[bool] = deque(maxlen=100)
        # fill in with one value at least to avoid taking
        # average of empty list later
        self._last_n_accepted.append(True)

        # define edges that can be fliped.
        forbidden = self.actuator.input_nodes + \
            self.actuator.output_nodes + self.actuator.frozen_nodes
        self._removable_edges: Set[LinearSpring] = OrderedSet()
        for edge in self.lattice._possible_edges:
            u, v = edge._nodes
            if (u.label not in forbidden) and (v.label not in forbidden):
                self._removable_edges.add(edge)

        # make sure there are edges that can be removed
        if len(self._removable_edges) == 0:
            raise RuntimeError(
                "The actuator has no edges that can be added/removed.")

        # init history container
        self.history: Dict[str, List] = {
            "temperature": [],
            "current_efficiency": [],
            "proposed_efficiency": [],
            "accepted": [],
            "relax_time": [],
            "relax_internal_steps": []
        }
        self.edges_history: List[Set] = []

        # init counter for consecutive rejects
        self._consecutive_rejects = 0

        # store value of numpy random state
        # see this post
        # https://stackoverflow.com/questions/32172054/how-can-i-retrieve-the-current-seed-of-numpys-random-number-generator
        # tl;dr cannot represent all possible mersenne twister states with a long int, need the full tuple of 600 and something
        # ints that you get with np.random.get_state(). Used in conjunction with np.random.set_state() it gives truly
        # reproducible simulations, that is, brings the RNG to the exact same position.
        self._initial_rng_state = np.random.get_state()

        # store initial state of actuator
        self._initial_actuator = copy.deepcopy(actuator)

        # to store all the params of all the calls to run
        self._run_calls_history: List[Dict] = list()

        # mark myself as valid
        # this can change for instance if relaxation
        # reaches maxiter
        self.is_valid = True

    @staticmethod
    def from_pickle(path: Union[str, Path]) -> "Metropolis":
        """
        Load pickled Metropolis object from file.


        Returns
        -------
        [Metropolis]
            Deserialized Metropolis python object.
        """
        metro = read_pickle(path)
        metro.lattice._set_recursion_limit()
        return metro

    def to_pickle(self, path: Union[str, Path]) -> None:
        """
        Save Metropolis instance as serialized pickle file.


        Parameters
        ----------
        path : Union[str, Path]
            File path where the pickled object will be stored.

        Raises
        ------
        RuntimeError
            If file already exists
        RuntimeError
            If parent dir does not exist
        """
        assert isinstance(path, (str, Path))
        path = Path(path)
        # make sure directory exists
        directory = path.parent
        if not directory.is_dir():
            raise RuntimeError(
                f"Trying to write to non-existing directory {directory}")
        # make sure file does not exist
        if path.exists():
            raise RuntimeError(
                f"File already exists {path}. Refusing to overwrite!")
        # pickle
        # avoid using protocol 5 if available (python 3.8 only feature)
        to_pickle(self, path, protocol=min(HIGHEST_PROTOCOL, 4))

    def run(
        self,
        num_steps: int,
        temperature: Optional[float] = None,
        initial_temperature: Optional[float] = None,
        final_temperature: Optional[float] = None,
        temperature_change_mode: str = "absolute",
        temperature_advance_on: str = "accepted",
        max_consecutive_rejects: Optional[int] = None,
        min_neighbours: Optional[int] = None,
        progress_bar: bool = True,
        early_stop_efficiency: Optional[float] = None
    ):
        """
        Run metropolis algorithm.

        Run the standard Metropolis–Hastings algorithm to find 
        configurations of self.lattice that have high efficiency 
        according to self.actuator. Run the algorithm for a fixed
        number of steps, at certain given temperature or
        temperature interval.

        Parameters
        ----------
        num_steps : int
            Number of steps.
        temperature : Optional[float], optional
            Constant temperature, by default None.
            Cannot be passed together with initial_temperature
            or final_temperature.
        initial_temperature : Optional[float], optional
            Initial temperature, by default None.
            Must be passed together with final_temperature.
        final_temperature : Optional[float], optional
            Final temperature, by default None.
            Must be passed together with initial_temperature.
        temperature_change : str, optional
            Make temperature changes in relative or absolute terms,
            by default "relative". In the first case, temperature
            changes as T = c * T at each timestep. In the second case,
            temperature changes as T = T + c at teach timestep.
        temperature_advance_on : str, optional
            Change the temperature at all steps or only at accepted steps.
        max_consecutive_rejects : int, optional
            Prematurely stop the simulation after n consecutive
            rejected MC steps. Ignored if temperature_advance_on
            is set to 'all'. Defaults to a sensible value that depends on the
            number of removable edges, see docs.
        min_neighbours : Optional[int], optional
            Consider only changes in the lattice configuration
            that leave all nodes with at least min_neighbours
            neighbours, by default None.
        progress_bar : bool, optional
            Show a progress bar with some useful statistics during
            Monte Carlo, by default True.
        early_stop_efficiency : float, optional
            Stop prematurely if efficiency reaches this value.
            Defaults to infinity, so that it never triggers.
        """

        # store call into history for reproducibility
        run_call = locals()
        self._run_calls_history.append(run_call)

        # store input
        self.min_neighbours: Optional[int] = min_neighbours
        self.progress_bar = progress_bar

        # use coupons collector problem bound
        # to set a sensible value for max_rejects
        if max_consecutive_rejects is None:
            # we are 99.9% sure that all elements of _removable_edges
            # will be chosen at least once using this many trials
            max_consecutive_rejects = coupon_collector_bound(
                num_different_coupons=len(self._removable_edges),
                sureness=1e-3
            )
        # case temperature not constant
        if temperature is None:
            assert initial_temperature is not None
            assert final_temperature is not None
            assert initial_temperature > 0
            assert final_temperature >= 0
            self._run_varying_temperature(
                initial_temperature=initial_temperature,
                final_temperature=final_temperature,
                num_steps=num_steps,
                temperature_change_mode=temperature_change_mode,
                temperature_advance_on=temperature_advance_on,
                max_consecutive_rejects=max_consecutive_rejects,
                early_stop_efficiency=early_stop_efficiency
            )

        # case temperature constant
        # we just reuse the varying temperature code
        else:
            assert initial_temperature is None
            assert final_temperature is None
            assert temperature >= 0
            initial_temperature = temperature
            final_temperature = temperature
            self._run_varying_temperature(
                initial_temperature=temperature,
                final_temperature=temperature,
                num_steps=num_steps,
                temperature_advance_on=temperature_advance_on,
                temperature_change_mode=temperature_change_mode,
                max_consecutive_rejects=max_consecutive_rejects,
                early_stop_efficiency=early_stop_efficiency
            )

    def get_ML_dataset(self, dpi: int = 100, subset: Optional[List] = None):
        """get all the data out, ready for ML"""
        self._setup_viz()
        if subset is None:
            subset = list(range(len(self.edges_history)))
        # TODO: check that subset is a valid subset
        X = np.array([
            self._edges_to_array(self.edges_history[i], dpi=dpi)
            for i in subset
        ])
        y = np.array(self.history["proposed_efficiency"])[subset]
        return X, y

    def _run_varying_temperature(
        self,
        initial_temperature: float,
        final_temperature: float,
        num_steps: int,
        temperature_change_mode: str = "relative",
        temperature_advance_on: str = "accepted",
        max_consecutive_rejects: Union[int, float, None] = 5000,
        early_stop_efficiency: Optional[float] = None
    ) -> None:
        """
        Run metropolis while varying the temperature.

        Implements different ways of varying the temperature: either
        linear or geometric changes, which occur either at all steps
        or only at accepted steps.

        Parameters
        ----------
        initial_temperature : float
            Initial temperature, must be non-negative.
        final_temperature : float
            Final temperature, must be non-negative.
        num_steps : int
            Number of total steps when advance_on ='all' or
            number of accepted steps when advance_on='accepted'.
        temperature_change : str, optional
            Make temperature changes in relative or absolute terms,
            by default "relative".
        temperature_advance_on : str, optional
            Change the temperature after either 'all' steps or only
            after 'accepted' steps, by default "all".
        max_consecutive_rejects : int, optional
            Prematurely stop the simulation after n consecutive
            rejected MC steps. Ignored if temperature_advance_on
            is set to 'all'. Defaults to 5000.
        """
        if early_stop_efficiency is None:
            early_stop_efficiency = float("inf")

        if temperature_advance_on not in ["all", "accepted", None]:
            raise RuntimeError(
                f"advance_on keyword must be one of 'all' or 'accepted' ")

        if temperature_change_mode == "absolute":
            # create the temperatures
            temperatures = list(np.linspace(
                initial_temperature, final_temperature, num=num_steps + 1))
            # make sure max_consecutive_rejects was set to something sensible
            if max_consecutive_rejects is None:
                raise RuntimeError(
                    f"Please set the max_consecutive_rejects keyword.")
            if max_consecutive_rejects < 5:
                raise RuntimeError(f"Keyword max_consecutive_rejects was set to {max_consecutive_rejects},"
                                   "which is suspisciously low.")
        elif temperature_change_mode == "relative":
            # create the list of temperatures
            temperatures = list(np.geomspace(
                initial_temperature, final_temperature, num=num_steps + 1))
            # ignore max rejects by setting it to infinity
            max_consecutive_rejects = float("inf")
        else:
            raise RuntimeError(
                f"Do not know how to set temperatures with temperature_change_mode = {temperature_change_mode}."
                "Currently known modes: absolute, relative"
            )

        # set initial temperature
        pbar = tqdm(total=len(temperatures) - 1, disable=not self.progress_bar)
        temperature = temperatures.pop(0)
        # main loop, finish when list of temperatures is empty or when rejected too many
        while \
                temperatures and \
                self._consecutive_rejects < max_consecutive_rejects and \
                self.best_efficiency < early_stop_efficiency and \
                self.is_valid:
            # change temperature if we accepted, move the progress bar
            if temperature_advance_on == "all" or self._last_n_accepted[-1]:
                temperature = temperatures.pop(0)
                pbar.update(1)
            # do the step
            self._step(temperature=temperature)
            pbar.set_postfix(ordered_dict={
                "Temperature": temperature,
                "Efficiency": self.actuator.efficiency,
                "Acc. rate": np.mean(self._last_n_accepted),
            })

    def _get_available_edges(self, method: str = "all") -> List:
        AVAILABLE_METHODS = ["all", "active", "mincoord"]
        # allow all edges except those that touch input, output
        # or frozen nodes
        if method == "all":
            return list(self._removable_edges)

        # allow only edges that "would change something"
        # for instance, you cannot add an edge not connected to the structure
        elif method == "active":
            edges_and_nn = set(chain.from_iterable([
                edge._neighbouring_linear_springs
                for edge
                in self.lattice.edges
            ]))
            # we intersect with the ones that we can touch
            active_edges = edges_and_nn.intersection(self._removable_edges)
            return list(active_edges)

        # allow only edges that leave all nodes with degree larger than mincoord
        # this avoids weak structures
        elif method == "mincoord":
            assert isinstance(self.min_neighbours, int)
            # start with those not in isolation
            active_edges = set(self._get_available_edges(method="active"))
            mincoord_edges: List[LinearSpring] = []
            for edge in active_edges:
                node_u, node_v = edge._nodes
                degree_u = len(node_u.neighbours)
                degree_v = len(node_v.neighbours)
                if edge not in self.lattice.edges:
                    degree_u += 1
                    degree_v += 1
                elif edge in self.lattice.edges:
                    degree_u -= 1
                    degree_v -= 1
                if degree_u >= self.min_neighbours and degree_v >= self.min_neighbours:
                    mincoord_edges.append(edge)
            return mincoord_edges

        elif method in AVAILABLE_METHODS:
            raise NotImplementedError(f"method {method} not implemented")
        else:
            raise RuntimeError(f"method {method} not recognized")

    def _pick_edge(self, available_edges: List[LinearSpring]) -> LinearSpring:
        """pick and edge given a list of edges"""
        edge = np.random.choice(available_edges)
        return edge

    def _step(self, temperature: float) -> None:
        """do a single MC step"""
        # pick an edge
        if self.min_neighbours is not None:
            method = "mincoord"
        elif self.min_neighbours is None:
            method = "all"
        available_edges = self._get_available_edges(method=method)
        edge = self._pick_edge(available_edges)

        # flip the edge and measure change in efficiency
        current_efficiency = self.actuator.efficiency
        current_relax_params = copy.deepcopy(self.actuator._relax_params)
        self.lattice.flip_edge(edge)
        # relax the configuration, measuring time
        _act_start_time = process_time()
        self.actuator.act()
        _act_end_time = process_time()
        # make sure we did not reach the maximum number of relaxation steps
        # if so, mark the run as not valid
        _relax_internal_steps = self.actuator._relax_params["step_relax"][0]
        if _relax_internal_steps > self.actuator.max_iter:
            self.is_valid = False
        proposed_efficiency = self.actuator.efficiency
        delta_efficiency = proposed_efficiency - current_efficiency

        # now the edge is already flipped. if the efficiency
        # has increased we always keep it. if it has decreased,
        # we undo the move (reject the new configuration) with
        # probability 1 - prob_accept.

        # efficiency has decreased, reject the change with
        # probability 1 - prob_accept
        if delta_efficiency < 0:
            if temperature > 0:
                prob_accept = np.exp(delta_efficiency / temperature)
            elif temperature == 0:
                prob_accept = 0
            else:
                raise RuntimeError(
                    f"Cannot work with negative temperature {temperature}"
                )
            if np.random.uniform() <= 1 - prob_accept:
                # now we are rejecting the configuration
                accepted = False
                self.edges_history.append(
                    {edge for edge in self.lattice.edges})
                self._consecutive_rejects += 1
                # flip edge back to previous state
                self.lattice.flip_edge(edge)
                # copy back previous relaxed state
                self.actuator.efficiency = current_efficiency
                self.actuator._relax_params = current_relax_params
            else:
                # accepting config
                accepted = True
                self.edges_history.append(
                    {edge for edge in self.lattice.edges})
                self._consecutive_rejects = 0

        # efficiency has increased or not changed, we always accept
        # we only need to check if its the best one ever or not
        else:
            accepted = True
            self.edges_history.append({edge for edge in self.lattice.edges})
            self._consecutive_rejects = 0
            prob_accept = 1
            if self.actuator.efficiency > self.best_efficiency:
                self.best_efficiency = self.actuator.efficiency
                self.best_actuator = copy.deepcopy(self.actuator)

        # keep track of dynamic acceptance rate (deque)
        # this is shown in the progress bar, the list
        # keeps only last n values, forgets the rest
        # automatically
        self._last_n_accepted.append(accepted)
        # save important variables in a log
        self._save_history(
            accepted=accepted,
            temperature=temperature,
            proposed_efficiency=proposed_efficiency,
            _act_end_time=_act_end_time,
            _act_start_time=_act_start_time,
            _relax_internal_steps=_relax_internal_steps
        )

    def _save_history(self, accepted, temperature, proposed_efficiency, _act_end_time, _act_start_time, _relax_internal_steps):
        """Save a few key variables into a history log
        """
        self.history["accepted"].append(accepted)
        self.history["temperature"].append(temperature)
        self.history["current_efficiency"].append(self.actuator.efficiency)
        self.history["proposed_efficiency"].append(proposed_efficiency)
        self.history["relax_time"].append(_act_end_time - _act_start_time)
        self.history["relax_internal_steps"].append(_relax_internal_steps)

    def _setup_viz(self, size: Tuple[float, float] = (2, 2)):
        """
        Create a figure to hold graphical representations of the lattice.

        The figure can be updated without being destroyed and recreated via
        the segments arg of the linecollections object.


        See rest of class methods...
        """
        # create figure, axes
        self._viz: Dict[str, Any] = {}
        fig = plt.figure(frameon=False)
        fig.set_size_inches(size[0], size[1])
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_aspect(1)
        ax.set_axis_off()
        fig.add_axes(ax)
        self._viz["fig"] = fig
        self._viz["ax"] = ax
        # set limits according to lattice
        xlim, ylim = np.quantile(
            self.lattice._nodes_positions,
            q=[0, 1],
            axis=0
        ).T
        # small margin, depending on lattice dimensions
        dx = min(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 20
        xlim[0] -= dx
        xlim[1] += dx
        ylim[0] -= dx
        ylim[1] += dx
        self._viz["ax"].set_xlim(xlim)
        self._viz["ax"].set_ylim(ylim)
        # put linecollection
        self._viz["linecollection"] = LineCollection(
            segments=[
                ((edge._nodes[0].x, edge._nodes[0].y),
                 (edge._nodes[1].x, edge._nodes[1].y))
                for edge in self.lattice.edges
            ],
            colors="black"
        )
        self._viz["ax"].add_collection(self._viz["linecollection"])

    def _update_viz(self, edges: Set[LinearSpring]):
        "update the graphical representation of the lattice with a set of edges"
        segments = [
            ((edge._nodes[0].x, edge._nodes[0].y),
             (edge._nodes[1].x, edge._nodes[1].y))
            for edge in edges
        ]
        self._viz["linecollection"].set_segments(segments)

    def _edges_to_array(self, edges: Set[LinearSpring], dpi: int = 100):
        """get array representation of a set of edges"""
        # update the figure with the given set of edges
        self._update_viz(edges)
        # save the figure to a buffer of bytes
        buf = BytesIO()
        self._viz["fig"].savefig(buf, format='png', dpi=dpi)
        # read image from binary buffer
        buf.seek(0)
        im = plt.imread(buf)
        # get first channel only
        # TODO: make sure sizes match
        arr = im[:, :, 0:1]
        return arr
