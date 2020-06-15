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
Functions to plot mechanical metamaterials.

"""
from matplotlib.cm import Greys
from matplotlib.colors import LinearSegmentedColormap
from .lattice import Lattice
from .actuator import Actuator

from typing import Optional, Tuple

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Arc
from matplotlib.colors import LogNorm
from matplotlib.cm import get_cmap

from itertools import cycle

import numpy as np
import matplotlib

iter_colors = cycle([
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00"
])


def alpha_cmap(source_cmap="seismic", max_alpha: float = 1):
    """
    Make a colormap with transparency "in the middle", assuming that
    source_cmap is divergent. In other words, a colormap with non-transparent
    colors for large negative and positive values that gradually turn to 
    transparency for values going to zero.

    Parameters
    ----------
    source_cmap: str
        A named matplotlib colormap.
    max_alpaha: float
        The value of alpha at both end of the colormap.

    Returns
    -------
    magic_cmap: matplotlib.colors.LinearSegmentedColormap

    Notes
    ------
    It is assumed that source_cmap is divergent.
    The values of alpha of the obtained cmap are
    [max_alpha, ..., 0, ..., max_alpha]
    """
    my_cmap = [tuple(list(get_cmap(source_cmap)(x))[:3] +
                     [np.abs(x - 128) / 128 * max_alpha]) for x in range(256)]
    magic_cmap = LinearSegmentedColormap.from_list(
        colors=my_cmap,
        name="magic"
    )
    return magic_cmap


def show_actuator(
    actuator: Actuator,
    **kwargs
) -> None:
    if actuator.method == "displacement":
        _show_actuator_displacement(
            actuator=actuator,
            **kwargs
        )
    elif actuator.method == "force":
        _show_actuator_force(
            actuator=actuator,
            **kwargs
        )
    else:
        raise NotImplementedError(
            f"No visualization implemented for {actuator.method} method")


def _show_actuator_force(
    actuator: Actuator,
    input_color: str = "#d68f00",
    output_color: str = "#32a852",
    draw_output_springs: bool = False,
) -> None:
    # must act before showing
    actuator.act()
    # recover params from actuator
    lattice = actuator.lattice
    nodes_positions = lattice._nodes_positions
    input_nodes = actuator.input_nodes
    output_nodes = actuator.output_nodes
    frozen_nodes = actuator.frozen_nodes
    # get the displaced positions
    nodes_positions_displaced = np.array([
        actuator._relax_params["X"],
        actuator._relax_params["Y"]
    ]).T
    # these are the active edges
    active_edges = []
    for edge in lattice.edges:
        u, v = edge._nodes
        active_edges.append([u.label, v.label])
    # create a new lattice holding only the active edges
    lattice_displaced = Lattice(
        nodes_positions=nodes_positions_displaced,
        edges_indices=np.array(active_edges)
    )
    for edge in lattice_displaced._possible_edges:
        lattice_displaced.flip_edge(edge)

    # create a figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    ax.set_aspect(1)
    # draw non-displaced lattice
    _plot_lattice(lattice, color="0.6", ax=ax,
                  plot_angular_springs=False, alpha=0.6)
    # draw displaced lattice
    _plot_lattice(lattice_displaced, ax=ax, color="0.2",
                  plot_angular_springs=False, alpha=1,
                  )
    # draw input
    X0 = nodes_positions_displaced[input_nodes].T[0]
    Y0 = nodes_positions_displaced[input_nodes].T[1]
    X1 = X0 + 100 * actuator.input_vectors.T[0]
    Y1 = Y0 + 100 * actuator.input_vectors.T[1]
    ax.quiver(
        X0, Y0, X1 - X0, Y1 - Y0,
        angles='xy',
        width=0.005,
        color=input_color
    )
    if draw_output_springs:
        # draw output
        X0 = nodes_positions_displaced[output_nodes].T[0]
        Y0 = nodes_positions_displaced[output_nodes].T[1]
        # phantom spring other end
        X1 = nodes_positions[output_nodes].T[0] + actuator.output_vectors.T[0]
        Y1 = nodes_positions[output_nodes].T[1] + actuator.output_vectors.T[1]
        for x0, y0, x1, y1 in zip(X0, Y0, X1, Y1):
            _plot_spring(
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                ax=ax,
                spring_width=0.5
            )
    # draw frozen
    ax.scatter(
        *nodes_positions[frozen_nodes].T,
        marker="s",
        color="0.6",
        alpha=0.6,
        zorder=-1,
        s=100
    )
    # show the figure
    fig.show()


def _show_actuator_displacement(
    actuator: Actuator,
    input_color: str = "#DE4515",
    output_color: str = "#32a852",
    desired_output_color: str = "#185428",
) -> None:
    # must act before showing
    actuator.act()
    # recover params from actuator
    lattice = actuator.lattice
    nodes_positions = lattice._nodes_positions
    input_nodes = actuator.input_nodes
    output_nodes = actuator.output_nodes
    frozen_nodes = actuator.frozen_nodes
    # get the displaced positions
    nodes_positions_displaced = np.array([
        actuator._relax_params["X"],
        actuator._relax_params["Y"]
    ]).T
    # these are the active edges
    active_edges = []
    for edge in lattice.edges:
        u, v = edge._nodes
        active_edges.append([u.label, v.label])
    # create a new lattice holding only the active edges
    lattice_displaced = Lattice(
        nodes_positions=nodes_positions_displaced,
        edges_indices=np.array(active_edges)
    )
    for edge in lattice_displaced._possible_edges:
        lattice_displaced.flip_edge(edge)

    # create a figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    ax.set_aspect(1)
    # draw non-displaced lattice
    _plot_lattice(lattice, color="0.6", ax=ax,
                  plot_angular_springs=False, alpha=0.6)
    # draw displaced lattice
    _plot_lattice(lattice_displaced, ax=ax, color="0.2",
                  plot_angular_springs=False, alpha=1,
                  )
    # draw input
    X0 = nodes_positions[input_nodes].T[0]
    Y0 = nodes_positions[input_nodes].T[1]
    X1 = nodes_positions_displaced[input_nodes].T[0]
    Y1 = nodes_positions_displaced[input_nodes].T[1]
    ax.quiver(
        X0, Y0, X1 - X0, Y1 - Y0,
        angles='xy', scale_units='xy',
        scale=1.05,
        width=0.005,
        color=input_color
    )
    # draw real output
    X0 = nodes_positions[output_nodes].T[0]
    Y0 = nodes_positions[output_nodes].T[1]
    X1 = nodes_positions_displaced[output_nodes].T[0]
    Y1 = nodes_positions_displaced[output_nodes].T[1]
    ax.quiver(
        X0, Y0, X1 - X0, Y1 - Y0,
        angles='xy', scale_units='xy',
        scale=1.05,
        width=0.005,
        color=output_color
    )
    # draw desired output
    X0 = nodes_positions[output_nodes].T[0]
    Y0 = nodes_positions[output_nodes].T[1]
    X1 = nodes_positions[output_nodes].T[0] + actuator.output_vectors.T[0]
    Y1 = nodes_positions[output_nodes].T[1] + actuator.output_vectors.T[1]
    ax.quiver(
        X0, Y0, X1 - X0, Y1 - Y0,
        angles='xy', scale_units='xy',
        scale=1.05,
        width=0.003,
        color=desired_output_color,
        alpha=0.7,
        zorder=10,
    )
    # draw frozen
    ax.scatter(
        *nodes_positions[frozen_nodes].T,
        marker="s",
        color="0.6",
        alpha=0.6,
        zorder=-1,
        s=100
    )
    # show the figure
    fig.show()


def _plot_lattice(
    lattice: Lattice,
    ax: Optional[Axes] = None,
    alpha=1,
    plot_angular_springs: bool = True,
    color="0.2",
) -> None:
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    # fix the aspect ratio
    ax.set_aspect(1)
    # plot the nodes
    x, y = lattice._nodes_positions.T
    ax.scatter(x, y, s=10, c=color, alpha=alpha)

    # plot the edges
    segments = []
    _c = 1.2
    lognorm = LogNorm(vmin=1 / _c, vmax=1 * _c)
    cmap = get_cmap("coolwarm")
    for edge in lattice.edges:
        node0, node1 = edge._nodes
        line = (node0.x, node0.y), (node1.x, node1.y)
        segments.append(line)
    linecollection = LineCollection(
        segments=segments, colors=color, alpha=alpha)
    ax.add_collection(linecollection)

    # plot the angular springs
    if plot_angular_springs:
        for node in lattice.nodes:
            for angular_spring in node.angular_springs:
                theta1 = angular_spring._node_origin._get_angle(
                    angular_spring._node_start)
                theta2 = angular_spring._node_origin._get_angle(
                    angular_spring._node_end)
                arc = Arc(
                    xy=(angular_spring._node_origin.x,
                        angular_spring._node_origin.y),
                    height=0.05,
                    width=0.05,
                    angle=0,
                    theta1=np.rad2deg(theta1) + 5,
                    theta2=np.rad2deg(theta2) - 5,
                    color=next(iter_colors),
                    linewidth=2,
                    alpha=alpha
                )
                ax.add_patch(arc)


def _plot_spring(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    ax: Axes,
    spring_width=0.05,
    spring_num_waves=10
) -> None:
    """
    Plot a spring using a simple sine wave

    Parameters
    ----------
    x0 : float
        start of spring, x coordinate
    y0 : float
        start of spring, y coordinate
    x1 : float
        end of spring, x coordinate
    y1 : float
        end of spring, y coordinate
    ax : Axes
        matplotlib axes where to draw the spring
    spring_width : float, optional
        width of spring, from highest to lowest point across spring direction, by default 0.05
    spring_num_waves : int, optional
        number of sine waves, by default 10
    """
    x, y = _get_spring_points(
        x=x1 - x0,
        y=y1 - y0,
        ax=ax,
        spring_width=spring_width,
        spring_num_waves=spring_num_waves
    )
    x += x0
    y += y0
    ax.plot(x, y, c="#093285", lw=1)
    ax.scatter([x0, x1], [y0, y1], s=8, c="#093285")


def _get_spring_points(
    x: float,
    y: float,
    ax: Axes,
    spring_width=0.05,
    spring_num_waves=10
) -> Tuple:
    """draw a spring from (0, 0) to (x, y)"""
    spring_length = np.sqrt(x ** 2 + y ** 2)
    spring_rotation = np.arctan2(y, x)
    # now prepare it for no rotation
    w = np.linspace(0, 1, num=200) * np.math.pi * 2 * spring_num_waves
    w = np.concatenate([
        np.zeros(50),
        w,
        np.zeros(50)
    ])
    U = np.linspace(0, 1, num=300) * spring_length
    V = np.sin(w) * spring_width / 2
    # create rotation matrix
    c, s = np.cos(spring_rotation), np.sin(spring_rotation)
    R = np.array(((c, -s), (s, c)))
    X, Y = R @ np.array([U, V])
    return X, Y
