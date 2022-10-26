#!/usr/bin/env python

from functools import reduce
from itertools import cycle
from operator import mul

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
import numpy as np

from decentralized.util import split_agents, compute_pairwise_distance


plt.rcParams.update(
    {
        "axes.grid": False,
        "figure.constrained_layout.use": True,
        # "text.usetex": True,
        # "font.family": "serif",
        "font.serif": ["Palatino"],
        "ps.distiller.res": 8000,
    }
)


def set_bounds(xydata, ax=None, zoom=0.1):
    """Set the axis on plt.gca() by some margin beyond the data, default 10% margin

    Reference:
    https://github.com/zjwilliams20/pocketknives/blob/main/pocketknives/python/graphics.py

    """

    xydata = np.atleast_2d(xydata)

    if not ax:
        ax = plt.gca()

    xmarg = xydata[:, 0].ptp() * zoom
    ymarg = xydata[:, 1].ptp() * zoom
    ax.set(
        xlim=(xydata[:, 0].min() - xmarg, xydata[:, 0].max() + xmarg),
        ylim=(xydata[:, 1].min() - ymarg, xydata[:, 1].max() + ymarg),
    )


def nchoosek(n, k):
    """n! / (k! * (n - k)!)

    Parameters
    ----------
    n, k : int

    Returns
    -------
    int

    Reference:
    https://github.com/zjwilliams20/pocketknives/blob/main/pocketknives/python/numerical.py

    """

    k = min(k, n - k)
    num = reduce(mul, range(n, n - k, -1), 1)
    denom = reduce(mul, range(1, k + 1), 1)
    return num // denom


def plot_interaction_graph(graph):
    """Visualize the interaction graph using networkx"""

    plt.clf()

    # Remove self-looping nodes.
    graph = {k: [vi for vi in v if vi != k] for k, v in graph.items()}

    G = nx.Graph(graph)

    options = {
        "font_size": 10,
        "node_size": 600,
        "node_color": plt.cm.Set3.colors[: len(graph)],
        "edgecolors": "black",
    }

    nx.draw_networkx(G, nx.spring_layout(G, k=0.5), **options)
    plt.margins(0.1)
    plt.draw()


def plot_solve(X, J, x_goal, x_dims=None, color_agents=False, n_d=2):
    """Plot the resultant trajectory on plt.gcf()"""

    if n_d not in (2, 3):
        raise ValueError()

    if not x_dims:
        x_dims = [X.shape[1]]

    if n_d == 2:
        ax = plt.gca()
    else:
        ax = plt.gcf().add_subplot(projection="3d")

    N = X.shape[0]
    n = np.arange(N)

    X_split = split_agents(X, x_dims)
    x_goal_split = split_agents(x_goal.reshape(1, -1), x_dims)

    for i, (Xi, xg) in enumerate(zip(X_split, x_goal_split)):
        c = n
        if n_d == 2:
            if color_agents:
                c = plt.cm.tab10.colors[i]
                ax.plot(Xi[:, 0], Xi[:, 1], c=c, lw=5, zorder=1)
            else:
                ax.scatter(Xi[:, 0], Xi[:, 1], c=c)
            ax.scatter(Xi[0, 0], Xi[0, 1], 80, "g", "x", label="$x_0$")
            ax.scatter(xg[0, 0], xg[0, 1], 80, "r", "x", label="$x_f$")
        else:
            if color_agents:
                c = [plt.cm.tab10.colors[i]] * Xi.shape[0]
            ax.scatter(Xi[:, 0], Xi[:, 1], Xi[:, 2], c=c)
            ax.scatter(
                Xi[0, 0], Xi[0, 1], Xi[0, 2], s=80, c="g", marker="x", label="$x_0$"
            )
            ax.scatter(
                xg[0, 0], xg[0, 1], xg[0, 2], s=80, c="r", marker="x", label="$x_f$"
            )

    plt.margins(0.1)
    plt.title(f"Final Cost: {J:.3g}")
    plt.draw()


def plot_pairwise_distances(X, x_dims, n_dims, radius):
    """Render all-pairwise distances in the trajectory"""

    ax = plt.gca()
    ax.plot(compute_pairwise_distance(X, x_dims, n_dims[1]))
    ax.hlines(radius, *plt.xlim(), "r", ls="--", label="$d_{prox}$")
    ax.set_title("Inter-Agent Distances")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Pairwise Distance (m)")
    ax.legend()
    plt.draw()


def _setup_gif(axes, X, xf, x_dims, radius, distances):

    ax1, ax2 = axes
    n_agents = len(x_dims)
    handles1 = []
    for _, c in zip(range(n_agents), cycle(plt.cm.tab20.colors)):
        handles1.append(
            (
                ax1.plot(0, c=c, marker="o", markersize=4)[0],
                ax1.add_artist(
                    plt.Circle(
                        (np.nan, np.nan), radius, color="k", fill=True, alpha=0.3, lw=2
                    )
                ),
            )
        )

    for xg in split_agents(xf, x_dims):
        ax1.scatter(xg[0, 0], xg[0, 1], c="r", marker="x", zorder=10)

    X_cat = np.vstack(split_agents(X, x_dims))
    set_bounds(X_cat, axes[0], zoom=0.15)
    ax1.set_title("Trajectories")
    plt.draw()

    handles2 = []
    n_pairs = nchoosek(n_agents, 2)
    for _, c in zip(range(n_pairs), cycle(plt.cm.tab20.colors)):
        handles2.append(ax2.plot(0, c=c)[0])
    ax2.hlines(radius, 0, X.shape[0], "r", ls="--", label="$d_{prox}$")
    ax2.set_ylim(0.0, distances.max())
    ax2.set_title("Inter-Distances")
    ax2.set_ylabel("Distance [m]")
    ax2.set_xlabel("Time Step")
    ax2.legend()

    return (
        handles1,
        handles2,
    )


def _animate(t, handles1, handles2, X, x_dims, distances):
    """Animate the solution into a gif"""

    for i, (xi, hi) in enumerate(zip(split_agents(X, x_dims), handles1)):
        hi[0].set_xdata(xi[:t, 0])
        hi[0].set_ydata(xi[:t, 1])
        hi[1].set_center(xi[t - 1, :2])

    for i, hi in enumerate(handles2):
        hi.set_xdata(range(t))
        hi.set_ydata(distances[:t, i])

    plt.draw()
    return (
        *handles1,
        *handles2,
    )


def make_trajectory_gif(gifname, X, xf, x_dims, radius):
    """Create a GIF of the evolving trajectory"""

    _, axes = plt.subplots(1, 2, figsize=(10, 6))

    N = X.shape[0]
    distances = compute_pairwise_distance(X, x_dims)

    handles = _setup_gif(axes, X, xf.flatten(), x_dims, radius, distances)
    anim = FuncAnimation(
        plt.gcf(),
        _animate,
        frames=N + 1,
        fargs=(*handles, X, x_dims, distances),
        repeat=True,
    )
    anim.save(gifname, fps=N // 10, dpi=100)


def eyeball_scenario(x0, xf, n_agents, n_states):
    """Render the scenario in 2D"""
    plt.clf()

    plt.gca().set_aspect("equal")
    X = np.dstack(
        [x0.reshape(n_agents, n_states), xf.reshape(n_agents, n_states)]
    ).swapaxes(1, 2)
    for i, Xi in enumerate(X):
        plt.annotate(
            "", Xi[1, :2], Xi[0, :2], arrowprops=dict(facecolor=plt.cm.tab20.colors[i])
        )
    set_bounds(X.reshape(-1, n_states), zoom=0.2)
    plt.draw()
