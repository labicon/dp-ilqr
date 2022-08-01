#!/usr/bin/env python

"""Various utilities used in other areas of the code."""

from dataclasses import dataclass
import itertools
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation

π = np.pi


@dataclass
class Point(object):
    """Point in 3D"""

    x: float
    y: float
    z: float = 0

    @property
    def ndim(self):
        return 2 if self.z == 0 else 3

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return Point(self.x * other.x, self.y * other.y, self.z * other.z)

    def __repr__(self):
        return str((self.x, self.y, self.z))


def compute_pairwise_distance(X, x_dims, n_d=2):
    """Compute the distance between each pair of agents"""
    assert len(set(x_dims)) == 1

    n_agents = len(x_dims)
    n_states = x_dims[0]

    if n_agents == 1:
        raise ValueError("Can't compute pairwise distance for one agent.")

    pair_inds = np.array(list(itertools.combinations(range(n_agents), 2)))
    X_agent = X.reshape(-1, n_agents, n_states).swapaxes(0, 2)
    dX = X_agent[:n_d, pair_inds[:, 0]] - X_agent[:n_d, pair_inds[:, 1]]
    return np.linalg.norm(dX, axis=0)


def split_agents(Z, z_dims):
    """Partition a cartesian product state or control for individual agents"""
    return np.split(np.atleast_2d(Z), np.cumsum(z_dims[:-1]), axis=1)


def split_graph(Z, z_dims, graph):
    """Split up the state or control by grouping their ID's according to the graph"""
    assert len(set(z_dims)) == 1

    # Create a mapping from the graph to indicies.
    mapping = {id_: i for i, id_ in enumerate(list(graph))}

    n_z = z_dims[0]
    z_split = []
    for n, ids in graph.items():
        inds = [mapping[id_] for id_ in ids]
        z_split.append(
            np.concatenate([Z[:, i * n_z : (i + 1) * n_z] for i in inds], axis=1)
        )

    return z_split


def pos_mask(x_dims, n_d=2):
    """Return a mask that's true wherever there's a spatial position"""
    return np.array([i % x_dims[0] < n_d for i in range(sum(x_dims))])


def randomize_locs(n_pts, random=False, rel_dist=3.0, var=3.0, n_d=2):
    """Uniformly randomize locations of points in N-D while enforcing
    a minimum separation between them.
    """

    # Distance to move away from center if we're too close.
    Δ = 0.1 * n_pts
    x = var * np.random.uniform(-1, 1, (n_pts, n_d))

    if random:
        return x

    # Determine the pair-wise indicies for an arbitrary number of agents.
    pair_inds = np.array(list(itertools.combinations(range(n_pts), 2)))
    move_inds = np.arange(n_pts)

    # Keep moving points away from center until we satisfy radius
    while move_inds.size:
        center = np.mean(x, axis=0)
        distances = compute_pairwise_distance(x.flatten(), [n_d] * n_pts)

        move_inds = pair_inds[distances.flatten() <= rel_dist]
        x[move_inds] += Δ * (x[move_inds] - center)

    return x


def face_goal(x0, xf):
    """Make the agents face the direction of their goal with a little noise"""

    VAR = 0.01
    dX = xf[:, :2] - x0[:, :2]
    headings = np.arctan2(*np.rot90(dX, 1))

    x0[:, -1] = headings + VAR * np.random.randn(x0.shape[0])
    xf[:, -1] = headings + VAR * np.random.randn(x0.shape[0])

    return x0, xf


def random_setup(n_agents, n_states, is_rotation=False, n_d=2, energy=None, **kwargs):
    """Create a randomized set up of initial and final positions"""

    # We don't have to normlize for energy here
    x_i = randomize_locs(n_agents, n_d=n_d, **kwargs)

    # Rotate the initial points by some amount about the center.
    if is_rotation:
        θ = π + random.uniform(-π / 4, π / 4)
        R = Rotation.from_euler("z", θ).as_matrix()[:2, :2]
        x_f = x_i @ R - x_i.mean(axis=0)
    else:
        x_f = randomize_locs(n_agents, n_d=n_d, **kwargs)

    x0 = np.c_[x_i, np.zeros((n_agents, n_states - n_d))].reshape(-1, 1)
    xf = np.c_[x_f, np.zeros((n_agents, n_states - n_d))].reshape(-1, 1)

    # Normalize to satisfy the desired energy of the problem.
    if energy:
        x0 = normalize_energy(x0, [n_states] * n_agents, energy, n_d)
        xf = normalize_energy(xf, [n_states] * n_agents, energy, n_d)

    return x0, xf


def compute_energy(x, x_dims, n_d=2):
    """Determine the sum of distances from the origin"""
    return np.linalg.norm(x[pos_mask(x_dims, n_d)].reshape(-1, n_d), axis=1).sum()


def normalize_energy(x, x_dims, energy=10.0, n_d=2):
    """Zero-center the coordinates and then ensure the sum of
    squared distances == energy
    """

    # Don't mutate x's data for this function, keep it pure.
    x = x.copy()
    n_agents = len(x_dims)
    center = x[pos_mask(x_dims, n_d)].reshape(-1, n_d).mean(0)

    x[pos_mask(x_dims, n_d)] -= np.tile(center, n_agents).reshape(-1, 1)
    x[pos_mask(x_dims, n_d)] *= energy / compute_energy(x, x_dims, n_d)
    assert x.size == sum(x_dims)

    return x


def perturb_state(x, x_dims, n_d=2, var=0.5):
    """Add a little noise to the start to knock off perfect symmetries"""

    x = x.copy()
    x[pos_mask(x_dims, n_d)] += var * np.random.randn(*x[pos_mask(x_dims, n_d)].shape)

    return x


def plot_interaction_graph(graph):
    """Visualize the interaction graph using networkx"""

    plt.clf()

    # Remove self-looping nodes.
    graph = {k: [vi for vi in v if vi != k] for k, v in graph.items()}

    G = nx.Graph(graph)

    options = {
        "font_size": 10,
        "node_size": 600,
        "node_color": "white",
        "edgecolors": "black",
    }

    nx.draw_networkx(G, nx.spring_layout(G, k=1.5), **options)
    plt.margins(0.1)
    plt.draw()
