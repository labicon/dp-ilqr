#!/usr/bin/env python

"""Logic to combine dynamics and cost in one framework"""

import itertools

import numpy as np
from sklearn.cluster import DBSCAN
import torch

from .util import compute_pairwise_distance, split_agents
from .dynamics import DynamicalModel, MultiDynamicalModel
from .cost import ReferenceCost, GameCost


class ilqrProblem:
    """Centralized optimal control problem that combines dynamics and cost"""

    def __init__(self, dynamics, cost):
        self.dynamics = dynamics
        self.game_cost = cost

        self.n_agents = 1
        self.ids = [0]
        if isinstance(cost, GameCost):
            self.n_agents = len(cost.ref_costs)
        if isinstance(dynamics, MultiDynamicalModel):
            self.ids = [model.id for model in dynamics.submodels]

    def split(self, graph):
        """Split up this centralized problem into a list of decentralized
        sub-problems.
        """

        split_dynamics = self.dynamics.split(graph)
        split_costs = self.game_cost.split(graph)

        return [
            ilqrProblem(dynamics, cost)
            for dynamics, cost in zip(split_dynamics, split_costs)
        ]

    def extract(self, X, U, i):
        """Extract the state and controls for a particular agent i from the
        concatenated problem state/controls
        """

        if i not in self.ids:
            raise IndexError(f"Index {i} not in ids: {self.ids}.")

        ext_ind = self.ids.index(i)
        Xi = split_agents(X, self.game_cost.x_dims)[ext_ind]
        Ui = split_agents(U, self.game_cost.u_dims)[ext_ind]

        return Xi, Ui

    def __repr__(self):
        return f"ilqrProblem(\n\t{self.dynamics},\n\t{self.game_cost}\n)"


def define_inter_graph_threshold(X, n_agents, radius, x_dims):
    """Compute the interaction graph based on a simple thresholded distance
    for each pair of agents sampled over the trajectory
    """

    planning_radii = 4 * radius
    rel_dists = compute_pairwise_distance(X, x_dims).T

    N = X.shape[0]
    n_samples = 10
    sample_step = max(N // n_samples, 1)
    sample_slice = slice(0, N + 1, sample_step)

    # Put each pair of agents within each others' graphs if they are within
    # some threshold distance from each other.
    graph = {i: [i] for i in range(n_agents)}
    pair_inds = np.array(list(itertools.combinations(range(n_agents), 2)))
    for i, pair in enumerate(pair_inds):
        if torch.any(rel_dists[sample_slice, i] < planning_radii):
            graph[pair[0]].append(pair[1])
            graph[pair[1]].append(pair[0])

    graph = {i: sorted(ids) for i, ids in graph.items()}
    return graph


def define_inter_graph_dbscan(X, n_agents, n_states, radius):
    """Determine the interaction graph between agents depending on whether
    they share clusters over.

    NOTE: deprecated in favor of the speed and simplicity of
    ``define_inter_graph_threshold``.
    """

    # Organize states by agent keeping only position.
    X_pos = X.reshape(-1, n_agents, n_states)[..., :2]

    # Setup the DBSCAN clusterer, where the cluster threshold (ϵ) of
    # defining a cluster is the planning radius.
    planning_radii = 4 * radius
    dbscan = DBSCAN(min_samples=1, eps=planning_radii)

    # Sample some number of clusterings from the full trajectory.
    N = X.shape[0]
    n_samples = 10
    sample_step = max(N // n_samples, 1)
    sample_iter = range(0, N + 1, sample_step)
    labels = np.zeros((len(sample_iter), n_agents), dtype=int)
    for i, t in enumerate(sample_iter):
        labels[i] = dbscan.fit_predict(X_pos[t])

    # Define the interaction graph by matching up agents when they share a cluster.
    graph = {i: set([i]) for i in range(n_agents)}
    pair_inds = np.array(list(itertools.combinations(range(n_agents), 2)))
    for pair in pair_inds:
        intersect = np.intersect1d(labels[:, pair[0]], labels[:, pair[1]])
        if intersect.size:
            graph[pair[0]].add(pair[1])
            graph[pair[1]].add(pair[0])

    return graph


def _reset_ids():
    """Set each of the agent specific ID's to zero for understandability"""
    DynamicalModel._reset_ids()
    ReferenceCost._reset_ids()
