#!/usr/bin/env python

"""Logic to combine dynamics and cost in one framework"""

import itertools
import multiprocessing as mp
from time import perf_counter as pc

import numpy as np

from .control import ilqrSolver
from .cost import ReferenceCost, GameCost
from .dynamics import DynamicalModel, MultiDynamicalModel
from .util import compute_pairwise_distance, split_agents, split_graph


def solve_decentralized(problem, X, U, radius, is_mp=False, verbose=True, **kwargs):
    """Solve the problem via decentralization into subproblems"""

    x_dims = problem.game_cost.x_dims
    u_dims = problem.game_cost.u_dims

    N = U.shape[0]
    n_states = x_dims[0]
    n_controls = u_dims[0]
    n_agents = len(x_dims)
    ids = problem.ids

    # Compute interaction graph based on relative distances.
    graph = define_inter_graph_threshold(X, radius, x_dims, ids)
    if verbose:
        print(f"Interaction Graph: {graph}")

    # Split up the initial state and control for each subproblem.
    x0_split = split_graph(X[np.newaxis, 0], x_dims, graph)
    U_split = split_graph(U, u_dims, graph)

    X_dec = np.zeros((N + 1, n_agents * n_states))
    U_dec = np.zeros((N, n_agents * n_controls))

    # Solve all problems in one process, keeping results for each agent in *_dec.
    if not is_mp:
        for i, (subproblem, x0i, Ui, id_) in enumerate(
            zip(problem.split(graph), x0_split, U_split, ids)
        ):
            t0 = pc()
            Xi_agent, Ui_agent, id_ = solve_subproblem(
                (subproblem, x0i, Ui, id_), verbose=verbose, **kwargs
            )

            if verbose:
                print(
                    f"Problem {id_}: {graph[id_]}\nTook {pc() - t0} seconds\n"
                    + "=" * 60
                )

            X_dec[:, i * n_states : (i + 1) * n_states] = Xi_agent
            U_dec[:, i * n_controls : (i + 1) * n_controls] = Ui_agent

    # Solve in separate processes using imap.
    else:
        # Package up arguments for the subproblem solver.
        args = zip(problem.split(graph), x0_split, U_split, ids, [verbose] * len(graph))

        t0 = pc()
        with mp.Pool(processes=n_agents) as pool:
            for i, (Xi_agent, Ui_agent, id_) in enumerate(
                pool.imap_unordered(solve_subproblem, args)
            ):

                if verbose:
                    print(
                        "=" * 60
                        + f"\nProblem {id_}: {graph[id_]}\nTook {pc() - t0} seconds"
                    )
                X_dec[:, i * n_states : (i + 1) * n_states] = Xi_agent
                U_dec[:, i * n_controls : (i + 1) * n_controls] = Ui_agent

    # Evaluate the cost of this combined trajectory.
    full_solver = ilqrSolver(problem, N)
    _, J_full = full_solver._rollout(X[0], U_dec)

    return X_dec, U_dec, J_full


def solve_rhc(
    problem,
    x0,
    N,
    *args,
    centralized=True,
    n_d=2,
    step_size=1,
    J_converge=None,
    dist_converge=None,
    **kwargs,
):
    """Solve the problem in a receding horizon fashion either centralized or
    decentralized
    """

    if (J_converge is None) == (dist_converge is None):
        raise ValueError("Must either specify a convergence cost or distance")

    if J_converge:

        def predicate(_, J):
            return J >= J_converge

    elif dist_converge:
        xf = problem.game_cost.xf
        n_states = problem.dynamics.x_dims[0]
        n_agents = problem.n_agents

        def predicate(x, _):
            dist_to_goal = np.linalg.norm(
                (x - xf).reshape(n_agents, n_states)[:, :n_d], axis=1
            )
            return np.any(dist_to_goal > dist_converge)

    n_x = problem.dynamics.n_x
    n_u = problem.dynamics.n_u

    xi = x0.reshape(1, -1)
    X = xi.copy()
    U = np.zeros((N, n_u))
    centralized_solver = ilqrSolver(problem, N)

    J = np.inf
    X_full = np.zeros((0, n_x))
    U_full = np.zeros((0, n_u))

    while predicate(xi, J):

        if centralized:
            X, U, J = centralized_solver.solve(xi, U, **kwargs)
        else:
            X, U, J = solve_decentralized(problem, X, U, *args, **kwargs)

        xi = X[step_size]

        X_full = np.r_[X_full, X[:step_size]]
        U_full = np.r_[U_full, U[:step_size]]

        # Seed the next solve by staying at the last visited state.
        X = np.r_[X[step_size:], np.tile(X[-1], (step_size, 1))]
        U = np.r_[U[step_size:], np.zeros((step_size, n_u))]

    _, J_full = centralized_solver._rollout(x0, U_full)

    # Handle immediate convergence condition without any optimization.
    if not X_full.size and not U_full.size:
        X_full = x0.copy()
        U_full = np.zeros((problem.dynamics.n_u))

    return X_full, U_full, J_full


def solve_subproblem(args, **kwargs):
    """Solve the sub-problem and extract results for this agent"""

    subproblem, x0, U, id_ = args
    N = U.shape[0]

    subsolver = ilqrSolver(subproblem, N)
    Xi, Ui, _ = subsolver.solve(x0, U, **kwargs)
    return *subproblem.extract(Xi, Ui, id_), id_


def solve_subproblem_starmap(subproblem, x0, U, id_):
    """Package up the input arguments for compatiblity with mp.imap()."""
    return solve_subproblem((subproblem, x0, U, id_))


class ilqrProblem:
    """Centralized optimal control problem that combines dynamics and cost"""

    def __init__(self, dynamics, cost):
        self.dynamics = dynamics
        self.game_cost = cost
        self.n_agents = 1
        if isinstance(cost, GameCost):
            self.n_agents = len(cost.ref_costs)

    @property
    def ids(self):
        if not isinstance(self.dynamics, MultiDynamicalModel):
            raise NotImplementedError(
                "Only MultiDynamicalModel's have an 'ids' attribute"
            )
        if not self.dynamics.ids == self.game_cost.ids:
            raise ValueError(f"Dynamics and cost have inconsistent ID's: {self}")
        return self.dynamics.ids.copy()

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

    def extract(self, X, U, id_):
        """Extract the state and controls for a particular agent id_ from the
        concatenated problem state/controls
        """

        if id_ not in self.ids:
            raise IndexError(f"Index {id_} not in ids: {self.ids}.")

        ext_ind = self.ids.index(id_)
        Xi = split_agents(X, self.game_cost.x_dims)[ext_ind]
        Ui = split_agents(U, self.game_cost.u_dims)[ext_ind]

        return Xi, Ui

    def selfish_warmstart(self, x0, N):
        """Compute a 'selfish' warmstart by ignoring other agents"""

        print("=" * 80 + "\nComputing warmstart...")
        # Split out the full problem into separate problems for each agent.
        x0 = x0.reshape(1, -1)
        selfish_graph = {id_: [id_] for id_ in self.ids}
        subproblems = self.split(selfish_graph)
        x0s = split_agents(x0, self.game_cost.x_dims)

        U_warm = np.zeros((N, self.dynamics.n_u))
        t0_all = pc()
        for problem, x0i, id_ in zip(subproblems, x0s, self.ids):
            t0 = pc()
            solver = ilqrSolver(problem, N)
            _, Ui, _ = solver.solve(x0i)
            print(f"Problem {id_}: Took {pc() - t0} seconds\n" + "=" * 60)

            nu_i = problem.dynamics.n_u
            ind_i = self.ids.index(id_)
            U_warm[:, nu_i * ind_i : nu_i * (ind_i + 1)] = Ui

        print(f"All: {self.ids}\nTook {pc() - t0_all} seconds\n" + "=" * 80)

        return U_warm

    def __repr__(self):
        return f"ilqrProblem(\n\t{self.dynamics},\n\t{self.game_cost}\n)"


def define_inter_graph_threshold(X, radius, x_dims, ids):
    """Compute the interaction graph based on a simple thresholded distance
    for each pair of agents sampled over the trajectory
    """

    planning_radii = 2 * radius
    rel_dists = compute_pairwise_distance(X, x_dims).T

    N = X.shape[0]
    n_samples = 10
    sample_step = max(N // n_samples, 1)
    sample_slice = slice(0, N + 1, sample_step)

    # Put each pair of agents within each others' graphs if they are within
    # some threshold distance from each other.
    graph = {id_: [id_] for id_ in ids}
    pair_inds = np.array(list(itertools.combinations(ids, 2)))
    for i, pair in enumerate(pair_inds):
        if np.any(rel_dists[sample_slice, i] < planning_radii):
            graph[pair[0]].append(pair[1])
            graph[pair[1]].append(pair[0])

    graph = {agent_id: sorted(prob_ids) for agent_id, prob_ids in graph.items()}
    return graph


def _reset_ids():
    """Set each of the agent specific ID's to zero for understandability"""
    DynamicalModel._reset_ids()
    ReferenceCost._reset_ids()
