#!/usr/bin/env python

"""Logic related to the subdivision of a centralized problem

The main magic behind this splitting is the ``define_inter_graph_threshold``.
In other words, define the interaction graph via thresholding relative distances.
Additionally, this also includes the Receding Horizon Controller (RHC) implementation,
which must see the full scope of the decentralization.

"""

import itertools
import logging
from time import perf_counter as pc

import numpy as np

from .control import ilqrSolver
from .problem import solve_subproblem
from .util import split_graph, compute_pairwise_distance

g = 9.81


def solve_distributed(problem, X, U, radius, ignore_ids=None, pool=None, verbose=True, **kwargs):
    """Solve the problem via division into subproblems"""

    x_dims = problem.game_cost.x_dims
    u_dims = problem.game_cost.u_dims

    N = U.shape[0]
    n_states = x_dims[0]
    n_controls = u_dims[0]
    n_agents = len(x_dims)
    ids = problem.ids
    solve_info = {}

    if ignore_ids and any(id_ not in ids for id_ in ignore_ids):
        raise ValueError(f"Some of {ignore_ids} not in {ids}.")

    # Compute interaction graph based on relative distances.
    graph = define_inter_graph_threshold(X, radius, x_dims, ids)
    if verbose:
        print("=" * 80 + f"\nInteraction Graph: {graph}")

    # Split up the initial state and control for each subproblem.
    x0_split = split_graph(X[np.newaxis, 0], x_dims, graph)
    U_split = split_graph(U, u_dims, graph)

    X_dec = np.zeros((N + 1, n_agents * n_states))
    U_dec = np.zeros((N, n_agents * n_controls))

    # Solve all problems in one process, keeping results for each agent in *_dec.

    if not pool:
        for i, (subproblem, x0i, Ui, id_) in enumerate(
            zip(problem.split(graph), x0_split, U_split, ids)
        ):
            if id_ in ignore_ids:
                if verbose:
                    solve_info[id_] = (0.0, [id_])
                    print(f"Ignoring subproblem {id_}...")
                continue

            t0 = pc()
            Xi_agent, Ui_agent, id_ = solve_subproblem(
                (subproblem, x0i, Ui, id_, False), **kwargs
            )
            Δt = pc() - t0

            if verbose:
                print(f"Problem {id_}: {graph[id_]}\nTook {Δt} seconds\n")

            X_dec[:, i * n_states : (i + 1) * n_states] = Xi_agent
            U_dec[:, i * n_controls : (i + 1) * n_controls] = Ui_agent

            solve_info[id_] = (Δt, graph[id_])

    # Solve in separate processes using imap.
    else:
        # Package up arguments for the subproblem solver.
        args = zip(problem.split(graph), x0_split, U_split, ids, [verbose] * len(graph))

        t0 = pc()
        for i, (Xi_agent, Ui_agent, id_) in enumerate(
            pool.imap_unordered(solve_subproblem, args)
        ):

            Δt = pc() - t0
            if verbose:
                print(f"Problem {id_}: {graph[id_]}\nTook {Δt} seconds")
            X_dec[:, i * n_states : (i + 1) * n_states] = Xi_agent
            U_dec[:, i * n_controls : (i + 1) * n_controls] = Ui_agent

            # NOTE: This cannot be compared to the single-processed version due to
            # multi-processing overhead.
            solve_info[id_] = (Δt, graph[id_])

    # Evaluate the cost of this combined trajectory.
    full_solver = ilqrSolver(problem, N)
    _, J_full = full_solver._rollout(X[0], U_dec)

    return X_dec, U_dec, J_full, solve_info


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
    t_diverge=None,
    i_trial=None,
    verbose=False,
    **kwargs,
):
    """Solve the problem in a receding horizon fashion either centralized or
    decentralized
    """

    if (J_converge is None) == (dist_converge is None):
        raise ValueError("Must either specify a convergence cost or distance")

    xf = problem.game_cost.xf

    def distance_to_goal(x):
        return np.linalg.norm((x - xf).reshape(n_agents, n_states)[:, :n_d], axis=1)

    if J_converge:

        def predicate(_, J):
            return J >= J_converge

    elif dist_converge:
        n_states = problem.dynamics.x_dims[0]
        n_agents = problem.n_agents

        def predicate(x, _):
            return np.any(distance_to_goal(x) > dist_converge)

    n_x = problem.dynamics.n_x
    n_u = problem.dynamics.n_u
    model_name = problem.dynamics.submodels[0].__class__.__name__

    xi = x0.reshape(1, -1)
    X = xi.copy()
    # U = np.zeros((N, n_u))
    U = np.random.rand(N, n_u) * 0.01
    # U = np.tile([g, 0, 0], (N, n_agents))
    centralized_solver = ilqrSolver(problem, N)

    t = 0
    J = np.inf
    converged = True
    dt = problem.dynamics.dt
    ids = problem.ids.copy()
    X_full = np.zeros((0, n_x))
    U_full = np.zeros((0, n_u))

    while predicate(xi, J):
        if verbose:
            print(f"t: {t:.3g}")

        if centralized:
            X, U, J, solve_info = solve_centralized(
                centralized_solver, xi, U, ids, False, **kwargs
            )
            # print(f"Shape of X at each prediction horizon is{X.shape}")
        else:
            X, U, J, solve_info = solve_distributed(
                problem, X, U, *args, verbose=False, **kwargs
            )
            # print(f"Shape of X at each prediction horizon is{X.shape}")
        xi = X[step_size]

        X_full = np.r_[X_full, X[:step_size]]
        U_full = np.r_[U_full, U[:step_size]]

        # Seed the next solve by staying at the last visited state.
        X = np.r_[X[step_size:], np.tile(X[-1], (step_size, 1))]
        U = np.r_[U[step_size:], np.zeros((step_size, n_u))]

        times = [tup[0] for tup in solve_info.values()]
        subgraphs = [tup[1] for tup in solve_info.values()]
        distance_left = distance_to_goal(xi).tolist()
        logging.info(
            f'"{model_name}",{problem.n_agents},{i_trial},{centralized},'
            f'{False},{t},{J},{N},{dt},{converged},"{ids}","{times}","{subgraphs}",'
            f'"{distance_left}"'
        )

        if t_diverge and t >= t_diverge:
            converged = False
            if verbose:
                print("Failed to converge within allotted time...")
            break

        # Keep track of simulation time as we go.
        t += step_size * dt

    # Handle immediate convergence condition without any optimization.
    if not X_full.size and not U_full.size:
        X_full = x0.copy()
        U_full = np.zeros((1, problem.dynamics.n_u))

    # Rollout the final control sequence to evaluate the cost.
    _, J_full = centralized_solver._rollout(x0, U_full)
    # Final simulation time where the predicate was satisfied.
    tf = U_full.shape[0] * dt

    logging.info(
        f'"{model_name}",{problem.n_agents},{i_trial},{centralized},'
        f'{True},{tf},{J_full},{N},{dt},{converged},"{ids}","{times}","{subgraphs}",'
        f'"{distance_left}"'
    )

    return X_full, U_full, J_full


def define_inter_graph_threshold(X, radius, x_dims, ids):
    """Compute the interaction graph based on a simple thresholded distance
    for each pair of agents sampled over the trajectory
    """

    planning_radii = 2 * radius
    rel_dists = compute_pairwise_distance(X, x_dims)

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


def solve_centralized(solver, xi, U, ids, verbose, **kwargs):
    """Thin function call to unify profiling function traces"""

    t0 = pc()
    X, U, J = solver.solve(xi, U, verbose=verbose, **kwargs)
    Δt = pc() - t0
    solve_info = {id_: (Δt, ids) for id_ in ids}

    return X, U, J, solve_info
