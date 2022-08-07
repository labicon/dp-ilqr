#!/usr/bin/env python

"""Benchmark of the performance of centralized vs. decentralized potential iLQR

We conduct two primary analyses in this script, namely:
1. Allow unlimited solve time and stop after the solver converges or diverges.
2. Cap the solve time based on a "real-time" constraint.

The objective in 1 is to contrast solve times, whereas in 2 we contrast trajectory
quality in a real-time application of the algorithm. For both cases, we utilize
uniform random initial positions with stationary agents.

"""

import logging
from pathlib import Path
from time import strftime

import numpy as np

from decentralized.cost import GameCost, ProximityCost, ReferenceCost
from decentralized.dynamics import (
    DoubleIntDynamics4D,
    UnicycleDynamics4D,
    QuadcopterDynamics6D,
    MultiDynamicalModel,
)
from decentralized.decentralized import solve_rhc
from decentralized.problem import ilqrProblem
from decentralized.util import split_agents, random_setup


def multi_agent_run(model, x_dims, dt, N, radius, energy=10.0, n_d=2, **kwargs):
    """Single simulation comparing the centralized and decentralized solvers"""

    if not len(set(x_dims)) == 1:
        raise ValueError("Dynamics dimensions must be consistent")

    n_agents = len(x_dims)
    n_states = x_dims[0]
    STEP_SIZE = 1

    x0, xf = random_setup(
        n_agents,
        n_states,
        is_rotation=False,
        rel_dist=n_agents,
        var=1.0,
        n_d=n_d,
        random=True,
        energy=energy,
    )

    x_dims = [n_states] * n_agents

    ids = [100 + i for i in range(n_agents)]
    dynamics = MultiDynamicalModel([model(dt, id_) for id_ in ids])

    if model in {DoubleIntDynamics4D, UnicycleDynamics4D}:
        Q = 1.0 * np.diag([1, 1] + [0] * (n_states - 2))
        R = np.eye(2)
    elif model is QuadcopterDynamics6D:
        Q = np.eye(n_states)
        R = np.eye(3)

    Qf = 1000.0 * np.eye(Q.shape[0])

    goal_costs = [
        ReferenceCost(xf_i, Q.copy(), R.copy(), Qf.copy(), id_)
        for xf_i, id_ in zip(split_agents(xf.T, x_dims), ids)
    ]
    prox_cost = ProximityCost(x_dims, radius)
    game_cost = GameCost(goal_costs, prox_cost)

    problem = ilqrProblem(dynamics, game_cost)

    # Solve the problem centralized.
    print("\t\t\tcentralized")
    Xc, Uc, Jc = solve_rhc(
        problem,
        x0,
        N,
        radius,
        centralized=True,
        n_d=n_d,
        step_size=STEP_SIZE,
        **kwargs,
    )

    # Solve the problem decentralized.
    print("\t\t\tdecentralized")
    Xd, Ud, Jd = solve_rhc(
        problem,
        x0,
        N,
        radius,
        centralized=False,
        n_d=n_d,
        step_size=STEP_SIZE,
        **kwargs,
    )


def setup_logger(limit_solve_time):
    analysis = "1" if not limit_solve_time else "2"
    LOG_PATH = Path(__file__).parent.parent / "logs"
    LOG_FILE = LOG_PATH / strftime(f"dec-mc-{analysis}_%m-%d-%y_%H.%M.%S.csv")
    if not LOG_PATH.is_dir():
        LOG_PATH.mkdir()
    print(f"Logging results to {LOG_FILE}")
    logging.basicConfig(filename=LOG_FILE, format="%(message)s", level=logging.INFO)
    logging.info(
        "dynamics,n_agents,trial,centralized,last,t,J,horizon,dt,converged,ids,times,"
        "subgraphs,dist_left"
    )


def monte_carlo_analysis(limit_solve_time=False):
    """Benchmark to evaluate algorithm over many random initial conditions"""

    setup_logger(limit_solve_time)

    n_trials_iter = range(30)
    n_agents_iter = range(3, 8)
    models = [
        DoubleIntDynamics4D,
        UnicycleDynamics4D,
        QuadcopterDynamics6D,
    ]

    dt = 0.1
    N = 40
    ENERGY = 10.0
    radius = 0.5

    if limit_solve_time:
        t_kill = dt
        t_diverge = N * dt
    else:
        t_kill = None
        t_diverge = 5 * N * dt

    for model in models:
        print(f"{model.__name__}")
        for n_agents in n_agents_iter:
            print(f"\tn_agents: {n_agents}")
            for i_trial in n_trials_iter:
                print(f"\t\ttrial: {i_trial}")
                n_d = 3 if model is QuadcopterDynamics6D else 2
                x_dims = [model(-1).n_x] * n_agents
                multi_agent_run(
                    model,
                    x_dims,
                    dt,
                    N,
                    radius,
                    n_d=n_d,
                    t_kill=t_kill,
                    dist_converge=0.1,
                    t_diverge=t_diverge,
                    energy=ENERGY,
                    i_trial=i_trial,
                    verbose=False,
                )


def main():
    # monte_carlo_analysis(False)  # analysis 1
    monte_carlo_analysis(True)  # analysis 2


if __name__ == "__main__":
    main()
