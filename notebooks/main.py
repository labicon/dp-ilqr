#!/usr/bin/env python
# coding: utf-8

# # take2
#
# **GOAL**: Keep It Stupid Simple = KISS.
#
# **Dependencies:**
#  - [pocketknives](https://github.com/zjwilliams20/pocketknives)
#
# **References:**
#  1. [ilqgames/python](https://github.com/HJReachability/ilqgames/blob/master/python)


import numpy as np
import matplotlib.pyplot as plt
import torch

from decentralized import split_agents
import decentralized as dec

π = np.pi
plt.rcParams["axes.grid"] = True


def pos_mask(x_dims):
    """Return a mask that's true wherever there's an x or y position"""
    return np.array([i % x_dims[0] < 2 for i in range(sum(x_dims))])


# ## single-agent problem
def single_agent():
    dt = 0.1
    N = 50

    x = torch.tensor([-10, 10, 0], dtype=torch.float)
    x_goal = torch.zeros((3, 1), dtype=torch.float).T

    dynamics = dec.CarDynamics3D(dt)

    Q = torch.diag(torch.tensor([1.0, 1, 0]))
    Qf = 1000 * torch.eye(Q.shape[0])
    R = torch.eye(2)
    cost = dec.ReferenceCost(x_goal, Q, R, Qf)

    prob = dec.ilqrProblem(dynamics, cost)
    ilqr = dec.ilqrSolver(prob, N)
    X, U, J = ilqr.solve(x)
    # plot_solve(X, J, x_goal.numpy())


# ## multi-agent problem
def paper_setup():
    """Hardcoded example with reasonable consistency eyeballed from
    Potential-iLQR paper
    """
    x0 = torch.tensor(
        [[0.5, 1.5, 0.1, 0, 2.5, 1.5, π, 0, 1.5, 1.3, π / 2, 0]], dtype=torch.float
    ).T
    x_goal = torch.tensor([[2.5, 1.5, 0, 0, 0.5, 1.5, π, 0, 1.5, 2.2, π / 2, 0]]).T
    return x0, x_goal


def car_setup():
    """Same as paper_setup but using car dynamics"""
    x0, x_goal = paper_setup()
    car_mask = [i % 4 < 3 for i in range(x0.shape[0])]
    return x0[car_mask], x_goal[car_mask]


def bike_setup():
    """Same as paper_setup but using bike dynamics"""
    N_AGENTS = 3
    N_STATES = 5
    n_total_states = N_AGENTS * N_STATES
    bike_mask = [i % 5 < 4 for i in range(n_total_states)]

    x0, x_goal = paper_setup()
    x0_bike = torch.zeros(n_total_states, 1)
    x_goal_bike = torch.zeros_like(x0_bike)
    x0_bike[bike_mask] = x0
    x_goal_bike[bike_mask] = x_goal

    return x0_bike, x_goal_bike


def double_int_setup():
    x0, x_goal = paper_setup()
    theta_mask = [i % 4 == 2 for i in range(x0.shape[0])]
    with torch.no_grad():
        x0[theta_mask] = 0.0
        x_goal[theta_mask] = 0.0

    return x0, x_goal


def dec_test_setup():
    x0, x_goal = paper_setup()
    x0_other = torch.tensor([[5, 5, 0, 0, 6, 6, π / 4, 0]]).T
    x_goal_other = torch.tensor([[6, 4, -π / 2, 0, 4, 6, π / 4, 0]]).T

    x0 = torch.cat([x0, x0_other])
    x_goal = torch.cat([x_goal, x_goal_other])

    return x0, x_goal


def compute_energy(x, x_dims):
    """Determine the sum of distances from the origin"""
    return torch.sum(x[pos_mask(x_dims)].reshape(-1, 2).norm(dim=1)).item()


def normalize_energy(x, x_dims, energy=10.0):
    """Zero-center the coordinates and then ensure the sum of
    squared distances == energy
    """

    # Don't mutate x's data for this function, keep it pure.
    x = x.clone()
    n_agents = len(x_dims)
    center = x[pos_mask(x_dims)].reshape(-1, 2).mean(0)

    with torch.no_grad():
        x[pos_mask(x_dims)] -= center.tile(n_agents).reshape(-1, 1)
        x[pos_mask(x_dims)] *= energy / compute_energy(x, x_dims)
    assert x.numel() == sum(x_dims)

    return x


def perturb_state(x, x_dims, var=0.5):
    """Add a little noise to the start to knock off perfect symmetries"""

    x = x.clone()
    with torch.no_grad():
        x[pos_mask(x_dims)] += var * torch.randn_like(x[pos_mask(x_dims)])

    return x


def multi_agent():
    n_agents = 5
    n_states = 4
    n_controls = 2
    ENERGY = 10.0

    # x0, x_goal = paper_setup()
    # x0, x_goal = car_setup()
    # x0, x_goal = bike_setup()
    # x0, x_goal = double_int_setup()
    # x0, x_goal = dec_test_setup()
    x0, x_goal = dec.random_setup(
        n_agents, n_states, is_rotation=False, min_sep=1.0, var=1.0
    )

    x_dims = [n_states] * n_agents
    # u_dims = [2] * n_agents

    x0 = normalize_energy(x0, x_dims, ENERGY)
    x_goal = normalize_energy(x_goal, x_dims, ENERGY)
    # x0 = perturb_state(x0, x_dims)

    dt = 0.05
    N = 50
    # tol = 1e-3
    ids = [100 + i for i in range(n_agents)]

    model = dec.UnicycleDynamics4D
    # model = dec.CarDynamics3D
    # model = dec.BikeDynamics5D
    # model = dec.DoubleIntDynamics4D
    dynamics = dec.MultiDynamicalModel([model(dt, id_) for id_ in ids])

    Q = 4 * torch.diag(torch.tensor([1.0, 1, 0, 0]))
    # Qf = 1000 * torch.eye(Q.shape[0])
    Qf = 1000 * torch.diag(torch.tensor([1.0, 1, 1, 1]))
    R = torch.eye(2)

    # radius = ENERGY / 20
    radius = 0.5

    goal_costs = [
        dec.ReferenceCost(x_goal_i, Q.clone(), R.clone(), Qf.clone(), id_)
        for x_goal_i, id_ in zip(split_agents(x_goal.T, x_dims), ids)
    ]
    prox_cost = dec.ProximityCost(x_dims, radius)
    game_cost = dec.GameCost(goal_costs, prox_cost)

    prob = dec.ilqrProblem(dynamics, game_cost)
    # ## decentralized multi-agent

    X0 = torch.tile(x0.T, (N, 1))
    U0 = torch.zeros((N, n_controls * n_agents))
    X_dec, U_dec, J_dec = dec.solve_decentralized(prob, X0, U0, radius, is_mp=False)


if __name__ == "__main__":
    # single_agent()
    multi_agent()
