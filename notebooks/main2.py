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

import functools
import itertools
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import torch

from decentralized import control2
import pocketknives

π = np.pi

plt.rcParams["axes.grid"] = True


def split_agents(Z, z_dims):
    """Partition a cartesian product state or control for individual agents"""
    if torch.is_tensor(Z):
        return torch.split(torch.atleast_2d(Z), z_dims, dim=1)
    return np.split(np.atleast_2d(Z), np.cumsum(z_dims[:-1]), axis=1)


def pos_mask(x_dims):
    """Return a mask that's true wherever there's an x or y position"""
    return np.array([i % x_dims[0] < 2 for i in range(sum(x_dims))])


def plot_solve(X, J, x_goal, x_dims=None, dt=0.1):
    """Plot the resultant trajectory on plt.gcf()"""

    plt.clf()

    if not x_dims:
        x_dims = [X.shape[1]]

    N = X.shape[0]
    t = np.arange(N) * dt

    X_split = split_agents(X, x_dims)
    x_goal_split = split_agents(x_goal.reshape(1, -1), x_dims)

    for Xi, xg in zip(X_split, x_goal_split):
        plt.scatter(Xi[:, 0], Xi[:, 1], c=t)
        plt.scatter(Xi[0, 0], Xi[0, 1], 80, "g", "x", label="$x_0$")
        plt.scatter(xg[0, 0], xg[0, 1], 80, "r", "x", label="$x_f$")

    plt.margins(0.1)
    plt.title(f"Final Cost: {J:.3g}")


# ## single-agent problem
def unicycle_continuous(x, u):
    """
    Compute the time derivative of state for a particular state/control.
    NOTE: `x` and `u` should be 2D (i.e. column vectors).
    REF: [1]
    """
    assert isinstance(x, torch.Tensor) and isinstance(u, torch.Tensor)

    x_dot = torch.zeros(x.numel())
    x_dot[0] = x[3] * torch.cos(x[2])
    x_dot[1] = x[3] * torch.sin(x[2])
    x_dot[2] = u[0]
    x_dot[3] = u[1]
    return x_dot


def reference_cost(x, u, _x_goal, _Q, _R, _Qf=None, terminal=False):
    """Cost of reaching the goal"""

    assert isinstance(x, torch.Tensor) and isinstance(u, torch.Tensor)
    x = x.reshape(-1, 1)
    u = u.reshape(-1, 1)

    if _Qf is None:
        _Qf = torch.eye(_Q.shape[0])

    if terminal:
        return (x - _x_goal).T @ _Qf @ (x - _x_goal)
    return (x - _x_goal).T @ _Q @ (x - _x_goal) + u.T @ _R @ u


def single_agent_sim():
    dt = 0.1
    N = 50

    x = torch.tensor([-10, 10, 0, 0], dtype=torch.float, requires_grad=True)
    x_goal = torch.zeros((4, 1), dtype=torch.float)

    Q = torch.diag(torch.tensor([1.0, 1, 0, 0]))
    Qf = 1000 * torch.eye(Q.shape[0])
    R = torch.eye(2)
    goal_cost = functools.partial(reference_cost, _x_goal=x_goal, _Q=Q, _R=R, _Qf=Qf)

    ilqr = control2.iLQR(unicycle_continuous, goal_cost, x.numel(), 2, dt, N)
    X, U, J = ilqr.solve(x)
    # plot_solve(X, J, x_goal.numpy())


# ## multi-agent problem
def compute_pairwise_distance(X, x_dims):
    """Compute the distance between each pair of agents"""
    assert len(set(x_dims)) == 1

    n_agents = len(x_dims)
    n_states = x_dims[0]

    pair_inds = np.array(list(itertools.combinations(range(n_agents), 2)))
    X_agent = X.reshape(-1, n_agents, n_states).swapaxes(0, 2)
    dX = X_agent[:2, pair_inds[:, 0]] - X_agent[:2, pair_inds[:, 1]]

    if isinstance(X, np.ndarray):
        return np.linalg.norm(dX, axis=0)
    elif torch.is_tensor(X):
        return torch.linalg.norm(dX, dim=0)


def dynamics_nd(f, x, u, _x_dims, _u_dims):
    """Compute the continuous time derivative for n agents"""
    assert isinstance(x, torch.Tensor) and isinstance(u, torch.Tensor)
    return torch.cat(
        [
            f(xi.flatten(), ui.flatten())
            for xi, ui in zip(split_agents(x, _x_dims), split_agents(u, _u_dims))
        ]
    )


def proximity_cost(X, _compute_pairwise_distance, _x_dims, _radius):
    """Penalizes distances underneath some radius between agents"""

    distances = _compute_pairwise_distance(X, _x_dims)
    pair_costs = torch.fmin(torch.zeros((1)), distances - _radius) ** 2
    return pair_costs.sum(dim=0)


def multi_agent_cost(goal_cost, prox_cost, x, u, terminal=False):
    """Reference deviation costs plus collision avoidance costs"""
    return goal_cost(x, u, terminal=terminal) + 100 * prox_cost(x)


# ### initializing the scene
# Here, we define some initial positions within some distance of each other, and then
# rotate them about the origin by some random angle to hopefully create some
# interesting scenarios.


def randomize_locs(n_pts, min_sep=3.0, var=3.0, n_dim=2):
    """Uniformly randomize locations of points in N-D while enforcing
       a minimum separation between them.
    """

    # Distance to move away from center if we're too close.
    Δ = 0.1 * n_pts
    x = var * np.random.uniform(-1, 1, (n_pts, n_dim))

    # Determine the pair-wise indicies for an arbitrary number of agents.
    pair_inds = np.array(list(itertools.combinations(range(n_pts), 2)))
    move_inds = np.arange(n_pts)

    # Keep moving points away from center until we satisfy radius
    while move_inds.size:
        center = np.mean(x, axis=0)
        distances = compute_pairwise_distance(x.flatten(), [n_dim] * n_pts)

        move_inds = pair_inds[distances.flatten() <= min_sep]
        x[move_inds] += Δ * (x[move_inds] - center)

    return x


def face_goal(x0, x_goal):
    """Make the agents face the direction of their goal with a little noise"""

    VAR = 0.01
    dX = x_goal[:, :2] - x0[:, :2]
    headings = np.arctan2(*np.rot90(dX, 1))

    x0[:, 2] = headings + VAR * np.random.randn(x0.shape[0])
    x_goal[:, 2] = headings + VAR * np.random.randn(x0.shape[0])

    return x0, x_goal


def randy_setup():
    """Hardcoded example with reasonable consistency eyeballed from Potential-iLQR
       paper
    """
    x0 = torch.tensor(
        [[0.5, 1.5, 0.1, 0, 2.5, 1.5, π, 0, 1.5, 1.3, π / 2, 0]],
        dtype=torch.float,
        requires_grad=True,
    ).T
    x_goal = torch.tensor([[2.5, 1.5, 0, 0, 0.5, 1.5, π, 0, 1.5, 2.2, π / 2, 0]]).T
    return x0, x_goal


# To be consistent between simulations, we normalize for the scale of the setup by
# computing the *energy*, or the sum of distances from the origin of the points.
# This should be the same for all runs.


def compute_energy(x, x_dims):
    """Determine the sum of distances from the origin"""
    return torch.sum(x[pos_mask(x_dims)].reshape(-1, 2).norm(dim=1)).item()


def normalize_energy(x, x_dims, energy=10.0):
    """Zero-center the coordinates and then ensure the sum of squared
       distances == energy
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


def random_setup(n_agents, **kwargs):
    """Create a randomized set up of initial and final positions"""

    # Rotate the initial points by some amount about the center.
    theta = π + random.uniform(-π / 4, π / 4)
    R = Rotation.from_euler("z", theta).as_matrix()[:2, :2]

    # We don't have to normlize for energy here
    x_i = randomize_locs(n_agents, **kwargs)
    x_f = x_i @ R + x_i.mean(axis=0)
    # x_f = randomize_locs(n_agents, 3.0)

    x0 = np.c_[x_i, np.zeros((n_agents, 2))]
    x_goal = np.c_[x_f, np.zeros((n_agents, 2))]
    x0, x_goal = face_goal(x0, x_goal)

    x0 = torch.from_numpy(x0).requires_grad_(True).type(torch.float)
    x_goal = torch.from_numpy(x_goal).type(torch.float)

    return x0.reshape(-1, 1), x_goal.reshape(-1, 1)


def multi_agent_sim():
    n_agents = 3
    # radius_init = 1.0
    # x0, x_goal = random_setup(n_agents, min_sep=radius_init, var=2.0)
    x0, x_goal = randy_setup()

    x_dims = [4] * n_agents
    u_dims = [2] * n_agents
    x0 = normalize_energy(x0, x_dims)
    x_goal = normalize_energy(x_goal, x_dims)

    x0 = perturb_state(x0, x_dims)

    plt.clf()
    plt.gca().set_aspect("equal")
    X = torch.dstack(
        [x0.reshape(n_agents, 4).detach(), x_goal.reshape(n_agents, 4).detach()]
    ).swapaxes(1, 2)
    for i, Xi in enumerate(X):
        plt.annotate(
            "",
            Xi[1, :2],
            Xi[0, :2],
            arrowprops=dict(facecolor=plt.cm.tab20.colors[2 * i]),
        )
    pocketknives.set_bounds(X.reshape(-1, 4), zoom=0.2)

    dt = 0.05
    N = 50
    tol = 1e-3

    dynamics_8d = functools.partial(
        dynamics_nd, unicycle_continuous, _x_dims=x_dims, _u_dims=u_dims
    )

    Q = 4 * torch.diag(torch.tensor([1.0, 1, 0, 0]).tile(n_agents))
    # Qf = 1000 * torch.eye(Q.shape[0])
    Qf = 1000 * torch.diag(torch.tensor([1.0, 1, 0, 1]).tile(n_agents))
    R = torch.eye(2 * n_agents)

    radius = 0.7

    goal_cost = functools.partial(reference_cost, _x_goal=x_goal, _Q=Q, _R=R, _Qf=Qf)
    prox_cost = functools.partial(
        proximity_cost,
        _x_dims=x_dims,
        _radius=radius,
        _compute_pairwise_distance=compute_pairwise_distance,
    )
    multi_cost = functools.partial(multi_agent_cost, goal_cost, prox_cost)

    ilqr = control2.iLQR(dynamics_8d, multi_cost, x0.numel(), 2*n_agents, dt, N)
    X, U, J = ilqr.solve(x0, tol=tol)

    # plot_solve(X, J, x_goal.numpy(), x_dims)


def setup(X, x_goal, x_dims, radius):
    plt.clf()

    n_agents = len(x_dims)
    ax = plt.gca()
    handles = []
    for i in range(n_agents):
        handles.append(
            (
                plt.plot(0, c=plt.cm.tab20.colors[2 * i], marker="o", markersize=4)[0],
                ax.add_artist(
                    plt.Circle(np.nan, radius, color="k", fill=True, alpha=0.3, lw=2)
                ),
            )
        )

    for xg in split_agents(x_goal, x_dims):
        plt.scatter(xg[0, 0], xg[0, 1], c="r", marker="x", zorder=10)

    X_cat = np.vstack(split_agents(X, x_dims))
    pocketknives.set_bounds(X_cat, zoom=0.3)

    return handles


def animate(t, handles, X, x_dims, x_goal):
    """Animate the solution into a gif"""
    for (i, xi), hi in zip(enumerate(split_agents(X, x_dims)), handles):
        hi[0].set_xdata(xi[:t, 0])
        hi[0].set_ydata(xi[:t, 1])
        hi[1].set_center(xi[t - 1, :2])


def main():
    # single_agent_sim()
    multi_agent_sim()


if __name__ == "__main__":
    main()
