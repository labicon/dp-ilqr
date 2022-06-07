#!/usr/bin/env python

# # take2
# 
# **GOAL**: Keep It Stupid Simple = KISS.
# 
# **References**:
#  1. [ilqgames/python](https://github.com/HJReachability/ilqgames/blob/master/python)


import functools
import itertools

import numpy as np
import matplotlib.pyplot as plt
import torch

from decentralized import control2
import pocketknives

plt.rcParams['axes.grid'] = True


def split_agents(Z, z_dims):
    """Partition a cartesian product state or control for individual agents"""
    if torch.is_tensor(Z):
        return torch.split(torch.atleast_2d(Z), z_dims, dim=1)
    return np.split(np.atleast_2d(Z), np.cumsum(z_dims[:-1]), axis=1)


def plot_solve(X, J, x_goal, x_dims=None):
    """Plot the resultant trajectory on plt.gcf()"""

    plt.clf()
    
    if not x_dims:
        x_dims = [X.shape[1]]
        
    N = X.shape[0]
    t = np.arange(N) * dt
    
    X_split = split_agents(X, x_dims)
    x_goal_split = split_agents(x_goal.reshape(1,-1), x_dims)
    
    for Xi, xg in zip(X_split, x_goal_split):
        plt.scatter(Xi[:,0], Xi[:,1], c=t)
        plt.scatter(Xi[0,0], Xi[0,1], 80, 'g', 'x', label="$x_0$")
        plt.scatter(xg[0,0], xg[0,1], 80, 'r', 'x', label="$x_f$")
    
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
    x = x.reshape(-1,1)
    u = u.reshape(-1,1)
    
    if _Qf is None:
        _Qf = torch.eye(Q.shape[0])
    
    if terminal:
        return (x - _x_goal).T @ _Qf @ (x - _x_goal)
    return (x - _x_goal).T @ _Q @ (x - _x_goal) + u.T @ _R @ u


def single_agent_sim():
    dt = 0.1
    N = 50
    
    x = torch.tensor([-10, 10, 0, 0], dtype=torch.float, requires_grad=True)
    x_goal = torch.zeros((4, 1), dtype=torch.float)
    
    Q = torch.diag(torch.tensor([1., 1, 0, 0]))
    Qf = 1000 * torch.eye(Q.shape[0])
    R = torch.eye(2)
    goal_cost = functools.partial(reference_cost, _x_goal=x_goal, _Q=Q, _R=R, _Qf=Qf)
    
    ilqr = control2.iLQR(unicycle_continuous, goal_cost, x.numel(), 2, dt, N)
    X, U, J = ilqr.solve(x)
    plot_solve(X, J, x_goal.numpy())


# ## multi-agent problem
def dynamics_nd(f, x, u, _x_dims, _u_dims):
    """Compute the continuous time derivative for n agents"""
    assert isinstance(x, torch.Tensor) and isinstance(u, torch.Tensor)
    return torch.cat([
        f(xi.flatten(), ui.flatten()) 
        for xi, ui in zip(split_agents(x, _x_dims), split_agents(u, _u_dims))
    ])


def proximity_cost(x, _x_dims, _radius):
    """Penalizes distances underneath some radius between agents"""
    assert len(set(_x_dims)) == 1

    assert torch.is_tensor(x)
    
    n_agents = len(_x_dims)
    n_states = _x_dims[0]

    pair_inds = torch.tensor(list(itertools.combinations(range(n_agents), 2)))
    x_agent = x.reshape(-1,n_agents,n_states).swapaxes(0,2)
    distances = torch.linalg.norm(x_agent[:2,pair_inds[:,0]] - x_agent[:2,pair_inds[:,1]], dim=0)

    pair_costs = torch.fmin(torch.zeros((1)), distances - _radius)**2
    return pair_costs.sum(dim=0)


def multi_agent_cost(goal_cost, prox_cost, x, u, terminal=False):
    """Reference deviation costs plus collision avoidance costs"""
    return goal_cost(x, u, terminal=terminal) + 50*prox_cost(x)


def multi_agent_sim():
    dt = 0.1
    N = 60
    
    x_dims = [4, 4]
    u_dims = [2, 2]
    x = torch.tensor([-5, -5, 0, 0,
                      -5,  5, 0, 0], 
                     dtype=torch.float, requires_grad=True)
    x_goal = torch.tensor([[5,  5, 0, 0,
                            5, -4, 0, 0]],
                          dtype=torch.float).T
    
    dynamics_8d = functools.partial(dynamics_nd, unicycle_continuous, _x_dims=x_dims, _u_dims=u_dims)
    
    Q = torch.diag(torch.tensor([1., 1, 0, 0]).tile(2))
    Qf = 1000 * torch.eye(Q.shape[0])
    R = torch.eye(4)
    radius = 0.5
    
    goal_cost = functools.partial(reference_cost, _x_goal=x_goal, _Q=Q, _R=R, _Qf=Qf)
    prox_cost = functools.partial(proximity_cost, _x_dims=x_dims, _radius=radius)
    multi_cost = functools.partial(multi_agent_cost, goal_cost, prox_cost)
    
    # %%prun
    ilqr = control2.iLQR(dynamics_8d, multi_cost, x.numel(), 4, dt, N)
    X, U, J = ilqr.solve(x)


def main():
    # single_agent_sim()
    multi_agent_sim()


if __name__ == "__main__":
    main()
    
    
    
