#!/usr/bin/env python

"""Implements various cost structures in the LQ Game"""

import abc

import numpy as np
import torch
from scipy.optimize import approx_fprime

from decentralized.util import Point, compute_pairwise_distance, split_agents


class Cost(abc.ABC):
    """
    Abstract base class for cost objects.
    """

    @abc.abstractmethod
    def __call__(self, *args):
        """Returns the cost evaluated at the given state and control"""
        pass

    def quadraticize(self, x, u, **kwargs):
        """
        Compute a quadratic approximation to the overall cost for a
        particular choice of state `x`, and controls `u` for each player.
        Returns the gradient and Hessian of the overall cost such that:
        ```
        cost(x + dx, [ui + dui]) \approx
                cost(x, u1, u2) +
                grad_x^T dx +
                0.5 * (dx^T hess_x dx + sum_i dui^T hess_ui dui)
        ```
        REF: [1]
        """

        n_x = x.numel()
        n_u = u.numel()

        def cost_fn(x, u):
            return self.__call__(x, u, **kwargs)

        L_x, L_u = torch.autograd.functional.jacobian(cost_fn, (x, u))
        L_x = L_x.reshape(n_x)
        L_u = L_u.reshape(n_u)

        (L_xx, _), (L_ux, L_uu) = torch.autograd.functional.hessian(cost_fn, (x, u))
        L_xx = L_xx.reshape(n_x, n_x)
        L_ux = L_ux.reshape(n_u, n_x)
        L_uu = L_uu.reshape(n_u, n_u)

        return L_x, L_u, L_xx, L_uu, L_ux


class ReferenceCost(Cost):
    """
    The cost of a state and control from some reference trajectory.
    """

    _id = 0

    def __init__(self, xf, Q, R, Qf=None, id=None):

        if Qf is None:
            Qf = torch.eye(Q.shape[0])

        if not id:
            id = ReferenceCost._id
            ReferenceCost._id += 1

        # Define states as rows so that xf doesn't broadcast x in __call__.
        self.xf = xf.reshape(1, -1)

        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.id = id

    @property
    def x_dim(self):
        return self.Q.shape[0]

    @property
    def u_dim(self):
        return self.R.shape[0]

    @classmethod
    def _reset_ids(cls):
        cls._id = 0

    def __call__(self, x, u, terminal=False):
        if not terminal:
            u = u.reshape(1, -1)
            return (x - self.xf) @ self.Q @ (x - self.xf).T + u @ self.R @ u.T
        return (x - self.xf) @ self.Qf @ (x - self.xf).T

    def __repr__(self):
        return (
            f"ReferenceCost(\n\tQ: {self.Q},\n\tR: {self.R},\n\tQf: {self.Qf}"
            f",\n\tid: {self.id}\n)"
        )


class ProximityCost(Cost):
    def __init__(self, x_dims, radius):
        self.x_dims = x_dims
        self.radius = radius

    def __call__(self, x, *_):
        """Penalizes distances underneath some radius between agents"""
        if len(self.x_dims) == 1:
            return 0.0
        distances = compute_pairwise_distance(x, self.x_dims)
        pair_costs = torch.fmin(torch.zeros((1)), distances - self.radius) ** 2
        return pair_costs.sum(dim=0)

    def __repr__(self):
        return f"ProximityCost(x_dims: {self.x_dims}, radius: {self.radius})"


class CouplingCost(Cost):
    
    """
    Models the couplings between different agents in the MultiDynamicalModel sense, i.e.
    how should we penalize two agents in the aggregate state for their relative distance?
    
    NOTE: This logic assumes that interactions between agents are symmetric, such that we
    can add jacobians & hessians equally in both directions.
    """
    
    def __init__(self, x_dims, radius):
        self.x_dims = x_dims
        self.radius = radius
        self.n_agents = len(x_dims)
    
    def __call__(self, x):
        if len(self.x_dims) == 1:
            return 0.0
        distances = compute_pairwise_distance(x, self.x_dims)
        pair_costs = np.fmin(np.zeros(1), distances - self.radius) ** 2
        return pair_costs.sum(axis=0)
    
    def quadraticize(self, x):
        nx = sum(self.x_dims)
        nx_per_agent = self.x_dims[0]
        L_x = np.zeros((nx))
        L_xx = np.zeros((nx, nx))
        
        for i in range(self.n_agents):            
            for j in range(i+1, self.n_agents):
                L_xi = np.zeros((nx))
                L_xxi = np.zeros((nx, nx))
                
                L_x_pair, L_xx_pair = quadraticize_distance(
                    Point(*x[..., nx_per_agent*i : nx_per_agent*i+2]), 
                    Point(*x[..., nx_per_agent*j : nx_per_agent*j+2]), 
                    self.radius
                )
                
                ix, iy = nx_per_agent*i, nx_per_agent*i + 1
                jx, jy = nx_per_agent*j, nx_per_agent*j + 1
                
                L_xi[ix] = +L_x_pair[0]
                L_xi[jx] = -L_x_pair[0]
                L_xi[iy] = +L_x_pair[1]
                L_xi[jy] = -L_x_pair[1]
                L_xxi[ix,ix] = L_xxi[jx,jx] = L_xx_pair[0,0]
                L_xxi[iy,iy] = L_xxi[jy,jy] = L_xx_pair[1,1]
                L_xxi[ix,iy] = L_xxi[iy,ix] = L_xxi[jx,jy] = L_xxi[jy,jx] = L_xx_pair[0,1]
                
                L_x += L_xi
                L_xx += L_xxi
                
        return L_x, None, L_xx, None, None


class GameCost(Cost):
    def __init__(self, reference_costs, proximity_cost=None):

        if not proximity_cost:

            def proximity_cost(*_):
                return 0.0

        self.ref_costs = reference_costs
        self.prox_cost = proximity_cost

        self.REF_WEIGHT = 1.0
        self.PROX_WEIGHT = 200.0

        self.x_dims = [ref_cost.x_dim for ref_cost in self.ref_costs]
        self.u_dims = [ref_cost.u_dim for ref_cost in self.ref_costs]
        self.ids = [ref_cost.id for ref_cost in self.ref_costs]

    def __call__(self, x, u, terminal=False):
        x_split = split_agents(x, self.x_dims)
        u_split = split_agents(u, self.u_dims)

        ref_total = 0.0
        for ref_cost, xi, ui in zip(self.ref_costs, x_split, u_split):
            ref_total += ref_cost(xi, ui, terminal)[0]

        return self.PROX_WEIGHT * self.prox_cost(x) + self.REF_WEIGHT * ref_total

    def split(self, graph):
        """Split this model into sub game-costs dictated by the interaction graph"""

        # Assume all states and radii are the same between agents.
        n_states = self.ref_costs[0].x_dim
        radius = self.prox_cost.radius

        game_costs = []
        for problem in graph:
            goal_costs_i = [
                ref_cost for ref_cost in self.ref_costs if ref_cost.id in graph[problem]
            ]
            prox_cost_i = ProximityCost([n_states] * len(graph[problem]), radius)
            game_costs.append(GameCost(goal_costs_i, prox_cost_i))

        return game_costs

    def __repr__(self):
        ids = [ref_cost.id for ref_cost in self.ref_costs]
        return f"GameCost(\n\tids: {ids},\n\tprox_cost: {self.prox_cost}\n)"


def quadraticize_distance(point_a, point_b, radius):
    """Quadraticize the distance between two points thresholded by a radius,
       returning the corresponding 2x1 jacobian and 2x2 hessian.
       
       NOTE: we assume that the states are organized in matrix form as [x, y, ...]
       rather than [y, x].

    """
    
    assert point_a.ndim == point_b.ndim

    L_x = np.zeros((2))
    L_xx = np.zeros((2, 2))
    
    dx = point_a.x - point_b.x
    dy = point_a.y - point_b.y
    distance = np.hypot(dx, dy)

    if distance > radius:
        return L_x, L_xx
    
    L_x = 2 * (distance - radius) / distance * np.array([dx, dy])

    L_xx[0,0] = (
        2*radius*dx**2 / distance**3
      - 2*radius / distance 
      + 2
    )
    L_xx[0,1] = L_xx[1,0] = \
        2*radius*dx*dy / np.sqrt(
            point_b.x**2 - 2*point_b.x*point_a.x + point_b.y**2 
          + point_a.x**2 - 2*point_b.y*point_a.y + point_a.y**2
        ) ** 3
    L_xx[1,1] = (
        2*radius*dy**2 / distance**3
      - 2*radius / distance 
      + 2
    )

    return L_x, L_xx


def quadraticize_finite_difference(cost, x, u, terminal=False):
    """Finite difference quadraticized cost

    NOTE: deprecated in favor of automatic differentiation in lieu of speed and
    consistency.
    """
    jac_eps = np.sqrt(np.finfo(float).eps)
    hess_eps = np.sqrt(jac_eps)

    n_x = x.shape[0]
    n_u = u.shape[0]

    def Lx(x, u):
        return approx_fprime(x, lambda x: cost(x, u, terminal), jac_eps)

    def Lu(x, u):
        return approx_fprime(u, lambda u: cost(x, u, terminal), jac_eps)

    L_xx = np.vstack(
        [approx_fprime(x, lambda x: Lx(x, u)[i], hess_eps) for i in range(n_x)]
    )

    L_uu = np.vstack(
        [approx_fprime(u, lambda u: Lu(x, u)[i], hess_eps) for i in range(n_u)]
    )

    L_ux = np.vstack(
        [approx_fprime(x, lambda x: Lu(x, u)[i], hess_eps) for i in range(n_u)]
    )

    return Lx(x, u), Lu(x, u), L_xx, L_uu, L_ux
