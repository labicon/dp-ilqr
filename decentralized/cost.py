#!/usr/bin/env python

"""Implements various cost structures in the LQ Game"""

import abc

import numpy as np
import torch
from scipy.optimize import approx_fprime

from .util import compute_pairwise_distance, split_agents


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

    def __init__(self, xf, Q, R, Qf=None):

        if Qf is None:
            Qf = torch.eye(Q.shape[0])

        # Define states as rows so that xf doesn't broadcast x in __call__.
        self.xf = xf.reshape(1, -1)

        self.Q = Q
        self.R = R
        self.Qf = Qf

    @property
    def x_dim(self):
        return self.Q.shape[0]

    @property
    def u_dim(self):
        return self.R.shape[0]

    def __call__(self, x, u, terminal=False):
        if not terminal:
            return (x - self.xf) @ self.Q @ (x - self.xf).T + u @ self.R @ u.T
        return (x - self.xf) @ self.Qf @ (x - self.xf).T

    def __repr__(self):
        return f"ReferenceCost(\n\tQ: {self.Q},\n\tR: {self.R},\n\tQf: {self.Qf}\n)"


class ProximityCost(Cost):
    def __init__(self, x_dims, radius):
        self.x_dims = x_dims
        self.radius = radius

    def __call__(self, x):
        """Penalizes distances underneath some radius between agents"""
        distances = compute_pairwise_distance(x, self.x_dims)
        pair_costs = torch.fmin(torch.zeros((1)), distances - self.radius) ** 2
        return pair_costs.sum(dim=0)

    def __repr__(self):
        return f"ProximityCost(\n\tx_dims: {self.x_dims},\n\tradius: {self.radius}\n)"


class GameCost(Cost):
    def __init__(self, reference_costs, proximity_cost=None):

        if not proximity_cost:

            def proximity_cost(*_):
                return 0.0

        self.ref_costs = reference_costs
        self.prox_cost = proximity_cost

        self.REF_WEIGHT = 1.0
        self.PROX_WEIGHT = 100.0

    def __call__(self, x, u, terminal=False):
        x_dims = [ref_cost.x_dim for ref_cost in self.ref_costs]
        u_dims = [ref_cost.u_dim for ref_cost in self.ref_costs]
        x_split = split_agents(x, x_dims)
        u_split = split_agents(u, u_dims)

        ref_total = 0.0
        for ref_cost, xi, ui in zip(self.ref_costs, x_split, u_split):
            ref_total += ref_cost(xi, ui, terminal)[0]

        return self.PROX_WEIGHT * self.prox_cost(x) + self.REF_WEIGHT * ref_total


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
