#!/usr/bin/env python

"""Implements various cost structures in the LQ Game"""

import abc

import numpy as np
from scipy.optimize import approx_fprime

from .util import Point, compute_pairwise_distance, split_agents_gen, uniform_block_diag


class Cost(abc.ABC):
    """
    Abstract base class for cost objects.
    """

    @abc.abstractmethod
    def __call__(self, *args):
        """Returns the cost evaluated at the given state and control"""
        pass

    @abc.abstractmethod
    def quadraticize():
        """Compute the jacobians and hessians of the operating point wrt. the states
        and controls
        """
        pass


class ReferenceCost(Cost):
    """
    The cost of a state and control from some reference trajectory.
    """

    _id = 0

    def __init__(self, xf, Q, R, Qf=None, id=None):

        if Qf is None:
            Qf = np.eye(Q.shape[0])

        if not id:
            id = ReferenceCost._id
            ReferenceCost._id += 1

        # Define states as rows so that xf doesn't broadcast x in __call__.
        # self.xf = xf.reshape(1, -1)
        self.xf = xf.flatten()

        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.id = id

        self.Q_plus_QT = Q + Q.T
        self.R_plus_RT = R + R.T
        self.nx = Q.shape[0]
        self.nu = R.shape[0]

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

    def quadraticize(self, x, u, terminal=False):
        x = x.flatten()
        u = u.flatten()

        L_x = (x - self.xf).T @ self.Q_plus_QT
        L_u = u.T @ self.R_plus_RT
        L_xx = self.Q_plus_QT
        L_uu = self.R_plus_RT
        L_ux = np.zeros((self.nu, self.nx))

        if terminal:
            L_x = (x - self.xf).T @ (self.Qf + self.Qf.T)
            L_xx = self.Qf + self.Qf.T
            L_u = np.zeros((self.nu))
            L_uu = np.zeros((self.nu, self.nu))

        return L_x, L_u, L_xx, L_uu, L_ux

    def __repr__(self):
        return (
            f"ReferenceCost(\n\tQ: {self.Q},\n\tR: {self.R},\n\tQf: {self.Qf}"
            f",\n\tid: {self.id}\n)"
        )


class ProximityCost(Cost):
    def __init__(self, x_dims, radius, n_dim):
        self.x_dims = x_dims
        self.radius = radius
        self.n_dim = n_dim
        self.n_agents = len(x_dims)

    def __call__(self, x, *_):
        if len(self.x_dims) == 1:
            return 0.0
        distances = compute_pairwise_distance(x, self.x_dims)
        pair_costs = np.fmin(np.zeros(1), distances - self.radius) ** 2
        return pair_costs.sum(axis=0)

    def quadraticize(self, x, *_):
        nx = sum(self.x_dims)
        nx_per_agent = self.x_dims[0]
        L_x = np.zeros((nx))
        L_xx = np.zeros((nx, nx))

        if self.n_dim == 2:
            for i in range(self.n_agents):
                for j in range(i + 1, self.n_agents):
                    L_xi = np.zeros((nx))
                    L_xxi = np.zeros((nx, nx))
                    
                    ix, iy = nx_per_agent * i, nx_per_agent * i + 1
                    jx, jy = nx_per_agent * j, nx_per_agent * j + 1

                    L_x_pair, L_xx_pair = quadraticize_distance_2d(
                        Point(*x[..., ix : iy + 1]),
                        Point(*x[..., jx : jy + 1]),
                        self.radius,
                    )

                    L_xi[np.array([ix, iy])] = +L_x_pair
                    L_xi[np.array([jx, jy])] = -L_x_pair

                    L_xxi[ix : iy + 1, ix : iy + 1] = +L_xx_pair
                    L_xxi[jx : jy + 1, jx : jy + 1] = +L_xx_pair
                    L_xxi[ix : iy + 1, jx : jy + 1] = -L_xx_pair
                    L_xxi[jx : jy + 1, ix : iy + 1] = -L_xx_pair

                    L_x += L_xi
                    L_xx += L_xxi

            return L_x, None, L_xx, None, None

        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                L_xi = np.zeros((nx))
                L_xxi = np.zeros((nx, nx))

                ix = nx_per_agent * i
                iy = ix + 1
                iz = ix + 2
                jx = nx_per_agent * j
                jy = jx + 1
                jz = jx + 2

                L_x_pair, L_xx_pair = quadraticize_distance_3d(
                    Point(*x[..., ix : iz + 1]),
                    Point(*x[..., jx : jz + 1]),
                    self.radius,
                )

                L_xi[np.array([ix, iy, iz])] = +L_x_pair
                L_xi[np.array([jx, jy, jz])] = -L_x_pair

                L_xxi[ix : iz + 1, ix : iz + 1] = +L_xx_pair
                L_xxi[jx : jz + 1, jx : jz + 1] = +L_xx_pair
                L_xxi[ix : iz + 1, jx : jz + 1] = -L_xx_pair
                L_xxi[jx : jz + 1, ix : iz + 1] = -L_xx_pair
                
                L_x += L_xi
                L_xx += L_xxi

        return L_x, None, L_xx, None, None


class GameCost(Cost):
    def __init__(self, reference_costs, proximity_cost=None):

        if not proximity_cost:

            def proximity_cost(_):
                return 0.0

        self.ref_costs = reference_costs
        self.prox_cost = proximity_cost

        self.REF_WEIGHT = 1.0
        self.PROX_WEIGHT = 200.0

        self.x_dims = [ref_cost.x_dim for ref_cost in self.ref_costs]
        self.u_dims = [ref_cost.u_dim for ref_cost in self.ref_costs]
        self.ids = [ref_cost.id for ref_cost in self.ref_costs]
        self.n_agents = len(reference_costs)

    @property
    def xf(self):
        return np.concatenate([ref_cost.xf for ref_cost in self.ref_costs])

    def __call__(self, x, u, terminal=False):
        ref_total = 0.0
        for ref_cost, xi, ui in zip(
            self.ref_costs,
            split_agents_gen(x, self.x_dims),
            split_agents_gen(u, self.u_dims),
        ):
            ref_total += ref_cost(xi, ui, terminal)

        return self.PROX_WEIGHT * self.prox_cost(x) + self.REF_WEIGHT * ref_total

    def quadraticize(self, x, u, terminal=False):
        L_xs, L_us = [], []
        L_xxs, L_uus, L_uxs = [], [], []

        # Compute agent quadraticizations in individual state spaces.
        for ref_cost, xi, ui in zip(
            self.ref_costs,
            split_agents_gen(x, self.x_dims),
            split_agents_gen(u, self.u_dims),
        ):
            L_xi, L_ui, L_xxi, L_uui, L_uxi = ref_cost.quadraticize(
                xi.flatten(), ui.flatten(), terminal
            )
            L_xs.append(L_xi)
            L_us.append(L_ui)
            L_xxs.append(L_xxi)
            L_uus.append(L_uui)
            L_uxs.append(L_uxi)

        L_x = self.REF_WEIGHT * np.hstack(L_xs)
        L_u = self.REF_WEIGHT * np.hstack(L_us)
        L_xx = self.REF_WEIGHT * uniform_block_diag(*L_xxs)
        L_uu = self.REF_WEIGHT * uniform_block_diag(*L_uus)
        L_ux = self.REF_WEIGHT * uniform_block_diag(*L_uxs)

        # Incorporate coupling costs in full cartesian state space.
        if self.n_agents > 1:
            L_x_prox, _, L_xx_prox, _, _ = self.prox_cost.quadraticize(x, u)
            L_x += self.PROX_WEIGHT * L_x_prox
            L_xx += self.PROX_WEIGHT * L_xx_prox

        return L_x, L_u, L_xx, L_uu, L_ux

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


def quadraticize_distance_2d(point_a, point_b, radius):
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

    L_xx[np.diag_indices(2)] = (
        2 * radius * np.array([dx, dy]) ** 2 / distance**3 - 2 * radius / distance + 2
    )
    L_xx[0, 1] = L_xx[1, 0] = (
        2
        * radius
        * dx
        * dy
        / np.sqrt(
            point_a.hypot2()
            + point_b.hypot2()
            - 2 * (point_b.x * point_a.x + point_b.y * point_a.y)
        )
        ** 3
    )

    return L_x, L_xx


def quadraticize_distance_3d(point_a, point_b, radius):
    """3D analog to the previous function"""

    assert point_a.ndim == point_b.ndim

    L_x = np.zeros((3))
    L_xx = np.zeros((3, 3))

    dx = point_a.x - point_b.x
    dy = point_a.y - point_b.y
    dz = point_a.z - point_b.z
    distance = np.sqrt(dx*dx + dy*dy + dz*dz)

    if distance > radius:
        return L_x, L_xx

    L_x = 2 * (distance - radius) / distance * np.array([dx, dy, dz])

    L_xx[np.diag_indices(3)] = (
        2 * radius * np.array([dx, dy, dz]) ** 2 / distance**3
        - 2 * radius / distance
        + 2
    )

    cross_factors = (
        2
        * radius
        / np.sqrt(
            point_a.hypot2()
            + point_b.hypot2()
            - 2
            * (point_a.x * point_b.x + point_a.y * point_b.y + point_a.z * point_b.z)
        )
        ** 3
    )

    L_xx[0, 1] = L_xx[1, 0] = dx * dy * cross_factors
    L_xx[0, 2] = L_xx[2, 0] = dx * dz * cross_factors
    L_xx[1, 2] = L_xx[2, 1] = dy * dz * cross_factors

    return L_x, L_xx


def quadraticize_finite_difference(cost, x, u, terminal=False, jac_eps=None):
    """Finite difference quadraticized cost

    NOTE: deprecated in favor of automatic differentiation in lieu of speed and
    consistency.
    """
    if not jac_eps:
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
