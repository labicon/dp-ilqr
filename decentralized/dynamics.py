#!/usr/bin/env python

"""Dynamics module to simulate dynamical systems with examples"""

import abc

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import approx_fprime
import torch

from .util import split_agents


class DynamicalModel(abc.ABC):
    """Simulation of a dynamical model to be applied in the iLQR solution."""

    _id = 0

    def __init__(self, n_x, n_u, dt):
        self.n_x = n_x
        self.n_u = n_u
        self.dt = dt

        self.id = DynamicalModel._id
        DynamicalModel._id += 1

        self.NX_EYE = np.eye(self.n_x, dtype=np.float32)

    def __call__(self, x, u):
        """Zero-order hold to integrate continuous dynamics f"""
        return x + self.f(x, u) * self.dt

    @staticmethod
    @abc.abstractmethod
    def f():
        """Continuous derivative of dynamics with respect to time"""
        pass

    def linearize(self, x: torch.tensor, u: torch.tensor, discrete=False):
        """Compute the Jacobian linearization of the dynamics for a particular state
        and controls for all players.
        """

        A, B = torch.autograd.functional.jacobian(self.f, (x, u))

        if discrete:
            return A, B

        # Compute the discretized jacobians with euler integration.
        A = self.dt * A.reshape(self.n_x, self.n_x) + self.NX_EYE
        B = self.dt * B.reshape(self.n_x, self.n_u)
        return A, B


class MultiDynamicalModel(DynamicalModel):
    """Encompasses the dynamical simulation and linearization for a collection of
    DynamicalModel's
    """

    def __init__(self, submodels):
        self.submodels = submodels
        self.n_players = len(submodels)

        self.x_dims = [submodel.n_x for submodel in submodels]
        self.u_dims = [submodel.n_u for submodel in submodels]

        super().__init__(sum(self.x_dims), sum(self.u_dims), submodels[0].dt)

    def f(self, x, u):
        """Integrate the dynamics for the combined decoupled dynamical model"""
        return torch.cat(
            [
                submodel.f(xi.flatten(), ui.flatten())
                for submodel, xi, ui in zip(self.submodels, *self.partition(x, u))
            ]
        )

    def partition(self, x, u):
        """Helper to split up the states and control for each subsystem"""
        return split_agents(x, self.x_dims), split_agents(u, self.u_dims)


class DoubleIntDynamics4D(DynamicalModel):
    def __init__(self, dt):
        super().__init__(4, 2, dt)

    @staticmethod
    def f(x, u):
        *_, vx, vy = x
        ax, ay = u
        return torch.stack([vx, vy, ax, ay])


class CarDynamics3D(DynamicalModel):
    def __init__(self, dt):
        super().__init__(3, 2, dt)

    @staticmethod
    def f(x, u):
        *_, theta = x
        v, omega = u
        return torch.stack([v * torch.cos(theta), v * torch.sin(theta), omega])


class UnicycleDynamics4D(DynamicalModel):
    def __init__(self, dt):
        super().__init__(4, 2, dt)

    @staticmethod
    def f(x, u):
        *_, theta, v = x
        a, omega = u
        return torch.stack([v * torch.cos(theta), v * torch.sin(theta), a, omega])


class BikeDynamics5D(DynamicalModel):
    def __init__(self, dt):
        super().__init__(5, 2, dt)

    @staticmethod
    def f(x, u):
        *_, theta, v, phi = x
        a, phi_dot = u
        return torch.stack(
            [v * torch.cos(theta), v * torch.sin(theta), torch.tan(phi), a, phi_dot]
        )


# Based off of https://github.com/anassinator/ilqr/blob/master/ilqr/dynamics.py
def linearize_finite_difference(f, x, u):
    """ "Linearization using finite difference.

    NOTE: deprecated in favor of automatic differentiation.
    """

    n_x = x.size
    jac_eps = np.sqrt(np.finfo(float).eps)

    A = np.vstack([approx_fprime(x, lambda x: f(x, u)[i], jac_eps) for i in range(n_x)])
    B = np.vstack([approx_fprime(u, lambda u: f(x, u)[i], jac_eps) for i in range(n_x)])

    return A, B


def linearize_multi(submodels, partition, x, u):
    """Compute the submodel-linearizations

    NOTE: deprecated in favor of automatic differentiation.
    """

    sub_linearizations = [
        submodel.linearize(xi.flatten(), ui.flatten())
        for submodel, xi, ui in zip(submodels, *partition(x, u))
    ]

    sub_As = [AB[0] for AB in sub_linearizations]
    sub_Bs = [AB[1] for AB in sub_linearizations]

    return block_diag(*sub_As), block_diag(*sub_Bs)
