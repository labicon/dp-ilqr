#!/usr/bin/env python

"""Dynamics module to simulate dynamical systems with examples"""

import abc

import numpy as np
from scipy.linalg import block_diag
import torch

from .util import split_agents


class DynamicalModel(abc.ABC):
    """Simulation of a dynamical model to be applied in the iLQR solution."""

    def __init__(self, n_x, n_u, dt):
        self.n_x = n_x
        self.n_u = n_u
        self.dt = dt

    def __call__(self, x, u):
        """Zero-order hold to integrate continuous dynamics f"""
        return x + self.f(x, u) * self.dt

    @abc.abstractmethod
    def f():
        """Continuous derivative of dynamics with respect to time"""
        pass

    @abc.abstractmethod
    def linearize():
        """Jacobian linearization with respect to states and controls"""
        pass


class AutoDiffModel(DynamicalModel):
    """Mix-in to use torch automatic differentiation for linearization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.NX_EYE = np.eye(self.n_x, dtype=np.float32)

    def linearize(self, x, u, discrete=False):
        """Compute the Jacobian linearization of the dynamics for a particular state
           and controls for all players.
        """

        assert torch.is_tensor(x) == torch.is_tensor(u)
        convert_to_torch = not torch.is_tensor(x) and not torch.is_tensor(u)

        if convert_to_torch:
            x = torch.from_numpy(x).requires_grad_(True)
            u = torch.from_numpy(u).requires_grad_(True)

        A, B = torch.autograd.functional.jacobian(self.f, (x, u))

        if convert_to_torch:
            A = A.detach().numpy()
            B = B.detach().numpy()

        if discrete:
            return A, B

        # Compute the discretized jacobians with euler integration.
        A = self.dt * A.reshape(self.n_x, self.n_x) + self.NX_EYE
        B = self.dt * B.reshape(self.n_x, self.n_u)
        return A, B


class MultiDynamicalModel(AutoDiffModel):
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
        mm = torch if torch.is_tensor(x) else np
        return mm.stack(
            [
                submodel.f(xi.flatten(), ui.flatten())
                for submodel, xi, ui in zip(self.submodels, *self.partition(x, u))
            ]
        ).flatten()

    def partition(self, x, u):
        """Helper to split up the states and control for each subsystem"""
        return split_agents(x, self.x_dims), split_agents(u, self.u_dims)

    def _linearize_arbitary(self, x, u):
        """Compute the submodel-linearizations (deprecated)
           NOTE: This function is an alternate way to do the linearization rather than
           using the AutoDiffModel mix-in.
        """

        sub_linearizations = [
            submodel.linearize(xi.flatten(), ui.flatten())
            for submodel, xi, ui in zip(self.submodels, *self.partition(x, u))
        ]

        sub_As = [AB[0] for AB in sub_linearizations]
        sub_Bs = [AB[1] for AB in sub_linearizations]

        return block_diag(*sub_As), block_diag(*sub_Bs)
