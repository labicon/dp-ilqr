#!/usr/bin/env python

"""Dynamics module to simulate dynamical systems with examples"""

import abc

import numpy as np
from scipy.optimize import approx_fprime
from scipy.integrate import solve_ivp
import sympy as sym

from .util import split_agents_gen, uniform_block_diag
from .bbdynamicswrap import integrate, linearize, f, Model

np.set_printoptions(precision=3)


def rk4_integration(f, x0, u, h, dh=None):
    """Classic Runge-Kutta Method with sub-integration"""

    if not dh:
        dh = h

    t = 0.0
    x = x0.copy()

    while t < h - 1e-8:
        step = min(dh, h - t)

        k0 = f(x, u)
        k1 = f(x + 0.5 * k0 * step, u)
        k2 = f(x + 0.5 * k1 * step, u)
        k3 = f(x + k2 * step, u)

        x += step * (k0 + 2.0 * k1 + 2.0 * k2 + k3) / 6.0
        t += step

    return x


def forward_euler_integration(f, x, u, h):
    """Simple 1st Order Method to integrate f with step size h"""
    return x + f(x, u) * h


def scipy_integration(f, x, u, h, **kwargs):
    sol = solve_ivp(lambda _, x, u: f(x, u), [0, h], x, args=(u,), t_eval=[h], **kwargs)
    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.y.flatten()


class DynamicalModel(abc.ABC):
    """Simulation of a dynamical model to be applied in the iLQR solution."""

    _id = 0

    def __init__(self, n_x, n_u, dt, id=None):
        if not id:
            id = DynamicalModel._id
            DynamicalModel._id += 1

        self.n_x = n_x
        self.n_u = n_u
        self.dt = dt
        self.id = id
        self.NX_EYE = np.eye(self.n_x, dtype=np.float32)

    def __call__(self, x, u):
        """Zero-order hold to integrate continuous dynamics f"""

        # return forward_euler_integration(self.f, x, u, self.dt)
        return rk4_integration(self.f, x, u, self.dt, self.dt)
        # return scipy_integration(self.f, x, u, self.dt, method="RK23")

    @staticmethod
    def f():
        """Continuous derivative of dynamics with respect to time"""
        pass

    @abc.abstractmethod
    def linearize():
        """Linearization that computes jacobian at the current operating point"""
        pass

    @classmethod
    def _reset_ids(cls):
        cls._id = 0

    def __repr__(self):
        return f"{type(self).__name__}(n_x: {self.n_x}, n_u: {self.n_u}, id: {self.id})"


class SymbolicModel(DynamicalModel):
    """Mix-in for analytical linearization"""

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["A_num"]
        del state["B_num"]
        del state["_f"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__init__(self.dt)

    def f(self, x, u):
        return self._f(x, u)

    def linearize(self, x, u):
        """Linearization via numerical Jacobians A_num and B_num with Euler method"""
        return np.eye(x.size) + self.dt * self.A_num(x, u), self.dt * self.B_num(x, u)


class CppModel(DynamicalModel):
    """Implementation using a C++ dynamics library via Cython"""

    def __init__(self, dt, *args, **kwargs):
        super().__init__(dt, *args, **kwargs)

    def __call__(self, x, u):
        return integrate(x, u, self.dt, self.model)

    def f(self, x, u):
        return f(x, u, self.model)

    def linearize(self, x, u):
        return linearize(x, u, self.dt, self.model)


class MultiDynamicalModel(DynamicalModel):
    """Encompasses the dynamical simulation and linearization for a collection of
    DynamicalModel's
    """

    def __init__(self, submodels):
        self.submodels = submodels
        self.n_players = len(submodels)

        self.x_dims = [submodel.n_x for submodel in submodels]
        self.u_dims = [submodel.n_u for submodel in submodels]
        self.ids = [submodel.id for submodel in submodels]

        super().__init__(sum(self.x_dims), sum(self.u_dims), submodels[0].dt, -1)

    def f(self, x, u):
        """Derivative of the current combined states and controls"""
        xn = np.zeros_like(x)
        nx = self.x_dims[0]
        nu = self.u_dims[0]
        for i, model in enumerate(self.submodels):
            xn[i * nx : (i + 1) * nx] = model.f(
                x[i * nx : (i + 1) * nx], u[i * nu : (i + 1) * nu]
            )
        return xn

    def __call__(self, x, u):
        """Zero-order hold to integrate continuous dynamics f"""

        # return forward_euler_integration(self.f, x, u, self.dt)
        # return rk4_integration(self.f, x, u, self.dt, self.dt)
        xn = np.zeros_like(x)
        nx = self.x_dims[0]
        nu = self.u_dims[0]
        for i, model in enumerate(self.submodels):
            xn[i * nx : (i + 1) * nx] = model.__call__(
                x[i * nx : (i + 1) * nx], u[i * nu : (i + 1) * nu]
            )
        return xn

    def linearize(self, x, u):
        sub_linearizations = [
            submodel.linearize(xi.flatten(), ui.flatten())
            for submodel, xi, ui in zip(
                self.submodels,
                split_agents_gen(x, self.x_dims),
                split_agents_gen(u, self.u_dims),
            )
        ]

        sub_As = [AB[0] for AB in sub_linearizations]
        sub_Bs = [AB[1] for AB in sub_linearizations]

        return uniform_block_diag(*sub_As), uniform_block_diag(*sub_Bs)

    def split(self, graph):
        """Split this model into submodels dictated by the interaction graph"""
        split_dynamics = []
        for problem in graph:
            split_dynamics.append(
                MultiDynamicalModel(
                    [model for model in self.submodels if model.id in graph[problem]]
                )
            )

        return split_dynamics

    def __repr__(self):
        sub_reprs = ",\n\t".join([repr(submodel) for submodel in self.submodels])
        return f"MultiDynamicalModel(\n\t{sub_reprs}\n)"


class DoubleIntDynamics4D(CppModel):
    def __init__(self, dt, *args, **kwargs):
        super().__init__(4, 2, dt, *args, **kwargs)
        self.model = Model.DoubleInt4D


class CarDynamics3D(CppModel):
    def __init__(self, dt, *args, **kwargs):
        super().__init__(3, 2, dt, *args, **kwargs)
        self.model = Model.Car3D


class UnicycleDynamics4D(CppModel):
    def __init__(self, dt, *args, **kwargs):
        super().__init__(4, 2, dt, *args, **kwargs)
        self.model = Model.Unicycle4D


class QuadcopterDynamics6D(CppModel):
    def __init__(self, dt, *args, **kwargs):
        super().__init__(6, 3, dt, *args, **kwargs)
        self.model = Model.Quadcopter6D


class QuadcopterDynamics12D(CppModel):
    def __init__(self, dt, *args, **kwargs):
        super().__init__(12, 4, dt, *args, **kwargs)
        self.model = Model.Quadcopter12D


class HumanDynamics6D(CppModel):
    def __init__(self, dt, *args, **kwargs):
        super().__init__(6, 3, dt, *args, **kwargs)
        self.model = Model.Human6D


# TODO: Consider making a CPP model for these two:
class BikeDynamics5D(SymbolicModel):
    def __init__(self, dt, *args, **kwargs):
        super().__init__(5, 2, dt, *args, **kwargs)

        p_x, p_y, theta, v, phi, a, rho = sym.symbols("p_x p_y theta v phi a rho")
        x = sym.Matrix([p_x, p_y, v, theta, phi])
        u = sym.Matrix([a, rho])

        x_dot = sym.Matrix(
            [
                x[2] * sym.cos(x[3]),
                x[2] * sym.sin(x[3]),
                u[0],
                x[2] * sym.tan(x[4]),
                u[1],
            ]
        )

        A = x_dot.jacobian(x)
        B = x_dot.jacobian(u)

        self._f = sym.lambdify((x, u), sym.Array(x_dot)[:, 0])
        self.A_num = sym.lambdify((x, u), A)
        self.B_num = sym.lambdify((x, u), B)


# Based off of https://github.com/anassinator/ilqr/blob/master/ilqr/dynamics.py
def linearize_finite_difference(f, x, u):
    """Linearization using finite difference"""

    n_x = x.size
    jac_eps = np.sqrt(np.finfo(float).eps)

    A = np.vstack([approx_fprime(x, lambda x: f(x, u)[i], jac_eps) for i in range(n_x)])
    B = np.vstack([approx_fprime(u, lambda u: f(x, u)[i], jac_eps) for i in range(n_x)])

    return A, B
