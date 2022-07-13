#!/usr/bin/env python

"""Dynamics module to simulate dynamical systems with examples"""

import abc
from hamcrest import none
import sympy as sym
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import approx_fprime
import torch

from util import split_agents #relative import doesn't work on my computer?


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

    @classmethod
    def _reset_ids(cls):
        cls._id = 0

    def __repr__(self):
        return f"{type(self).__name__}(n_x: {self.n_x}, n_u: {self.n_u}, id: {self.id})"


class AnalyticalModel(DynamicalModel):
    """Mix-in for analytical linearization"""
    
    def __init__(self, A_num, B_num, L_x, L_u, L_xx, L_uu, L_ux):
    
        self.A_num = A_num
        self.B_num = B_num
        self.L_x = L_x
        self.L_u = L_u
        self.L_xx = L_xx
        self.L_uu = L_uu
        self.L_ux = L_ux

    def linearize(self, x, u): 
        
        return self.A_num(x, u), self.B_num(x, u)

    def quadraticize(self, x, u):
        
        return self.L_x(x,u), self.L_u(x,u), self.L_xx(x,u), self.L_uu(x,u), self.L_ux(x,u)

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


class DoubleIntDynamics4D(DynamicalModel):
    def __init__(self, dt, *args, **kwargs):
        super().__init__(4, 2, dt, *args, **kwargs)

    @staticmethod
    def f(x, u):
        *_, vx, vy = x
        ax, ay = u
        return torch.stack([vx, vy, ax, ay])


class CarDynamics3D(DynamicalModel):
    def __init__(self, dt, *args, **kwargs):
        super().__init__(3, 2, dt, *args, **kwargs)

    @staticmethod
    def f(x, u):
        *_, theta = x
        v, omega = u
        return torch.stack([v * torch.cos(theta), v * torch.sin(theta), omega])


class UnicycleDynamics4D(DynamicalModel):
    def __init__(self, dt, *args, **kwargs):
        super().__init__(4, 2, dt, *args, **kwargs)

    @staticmethod
    def f(x, u):
        *_, theta, v = x
        a, omega = u
        return torch.stack([v * torch.cos(theta), v * torch.sin(theta), a, omega])


class BikeDynamics5D(DynamicalModel):
    def __init__(self, dt, *args, **kwargs):
        super().__init__(5, 2, dt)

    @staticmethod
    def f(x, u):
        *_, theta, v, phi = x
        a, phi_dot = u
        return torch.stack(
            [v * torch.cos(theta), v * torch.sin(theta), torch.tan(phi), a, phi_dot]
        )

class QuadcopterDynamics(DynamicalModel):
    def __init__(self, dt, *args, **kwargs):
        super().__init__(12, 4, dt)
        
    @staticmethod
    def f(x,u):
        # components of position (meters)
        o_x, o_y, o_z = sym.symbols('o_x, o_y, o_z')

        # yaw, pitch, and roll angles (radians)
        psi, theta, phi = sym.symbols('psi, theta, phi')

        # components of linear velocity (meters / second)
        v_x, v_y, v_z = sym.symbols('v_x, v_y, v_z')

        # components of angular velocity (radians / second)
        w_x, w_y, w_z = sym.symbols('w_x, w_y, w_z')

        # components of net rotor torque
        tau_x, tau_y, tau_z = sym.symbols('tau_x, tau_y, tau_z')

        # net rotor force
        f_z = sym.symbols('f_z')

        x = np.array([o_x,o_y,o_z,psi,theta,phi,v_x,v_y,v_z,w_x,w_y,w_z])
        
        u = np.array([tau_x, tau_y, tau_z, f_z])
        
        # m = sym.nsimplify(0.0315) #mass of a Crazyflie drone

        # # Principle moments of inertia of a Crazyflie drone
        # J_x = sym.nsimplify(1.7572149113694408e-05)
        # J_y = sym.nsimplify(1.856979710568617e-05)
        # J_z = sym.nsimplify(2.7159794713754086e-05)

        # # Acceleration of gravity
        # g = 9.81

        # #Linear and angular velocity vectors (in body frame)
        # v_01in1 = sym.Matrix([[v_x], [v_y], [v_z]])
        # w_01in1 = sym.Matrix([[w_x], [w_y], [w_z]])
        
        # #Create moment of inertia matrix (in coordinates of the body frame).
        # J_in1 = sym.diag(J_x, J_y, J_z)

        # Rz = sym.Matrix([[sym.cos(psi), -sym.sin(psi), 0],
        #          [sym.sin(psi), sym.cos(psi), 0],
        #          [0, 0, 1]])

        # Ry = sym.Matrix([[sym.cos(theta), 0, sym.sin(theta)],
        #                 [0, 1, 0],
        #                 [-sym.sin(theta), 0, sym.cos(theta)]])

        # Rx = sym.Matrix([[1, 0, 0],
        #                 [0, sym.cos(phi), -sym.sin(phi)],
        #                 [0, sym.sin(phi), sym.cos(phi)]])


        # R_1in0 = Rz * Ry * Rx

        # #Mapping from angular velocity to angular rates
        # Ninv = sym.Matrix.hstack((Ry * Rx).T * sym.Matrix([[0], [0], [1]]),
        #                       (Rx).T * sym.Matrix([[0], [1], [0]]),
        #                                sym.Matrix([[1], [0], [0]]))
        # N = sym.simplify(Ninv.inv()) #this matrix N is what we actually want

        # #forces
        # f_in1 = R_1in0.T * sym.Matrix([[0], [0], [-m * g]]) + sym.Matrix([[0], [0], [f_z]])

        # #torques
        # tau_in1 = sym.Matrix([[tau_x], [tau_y], [tau_z]])

        #EOM:
        # f_sym = sym.Matrix.vstack(R_1in0 * v_01in1,
        #                   N * w_01in1,
        #                   (1 / m) * (f_in1 - w_01in1.cross(m * v_01in1)),
        #                   J_in1.inv() * (tau_in1 - w_01in1.cross(J_in1 * w_01in1)))

        #Full equations of motion : See derivation above
        f_sym = torch.stack([
                  x[6]*sym.cos(x[3])*sym.cos(x[4]) + x[7]*(sym.sin(x[5])*sym.sin(x[4])*sym.cos(x[3])-sym.sin(x[3])*sym.cos(x[5])) + x[8]*(sym.sin(x[5])*sym.sin(x[3])+sym.sin(x[4])*sym.cos(x[5])*sym.cos(x[3])),
                  x[6]*sym.sin(x[3])*sym.cos(x[4]) + x[7]*(sym.sin(x[5])*sym.sin(x[3])*sym.sin(x[4])+sym.cos(x[5])*sym.cos(x[3])) + x[8]*(-sym.sin(x[5])*sym.cos(x[3])+sym.sin(x[3])*sym.sin(x[4])*sym.cos(x[5])),
                  -x[6]*sym.sin(x[4])+x[7]*sym.sin(x[5])*sym.cos(x[4]) + x[8]*sym.cos(x[5])*sym.cos(x[4]),
                  x[10]*sym.sin(x[5])/sym.cos(x[4]) + x[11]*sym.cos(x[5])/sym.cos(x[4]),
                  x[10]*sym.cos(x[5])-x[11]*sym.sin(x[5]),
                  x[9] + x[10]*sym.sin(x[5])*sym.tan(x[4]) + x[11] * sym.cos(x[5]) * sym.tan(x[4]),
                  x[7]*x[11]-x[8]*x[10] + 9.81 * sym.sin(x[4]),
                  -x[6]*x[11] + x[8]*x[9] - 9.81 * sym.sin(x[5])*sym.cos(x[4]),
                  2000/63 * u[3] + x[6]*x[10] - x[7]*x[9] - 9.81 * sym.cos(x[5])*sym.cos(x[4]),
                  625000000000000000/10982593196059 * u[0] - (85899976080679/175721491136944) * x[10]*x[11],
                  5000000000000000000/92848985528431 * u[1] + (95876456000597/185697971056862) * x[9]*x[11],
                  10000000000000000000/271597947137541 * u[2] - (9976479919918/271597947137541) * x[9]*x[10]
                  ])

        return f_sym 

def linearize_analytical(f, x, u):

    "Linearization of dynamics using SYMPY symbolic differentiation"
    f = f.__func__(x,u)
    A_num = sym.lambdify(x, f.jacobian(x))
    B_num = sym.lambdify(u, f.jacobain(u))

    return A_num, B_num



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


