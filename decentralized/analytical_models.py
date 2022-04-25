#!/usr/bin/env python

"""Non-linear dynamics models with my attempt to linearize them analytically.

For some reason, these models aren't able to navigate the cost surface as well
as the ``NumericalDiffModel``'s.

NOTE: Deprecated in favor ``models.py``.
"""

import numpy as np

from .dynamics import DynamicalModel
from .models import V_LIMS, OMEGA_LIMS, ACC_LIMS


class CarDynamics(DynamicalModel):
    """Simplified UnicycleModel for modeling a car.
        state := [x position, y position, heading angle]
        control := [linear velocity, angular velocity]
        
        NOTE: These dynamics don't integrate intuitively, f is most likely
        the culprit.
    """
    
    name = "Car"
    
    def __init__(self, dt):
        super().__init__(3, 2, dt)
        
    @staticmethod
    def f(x, _, u):
        v = u[0]
        omega = u[1]
        theta = x[2]
        
        return np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            omega
        ])
    
    def linearize(self, x, u):
        
        # Advance dynamics in time for consistency with finite difference dynamics.
        # x = self.__call__(x, u)
        
        v = u[0]
        theta = x[2]
        
        # Analytical derivations ain't right.
        A = np.array([
            [1, 0, -v*self.dt*np.sin(theta)],
            [0, 1,  v*self.dt*np.cos(theta)],
            [0, 0,                        1]
        ])
        B = np.array([
            [np.cos(theta), -v*self.dt*np.sin(theta)],
            [np.sin(theta),  v*self.dt*np.cos(theta)],
            [            0,                        1]
        ]) * self.dt

        # A = np.eye(3)
        # B = np.array([
        #     [np.cos(theta), 0],
        #     [np.sin(theta), 0],
        #     [            0, 1]
        # ]) * self.dt
        
        return A, B
    
    def constrain(self, x, u):
        x[2] %= 2*np.pi
        u[0] = np.clip(u[0], *V_LIMS)
        u[1] = np.clip(u[1], *OMEGA_LIMS)
        return x, u


class UnicycleDynamics(DynamicalModel):
    """Simplified unicycle model for 2D trajectory planning.
        state := [x position, y position, linear velocity, heading angle]
        control := [linear acceleration, angular velocity]
    """
    
    name = "Unicycle"
    
    def __init__(self, dt):
        super().__init__(4, 2, dt)
        
    @staticmethod
    def f(x, _, u):
        v = x[2]
        theta = x[3]
        a = u[0]
        omega = u[1]
        
        return np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            a,
            omega
        ])
    
    def linearize(self, x, _):

        v = x[2]
        theta = x[3]

        A = np.array([
            [1, 0, self.dt*np.cos(theta), -self.dt*v*np.sin(theta)],
            [0, 1, self.dt*np.sin(theta),  self.dt*v*np.cos(theta)],
            [0, 0,                     1,                        0],
            [0, 0,                     0,                        1]
        ])

        # B = np.array([
        #     [self.dt*np.cos(theta), 0],
        #     [self.dt*np.sin(theta), 0],
        #     [                    1, 0],
        #     [                    0, 1]
        # ]) * self.dt
        
        B = np.array([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ], dtype=float) * self.dt
        
        return A, B
    
    def constrain(self, x, u):
        x[3] %= 2*np.pi
        u[0] = np.clip(u[0], *ACC_LIMS)
        u[1] = np.clip(u[1], *OMEGA_LIMS)
        return x, u
    

class BicycleDynamics(DynamicalModel):
    """Unicycle with steering component for front wheel.
        state := [x position, y position, heading, speed, steering angle]
        control := [acceleration, steering velocity]
    """
    
    name = "Bicycle"
    
    def __init__(self, dt):
        super().__init__(5, 2, dt)
        
    def __call__(self, x, u):
        # Forward Euler Method, since odeint isn't able to consistently 
        # integrate the dynamics. This was implemented explicitly to be consistent
        # with ilqr_driving.
        # u = self.constrain(u)
        return x + self.f(x, None, u)*self.dt
    
    @staticmethod
    def f(x, _, u):
        theta = x[2]
        v = x[3]
        phi = x[4]
        
        return np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            v * np.tan(phi),
            u[0],
            u[1]
        ])
    
    def linearize(self, x, u):
        theta = x[2]
        v = x[3]
        phi = x[4]
        dt = self.dt
        
        # NOTE: Analytical derivation doesn't agree with drake's dynamics
        # jacobians, use the experimentally derived linearizations instead.
        # A_03 = dt*(2*np.cos(theta) - dt*v*np.sin(theta)*np.tan(phi)) / 2
        # A_13 = dt*(2*np.sin(theta) + dt*v*np.cos(theta)*np.tan(phi)) / 2
        # A_04 = -dt**2 * v**2 * np.sin(theta) / (2 * np.cos(phi)**2)
        # A_14 =  dt**2 * v**2 * np.cos(theta) / (2 * np.cos(phi)**2)
        A_03 = dt*np.cos(theta)
        A_13 = dt*np.sin(theta)
        A_04 = 0
        A_14 = 0

        A_23 = dt*np.tan(phi)
        A_24 = dt*v / np.cos(phi)**2
        
        A = np.array([
            [1, 0, -dt*v*np.sin(theta), A_03, A_04],
            [0, 1,  dt*v*np.cos(theta), A_13, A_14],
            [0, 0,                   1, A_23, A_24],
            [0, 0,                   0,    1,    0],
            [0, 0,                   0,    0,    1]
        ])

        # B = A[:,3:] * dt
        B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ], dtype=float) * dt
        
        return A, B
    
    def constrain(self, x, u):
        x[2] %= 2*np.pi
        x[4] %= 2*np.pi
        return x, u
    
    @staticmethod
    def get_heading(X):
        return X[..., 2]
    
    