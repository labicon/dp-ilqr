# models.py
# Module for various implementations of dynamical models.

import re

import numpy as np
import matplotlib.pyplot as plt

from .dynamics import DynamicalModel, LinearModel, NumericalDiffModel


ACC_LIMS = [-1.0, 1.0] # [m/s**2]
V_LIMS = [-3.0, 3.0] # [m/s]
OMEGA_LIMS = [-np.pi/2, np.pi/2] # [rads/s]
STEER_LIMS = OMEGA_LIMS


class DoubleInt1dDynamics(LinearModel):
    """Canonical 2nd Order System used to demonstrate control principles.
        state := [position, velocity]
        control := [acceleration]
    """
    
    name = "Double Integrator 1D"
    
    def __init__(self, dt):
        super().__init__(2, 1, dt)
    
    @staticmethod
    def f(x, dt, u):
        v = x[1]
        a = u[0]
        return np.array([
            v + dt*a,
            a
        ])
    
    def linearize(self, _, __):
        
        A = np.array([
            [1, self.dt],
            [0,       1]
        ])

        B = np.array([
            [self.dt**2/2],
            [     self.dt]
        ])

        return A, B
    
    def constrain(self, x, u):
        u = np.clip(u, *ACC_LIMS)
        return x, u
    
    def plot(self, X, Jf=None, _=None):
        ax = plt.gca()
        t = np.arange(X.shape[0]) * self.dt
        
        ax.plot(t, X[:,0])
        ax.axhline(X[0,0], c='g', label='$x_0$: ' + str(X[0]))

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position [m]')
        

class DoubleInt2dDynamics(LinearModel):
    """Double integrator dynamics in two dimensions.
        state := [x position, y position, x velocity, y velocity]
        control := [x acceleration, y acceleration]
    """
    
    name = "Double Integrator 2D"
    
    def __init__(self, dt):
        super().__init__(4, 2, dt)
    
        # LTI system
        self.A = np.array([
            [1, 0, self.dt,       0],
            [0, 1,       0, self.dt],
            [0, 0,       1,       0],
            [0, 0,       0,       1]
        ])

        self.B = np.array([
            [self.dt/2,         0],
            [        0, self.dt/2],
            [        1,         0],
            [        0,         1]
        ]) * self.dt
    
    @staticmethod
    def f(x, dt, u):
        vx = x[2]
        vy = x[3]
        ax = u[0]
        ay = u[1]
        
        return np.array([
            vx + dt*ax,
            vy + dt*ay,
            ax,
            ay
        ])
    
    def linearize(self, _, __):
        return self.A, self.B
    
    def constrain(self, x, u):
        u = np.clip(u, *ACC_LIMS)
        return x, u
    
    @staticmethod
    def get_heading(X):
        return np.arctan2(X[...,3], X[...,2])
    
    
class CarDynamics(NumericalDiffModel):
    """Car Dynamics for finite difference."""
    
    name = "Car"
    
    def __init__(self, dt):
        
        def f(x, u):
            x_ = x[..., 0]
            y = x[..., 1]
            theta = x[..., 2]
            v = u[..., 0]
            omega = u[..., 1]
            
            theta_next = theta # + omega*dt
            x_dot = v*np.cos(theta_next)
            y_dot = v*np.sin(theta_next)
            
            return np.stack([
                x_ + x_dot*dt,
                y + y_dot*dt,
                theta + omega*dt # theta_next
            ]).T
        
        super().__init__(f, 3, 2, dt)
        
    def constrain(self, x, u):
        x[2] %= 2*np.pi
        u[0] = np.clip(u[0], *V_LIMS)
        u[1] = np.clip(u[1], *OMEGA_LIMS)
        return x, u
        
    
class UnicycleDynamics(NumericalDiffModel):
    """Unicycle with numerical difference."""
    
    name = "Unicycle"
    
    def __init__(self, dt):
        
        def f(x, u):
            """Discretized dynamics."""
            
            mm = np
            
            x_ = x[..., 0]
            y = x[..., 1]
            v = x[..., 2]
            theta = x[..., 3]
            a = u[..., 0]
            omega = u[..., 1]

            next_theta = theta + omega*dt
            # x_dot = (v + 0.5*a*dt)*mm.cos(next_theta)
            # y_dot = (v + 0.5*a*dt)*mm.sin(next_theta)
            x_dot = v*mm.cos(next_theta)
            y_dot = v*mm.sin(next_theta)
            
            return mm.stack([
                x_ + x_dot*dt,
                y + y_dot*dt,
                v + a*dt,
                next_theta
            ]).T
        
        super().__init__(f, 4, 2, dt)
    
    def constrain(self, x, u):
        u[0] = np.clip(u[0], *ACC_LIMS)
        u[1] = np.clip(u[1], *OMEGA_LIMS)
        return x, u
        

class BicycleDynamics(NumericalDiffModel):
    """Bicycle Dynamics for numerical difference."""
    
    name = "Bicycle"
    
    def __init__(self, dt):
    
        def f(x, u):
            
            mm = np
            
            x_ = x[..., 0]
            y = x[..., 1]
            theta = x[..., 2]
            v = x[..., 3]
            phi = x[..., 4]
            a = u[..., 0]
            phi_dot = u[..., 1]            
            
            theta_dot = mm.tan(phi)
            next_theta = theta + theta_dot*dt

            x_dot = v*mm.cos(next_theta)
            y_dot = v*mm.sin(next_theta)
            
            return mm.stack([
                x_ + x_dot*dt,
                y + y_dot*dt,
                theta + theta_dot*dt,
                v + a*dt,
                phi + phi_dot*dt
            ]).T
        
        super().__init__(f, 5, 2, dt)
    
    def constrain(self, x, u):
        u[0] = np.clip(u[0], *ACC_LIMS)
        u[1] = np.clip(u[1], *STEER_LIMS)
        return x, u
    
    @staticmethod
    def get_heading(X):
        return X[..., 2]
        
