# dynamics.py

from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class DynamicalModel(ABC):
    """Representation of a discretized dynamical model to be applied as constraints 
       in the OCP.
    """
    
    def __init__(self, n_x, n_u, dt):
        self.n_x = n_x
        self.n_u = n_u
        self.dt = dt
    
    def __call__(self, x, u):
        """Advance the model in time assuming linear dynamics"""
        
        u = self.constrain(u)

        # Approximate linearization
        # A, B = self.linearize(x, u)
        # return A@x + B@u
        
        # Euler integration
        # x_dot = self.f(x, self.dt, u)
        # return x + x_dot * self.dt
        
        # Adams/BDF method with automatic stiffness detection and switching
        args = tuple([u.flatten()]) # ensure u is passed off properly
        return odeint(self.f, x, (0, self.dt), args=args)[-1]
    
    @staticmethod
    @abstractmethod
    def f(*args):
        """Continuous derivative of dynamics with respect to time."""
        pass
    
    @abstractmethod
    def linearize(self, *args, **kwargs):
        """Returns the discretized and linearized dynamics A and B."""
        pass
    
    @abstractmethod
    def constrain(self, u):
        """Apply physical constraints to the control input u."""
        pass
    
    @abstractmethod
    def plot(self, *args, **kwargs):
        """Visualizes a state evolution over time."""
        pass
        

class DoubleIntegratorDynamics(DynamicalModel):
    """Canonical 2nd Order System used to demonstrate control principles.
        state := [position, velocity]
        control := [acceleration]
    """
    
    F_LIMS = [-1.0, 1.0] # [N]
    
    def __init__(self, dt=1.0):
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
    
    def constrain(self, u):
        # u = np.clip(u, *self.F_LIMS)
        return u
    
    def plot(self, X, xf=None):
        plt.clf()
        ax = plt.gca()
        t = np.arange(X.shape[0]) * self.dt
        
        ax.plot(t, X[:,0])
        ax.axhline(X[0,0], c='g', label='$x_0$')
        if xf is not None:
            ax.axhline(xf[0], c='r', label='$x_f$')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position [m]')
        ax.set_title('Double Integator')
        ax.legend()
        ax.grid()
    
    
class CarDynamics(DynamicalModel):
    """Simplified UnicycleModel for modeling a car.
        state := [x position, y position, heading angle]
        control := [linear velocity, angular velocity]
    """
    
    V_LIMS = [-3.0, 3.0] # [m/s]
    OMEGA_LIMS = [-np.pi/2, np.pi/2] # [rads/s]
    
    def __init__(self, dt=1.0):
        super().__init__(3, 2, dt)
        
    @staticmethod
    def f(x, _, u):
        v = u[0]
        omega = u[1]
        return np.array([
            v*np.cos(omega),
            v*np.sin(omega),
            omega
        ])
    
    def linearize(self, x, _):
        
        yaw = x[2] % (2*np.pi)
        A = np.eye(3)
        B = np.array([
            [np.cos(yaw), 0],
            [np.sin(yaw), 0],
            [          0, 1]
        ]) * self.dt
        
        return A, B
    
    def constrain(self, u):
        # u[0] = np.clip(u[0], *self.V_LIMS)
        # u[1] = np.clip(u[1], *self.OMEGA_LIMS)
        return u
        
    def plot(self, X, xf=None):
        plt.clf()
        ax = plt.gca()
        N = X.shape[0]
        t = np.arange(N) * self.dt

        h_scat = ax.scatter(X[:,0], X[:,1], c=t)
        ax.scatter(X[0,0], X[0,1], 80, 'g', 'x', label='$x_0$')
        if xf is not None:
            ax.scatter(xf[0], xf[1], 80, 'r', 'x', label='$x_f$')

        # bases = np.vstack([X[:,0], X[:,1]]).T
        # ends = np.vstack([X[:,0] + np.cos(X[:,2]), X[:,1] + np.sin(X[:,2])]).T
        # for i in range(N):
        #     plt.annotate('', ends[i], bases[i], arrowprops=dict(
        #         facecolor='black', headwidth=5, width=1, shrink=0))

        plt.colorbar(h_scat, label='Time [s]')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Car')
        ax.legend()
        ax.grid()


class UnicycleDynamics(DynamicalModel):
    """Simplified unicycle model for 2D trajectory planning.
        state := [x position, y position, linear velocity, heading angle]
        control := [linear acceleration, angular velocity]
    """
    
    ACC_LIMS = [-1, 1] # [m/s**2]
    OMEGA_LIMS = [-np.pi, np.pi] # [rads/s]
    
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
        theta = x[3] % (2*np.pi)

        A = np.eye(4)
        A[:2,2:] = np.array([
            [np.cos(theta), -v*np.sin(theta)],
            [np.sin(theta),  v*np.cos(theta)]
        ]) * self.dt

        B = np.array([
            [-self.dt*v*np.sin(theta), self.dt*np.cos(theta)],
            [ self.dt*v*np.cos(theta), self.dt*np.sin(theta)],
            [                       0,                     1],
            [                       1,                     0]
        ]) * self.dt

        return A, B
    
    def constrain(self, u):
        # u[0] = np.clip(u[0], *self.ACC_LIMS)
        # u[1] = np.clip(u[1], *self.OMEGA_LIMS)
        return u
    
    def plot(self, X, xf=None):
        """TODO: consider merging with CarDynamics somehow."""
        
        plt.clf()
        ax = plt.gca()
        N = X.shape[0]
        t = np.arange(N) * self.dt

        h_scat = ax.scatter(X[:,0], X[:,1], c=t)
        ax.scatter(X[0,0], X[0,1], 80, 'g', 'x', label='$x_0$')
        if xf is not None:
            ax.scatter(xf[0], xf[1], 80, 'r', 'x', label='$x_f$')

        plt.colorbar(h_scat, label='Time [s]')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Unicycle')
        ax.legend()
        ax.grid()
    
    