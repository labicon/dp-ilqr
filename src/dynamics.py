# dynamics.py

import abc

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class DynamicalModel(abc.ABC):
    """Representation of a discretized dynamical model to be applied as constraints 
       in the OCP.
    """
    
    ACC_LIMS = [-1.0, 1.0] # [m/s**2]
    V_LIMS = [-3.0, 3.0] # [m/s]
    OMEGA_LIMS = [-np.pi/2, np.pi/2] # [rads/s]
    
    def __init__(self, n_x, n_u, dt):
        self.n_x = n_x
        self.n_u = n_u
        self.dt = dt
    
    def __call__(self, x, u):
        """Advance the model in time by integrating the ODE, no assumption of linearity."""
        
        # Euler integration - also works for linear models
        # x_dot = self.f(x, self.dt, u)
        # return x + x_dot * self.dt

        # Adams/BDF method with automatic stiffness detection and switching
        args = tuple([u.flatten()]) # ensure u is passed off properly
        return odeint(self.f, x, (0, self.dt), args=args)[-1]
    
    @staticmethod
    @abc.abstractmethod
    def f(*args):
        """Continuous derivative of dynamics with respect to time."""
        pass
    
    @abc.abstractmethod
    def linearize(self, *args, **kwargs):
        """Returns the discretized and linearized dynamics A and B."""
        pass
    
    @abc.abstractmethod
    def constrain(self, u):
        """Apply physical constraints to the control input u."""
        pass
    
    def plot(self, X, xf=None, do_headings=False):
        """Visualizes a state evolution over time."""
        assert X.shape[1] >= 3, 'Must at least have x and y states for this to make sense.'
        
        if xf is None:
            xf = X[-1]

        plt.clf()
        ax = plt.gca()
        N = X.shape[0]
        t = np.arange(N) * self.dt

        h_scat = ax.scatter(X[:,0], X[:,1], c=t)
        ax.scatter(X[0,0], X[0,1], 80, 'g', 'x', label='$x_0$: ' + str(X[0]))
        ax.scatter(xf[0], xf[1], 80, 'r', 'x', label='$x_f$: ' + str(xf))

        plt.colorbar(h_scat, label='Time [s]')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Car')
        ax.legend()
        ax.grid()
        
        if do_headings:
            N = X.shape[0]
            bases = np.vstack([X[:-1,0], X[:-1,1]]).T
            dists = np.linalg.norm(np.diff(X[:,:2], axis=0), axis=1)
            ends = bases + dists[:,np.newaxis]*np.vstack([
                np.cos(X[:-1,-1]), 
                np.sin(X[:-1,-1])
            ]).T

            for i in range(N-1):
                plt.annotate('', ends[i], bases[i], arrowprops=dict(
                    facecolor='black', headwidth=5, width=1, shrink=0))
                
                
class LinearModel(DynamicalModel):
    """Dynamical model where the system can be fully integrated via its 
       discretized linearizations, i.e. x_{k+1} = A x_k + B u_k.
    """
    
    def __call__(self, x, u):
        # u = self.constrain(u)

        # Approximate linearization
        A, B = self.linearize(x, u)
        return A@x + B@u
    

class DoubleInt1dDynamics(LinearModel):
    """Canonical 2nd Order System used to demonstrate control principles.
        state := [position, velocity]
        control := [acceleration]
    """
    
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
        return np.clip(u, *self.ACC_LIMS)
    
    def plot(self, X, xf=None, _=None):
        plt.clf()
        ax = plt.gca()
        t = np.arange(X.shape[0]) * self.dt
        
        ax.plot(t, X[:,0])
        ax.axhline(X[0,0], c='g', label='$x_0$: ' + str(X[0]))
        if xf is not None:
            ax.axhline(xf[0], c='r', label='$x_f$: ' + str(xf))

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position [m]')
        ax.set_title('Double Integator 1D')
        ax.legend()
        ax.grid()
        

class DoubleInt2dDynamics(LinearModel):
    """Double integrator dynamics in two dimensions.
        state := [x position, y position, x velocity, y velocity]
        control := [x acceleration, y acceleration]
    """
    
    def __init__(self, dt=1.0):
        super().__init__(4, 2, dt)
    
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
        
        A = np.array([
            [1, 0, self.dt,       0],
            [0, 1,       0, self.dt],
            [0, 0,       1,       0],
            [0, 0,       0,       1]
        ])

        B = np.array([
            [self.dt/2,         0],
            [        0, self.dt/2],
            [        1,         0],
            [        0,         1]
        ]) * self.dt

        return A, B
    
    def constrain(self, u):
        return np.clip(u, *self.ACC_LIMS)
    
    def plot(self, X, xf=None, do_headings=False):
        # Augment the state with headings defined from the velocities.
        theta = np.arctan2(X[:,3], X[:,2])
        X = np.c_[X, theta]
        super().plot(X, xf, do_headings=do_headings)
        plt.gca().set_title('Double Integrator 2D')
    
    
class CarDynamics(DynamicalModel):
    """Simplified UnicycleModel for modeling a car.
        state := [x position, y position, heading angle]
        control := [linear velocity, angular velocity]
        
        NOTE: These dynamics don't integrate intuitively, f is most likely
        the culprit.
    """
    
    def __init__(self, dt=1.0):
        super().__init__(3, 2, dt)
        
    @staticmethod
    def f(x, _, u):
        v = u[0]
        omega = u[1]
        theta = x[2] % (2*np.pi)
        
        return np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            omega
        ])
    
    def linearize(self, x, u):
        
        v = u[0]
        theta = x[2] % (2*np.pi)
        
        A = np.eye(3)
        A[:2,-1] = np.array([
            -np.sin(theta),
             np.cos(theta)
        ]) * v * self.dt
        B = np.array([
            [np.cos(theta), -v*self.dt*np.sin(theta)],
            [np.sin(theta),  v*self.dt*np.cos(theta)],
            [            0,                        1]
        ]) * self.dt
        
        return A, B
    
    def constrain(self, u):
        u[0] = np.clip(u[0], *self.V_LIMS)
        u[1] = np.clip(u[1], *self.OMEGA_LIMS)
        return u
    
    
class UnicycleDynamics(DynamicalModel):
    """Simplified unicycle model for 2D trajectory planning.
        state := [x position, y position, linear velocity, heading angle]
        control := [linear acceleration, angular velocity]
    """
    
    def __init__(self, dt):
        super().__init__(4, 2, dt)
        
    @staticmethod
    def f(x, _, u):
        v = x[2]
        theta = x[3] % (2*np.pi)
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

        A = np.array([
            [1, 0, self.dt*np.cos(theta), -self.dt*v*np.sin(theta)],
            [0, 1, self.dt*np.sin(theta),  self.dt*v*np.cos(theta)],
            [0, 0,                     1,                        0],
            [0, 0,                     0,                        1]
        ])

        # NOTE: dynamics in risk_sensitive disagree with analytical derivation, 
        # so use theirs instead.
        # B = np.array([
        #     [self.dt*np.cos(theta), -v*self.dt*np.sin(theta)],
        #     [self.dt*np.sin(theta),  v*self.dt*np.cos(theta)],
        #     [                     1,                       0],
        #     [                     0,                       1]
        # ]) * self.dt
        B = np.array([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ], dtype=float) * self.dt
        
        return A, B
    
    def constrain(self, u):
        u[0] = np.clip(u[0], *self.ACC_LIMS)
        u[1] = np.clip(u[1], *self.OMEGA_LIMS)
        return u
    
    def plot(self, *args, **kwargs):
        super().plot(*args, **kwargs)
        plt.gca().set_title('Unicycle')

        
class BicycleDynamics(DynamicalModel):
    """Unicycle with steering component for front wheel.
        state := [x position, y position, heading, speed, steering angle]
        control := [acceleration, steering velocity]
    """
    
    def __init__(self, dt=1.0):
        super().__init__(5, 2, dt)
        
    def __call__(self, x, u):
        # Forward Euler Method, since odeint isn't able to consistently 
        # integrate the dynamics. This was implemented explicitly to be consistent
        # with ilqr_driving.
        # u = self.constrain(u)
        return x + self.f(x, None, u)*self.dt
    
    @staticmethod
    def f(x, _, u):
        theta = x[2] % (2*np.pi)
        v = x[3]
        phi = x[4] % (2*np.pi)
        
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

        # Same thing here in using experimentally derived jacobian.
        # B = A[:,3:] * dt
        B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ], dtype=float) * dt
        
        return A, B
    
    def constrain(self, u):
        pass
        
    def plot(self, X, xf=None, do_headings=None):
        # Augment the state with in the last column.
        X = np.c_[X, X[:,2]]
        super().plot(X, xf, do_headings=do_headings)
        plt.gca().set_title('Bicycle')
        
