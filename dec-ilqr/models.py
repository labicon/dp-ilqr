# models.py
# Module for various implementations of dynamical models.

import re

import numpy as np
import matplotlib.pyplot as plt

from dynamics import DynamicalModel, LinearModel, NumericalDiffModel


ACC_LIMS = [-1.0, 1.0] # [m/s**2]
V_LIMS = [-3.0, 3.0] # [m/s]
OMEGA_LIMS = [-np.pi/2, np.pi/2] # [rads/s]
STEER_LIMS = OMEGA_LIMS


class DoubleInt1dDynamics(LinearModel):
    """Canonical 2nd Order System used to demonstrate control principles.
        state := [position, velocity]
        control := [acceleration]
    """
    
    def __init__(self, dt):
        super(DoubleInt1dDynamics, self).__init__(2, 1, dt)
    
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
        # u = np.clip(u, *ACC_LIMS)
        return x, u
    
    def plot(self, X, xf=None, Jf=None, _=None):
        plt.clf()
        ax = plt.gca()
        t = np.arange(X.shape[0]) * self.dt
        
        ax.plot(t, X[:,0])
        ax.axhline(X[0,0], c='g', label='$x_0$: ' + str(X[0]))
        if xf is not None:
            ax.axhline(xf[0], c='r', label='$x_f$: ' + str(xf))

        title = 'Double Integrator 1D'
        if Jf is not None:
            title += f': $J_f$ = {Jf:.3g}'
            
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
    
    def __init__(self, dt):
        super(DoubleInt2dDynamics, self).__init__(4, 2, dt)
    
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
        # u = np.clip(u, *ACC_LIMS)
        return x, u
    
    def plot(self, X, xf=None, Jf=None, do_headings=False):
        # Augment the state with headings defined from the velocities.
        theta = np.arctan2(X[:,3], X[:,2])
        X = np.c_[X, theta]
        super().plot(X, xf, Jf=Jf, do_headings=do_headings)
        plt.gca().set_title('Double Integrator 2D')
    
    
class CarDynamics(DynamicalModel):
    """Simplified UnicycleModel for modeling a car.
        state := [x position, y position, heading angle]
        control := [linear velocity, angular velocity]
        
        NOTE: These dynamics don't integrate intuitively, f is most likely
        the culprit.
    """
    
    def __init__(self, dt):
        super(CarDynamics, self).__init__(3, 2, dt)
        
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
        # x[2] -= int(x[2]/np.pi)*2*np.pi
        # u[0] = np.clip(u[0], *V_LIMS)
        # u[1] = np.clip(u[1], *OMEGA_LIMS)
        return x, u
    
    
class CarDynamicsDiff(NumericalDiffModel):
    """Car Dynamics for finite difference."""
    
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
        
        super(CarDynamicsDiff, self).__init__(f, 3, 2, dt)
    
    
class UnicycleDynamics(DynamicalModel):
    """Simplified unicycle model for 2D trajectory planning.
        state := [x position, y position, linear velocity, heading angle]
        control := [linear acceleration, angular velocity]
    """
    
    def __init__(self, dt):
        super(UnicycleDynamics, self).__init__(4, 2, dt)
        
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
        # x[3] -= int(x[3]/np.pi)*2*np.pi
        # u[0] = np.clip(u[0], *ACC_LIMS)
        # u[1] = np.clip(u[1], *OMEGA_LIMS)
        return x, u
    
    def plot(self, *args, **kwargs):
        super().plot(*args, **kwargs)
        replace_title('Unicycle')
        
    
class UnicycleDynamicsDiff(NumericalDiffModel):
    """Unicycle with numerical difference."""
    
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
        
        super(UnicycleDynamicsDiff, self).__init__(f, 4, 2, dt)
    
    def constrain(self, x, u):
        u[0] = np.clip(u[0], *ACC_LIMS)
        u[1] = np.clip(u[1], *OMEGA_LIMS)
        return x, u
    
    def plot(self, *args, **kwargs):
        super().plot(*args, **kwargs)
        replace_title('Unicycle')
    
        
class BicycleDynamics(DynamicalModel):
    """Unicycle with steering component for front wheel.
        state := [x position, y position, heading, speed, steering angle]
        control := [acceleration, steering velocity]
    """
    
    def __init__(self, dt):
        super(BicycleDynamics, self).__init__(5, 2, dt)
        
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
        # x[2] -= int(x[2]/np.pi)*2*np.pi
        # x[4] -= int(x[4]/np.pi)*2*np.pi
        return x, u
        
    def plot(self, X, xf=None, Jf=None, do_headings=None):
        # Augment the state with in the last column.
        X = np.c_[X, X[:,2]]
        super(BicycleDynamics, self).plot(X, xf, Jf, do_headings=do_headings)
        replace_title('Bicycle')
        

class BicycleDynamicsDiff(NumericalDiffModel):
    """Bicycle Dynamics for numerical difference."""
    
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
        
        super(BicycleDynamicsDiff, self).__init__(f, 5, 2, dt)
    
    def constrain(self, x, u):
        u[0] = np.clip(u[0], *ACC_LIMS)
        u[1] = np.clip(u[1], *STEER_LIMS)
        return x, u
    
    def plot(self, X, xf=None, Jf=None, do_headings=None):
        # Augment the state with in the last column.
        X = np.c_[X, X[:,2]]
        super(BicycleDynamicsDiff, self).plot(X, xf, Jf, do_headings=do_headings)
        replace_title('Bicycle')
        
        
def replace_title(new_title):
    """Replace the first word of gca's title with something else."""
    
    gca = plt.gca()
    full_title = re.sub(r'^\w+', new_title, gca.get_title())
    gca.set_title(full_title)

    