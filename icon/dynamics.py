# dynamics.py

import abc

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import approx_fprime


class DynamicalModel(abc.ABC):
    """Representation of a discretized dynamical model to be applied as constraints 
       in the OCP.
    """
    
    def __init__(self, n_x, n_u, dt):
        self.n_x = n_x
        self.n_u = n_u
        self.dt = dt
    
    def __call__(self, x, u):
        """Advance the model in time by integrating the ODE, no assumption of linearity."""
        
        x, u = self.constrain(x, u)
        
        # Euler integration - works for linear models.
        # x_dot = self.f(x, self.dt, u)
        # return x + x_dot * self.dt

        # Integration of ODE using lsoda from FORTRAN library odepack.
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
    def constrain(self, x, u):
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
        # x, u = self.constrain(x, u)
        
        # Approximate linearization
        A, B = self.linearize(x, u)
        return A@x + B@u
    
    
class NumericalDiffModel(DynamicalModel):
    """Dynamical model where the linearizations are done via finite difference. Based off of
       https://github.com/anassinator/ilqr/blob/master/ilqr/dynamics.py
    """
    
    def __init__(self, f, *args, j_eps=None):

        self._f = f
        self.j_eps = j_eps if j_eps else np.sqrt(np.finfo(float).eps)

        super(NumericalDiffModel, self).__init__(*args)
        
    def __call__(self, x, u):
        return self.f(x, u)
    
    def f(self, x, u):
        return self._f(x, u)
    
    def constrain(self, x, u):
        pass
    
    def linearize(self, x, u):
        A = np.vstack([
            approx_fprime(x, lambda x: self.f(x, u)[i], self.j_eps) for i in range(self.n_x)
        ])
        
        B = np.vstack([
            approx_fprime(u, lambda u: self.f(x, u)[i], self.j_eps) for i in range(self.n_x)
        ])
        
        return A, B
