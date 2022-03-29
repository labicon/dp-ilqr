#!/usr/bin/env python

import abc

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import approx_fprime
from scipy.linalg import block_diag


class DynamicalModel(abc.ABC):
    """
    Representation of a discretized dynamical model to be applied as constraints in the OCP.
    """
    
    name = "Dynamical Model"
    
    def __init__(self, n_x, n_u, dt):
        self.n_x = n_x
        self.n_u = n_u
        self.dt = dt
    
    def __call__(self, x, u):
        """Advance the model in time by integrating the ODE, no assumption of linearity."""
        
        # NOTE: Hold off on applying constraints until better understood.
        # x, u = self.constrain(x, u)
        
        # Euler integration - works for linear models.
        x_dot = self.f(x, self.dt, u)
        return x + x_dot*self.dt

        # Integration of ODE using lsoda from FORTRAN library odepack.
        args = tuple([u.flatten()]) # ensure u is passed off properly
        return odeint(self.f, x, (0, self.dt), args=args)[-1]
    
    @staticmethod
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
    
    @staticmethod
    def get_heading(X):
        """Retrieve the heading from a given state vector."""
        # Assume it's stored in the last column.
        return X[..., -1]
    
    def plot(self, X, do_headings=False, coupling_radius=None):
        """Visualizes a state evolution over time with some annotations."""
        
        assert X.shape[1] >= 3, 'Must at least have x and y states for this to make sense.'
        
        ax = plt.gca()
        N = X.shape[0]
        t = np.arange(N) * self.dt

        # NOTE: cmap=plt.get_cmap('Greys_r') provides more contrast but can get busy.
        ax.scatter(X[:,0], X[:,1], c=t)
        ax.scatter(X[0,0], X[0,1], 80, 'g', 'x', label='$x_0$')

        # Annotate the collision radius.
        if coupling_radius:
            ax.add_artist(plt.Circle(
                    (X[-1,0], X[-1,1]), coupling_radius, 
                    color='k', fill=True, alpha=0.3, lw=2
                ))
            # alphas = np.log10(np.logspace(0, 1, N+1))
            # for i, x in enumerate(X):
            #     ax.add_artist(plt.Circle(
            #         (x[0], x[1]), coupling_radius, 
            #         color='k', fill=True, alpha=0.02, lw=2
            #     ))
        
        if do_headings:
            bases = np.vstack([X[:-1,0], X[:-1,1]]).T
            dists = np.linalg.norm(np.diff(X[:,:2], axis=0), axis=1)
            theta = self.get_heading(X)[:-1]
            ends = bases + dists[:,np.newaxis]*np.vstack([
                np.cos(theta),
                np.sin(theta)
            ]).T

            for i in range(N-1):
                plt.annotate('', ends[i], bases[i], arrowprops=dict(
                    facecolor='black', headwidth=5, width=1, shrink=0))
                
                
class LinearModel(DynamicalModel):
    """
    Dynamical model where the system can be fully integrated via its discretized 
    linearizations, i.e. x_{k+1} = A x_k + B u_k.
    """
    
    def __call__(self, x, u):
        # x, u = self.constrain(x, u)
        
        # Approximate linearization
        A, B = self.linearize(x, u)
        return A@x + B@u
    
    
class NumericalDiffModel(DynamicalModel):
    """
    Dynamical model where the linearizations are done via finite difference. Based off of
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

    
class MultiDynamicalModel(DynamicalModel):
    
    """
    Encompasses the dynamical simulation and linearization for a collection of DynamicalModel's.
    """
    
    def __init__(self, submodels):
        self.dt = submodels[0].dt # assume they line up
        self.submodels = submodels
        self.n_players = len(submodels)
        
        self.x_dims = [submodel.n_x for submodel in submodels]
        self.u_dims = [submodel.n_u for submodel in submodels]
        self.n_x = sum(self.x_dims)
        self.n_u = sum(self.u_dims)
    
    def __call__(self, x, u):
        """Integrate the dynamics for the combined decoupled dynamical model."""
        
        x_split, u_split = self.partition(x, u)
        sub_states = [submodel(xi, ui) for submodel, xi, ui in zip(self.submodels, x_split, u_split)]
        return np.concatenate(sub_states, axis=0)
    
    def partition(self, x, u):
        """Helper to split up the states and control for each subsystem."""
        
        x_split = np.split(x, np.cumsum(self.x_dims[:-1]))
        u_split = np.split(u, np.cumsum(self.u_dims[:-1]))
        return x_split, u_split
    
    def linearize(self, x, u):
        """Compute the linearizations of each of the submodels and return as block diagonal
           A and B.
        """
        
        x_split, u_split = self.partition(x, u)
        sub_linearizations = [
            submodel.linearize(xi, ui) for submodel, xi, ui in zip(self.submodels, x_split, u_split)
        ]
        
        sub_As = [AB[0] for AB in sub_linearizations]
        sub_Bs = [AB[1] for AB in sub_linearizations]
        
        return block_diag(*sub_As), block_diag(*sub_Bs)
            
    def constrain(self, x, u):
        return x, u
    
    def plot(self, X, do_headings=False, coupling_radius=1.0):
        """Delegate plotting to subsystem models."""
        
        X_split = np.split(X, np.cumsum(self.x_dims[:-1]), axis=1)
        for X, submodel in zip(X_split, self.submodels):
            submodel.plot(X, do_headings=do_headings, coupling_radius=coupling_radius)
            
                             