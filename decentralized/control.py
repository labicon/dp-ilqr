#!/usr/bin/env python

"""Various implementations of LQR controllers including LQR and iLQR

[1] Jackson. AL iLQR Tutorial. https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf

"""


import abc

import matplotlib.pyplot as plt
import numpy as np

from .dynamics import DynamicalModel
from .cost import Cost


class BaseController(abc.ABC):
    """
    Abstract base class for Optimal Control Problem solvers.
    """
    
    def __init__(self, dynamics, cost, N):
        assert N > 1
        assert issubclass(dynamics.__class__, DynamicalModel)
        assert issubclass(cost.__class__, Cost)
        
        self.dynamics = dynamics
        self.cost = cost
        self.N = N
    
    @property
    def n_x(self):
        return self.dynamics.n_x
    
    @property
    def n_u(self):
        return self.dynamics.n_u
    
    @property
    def dt(self):
        return self.dynamics.dt
    
    @property
    def xf(self):
        return self.cost.xf
    
    @abc.abstractmethod
    def run(self, x0):
        """Implements functionality to solve the OCP at the current state x0."""
        pass
    
    def _rollout(self, x0, U):
        """Rollout the system from an initial state with a control sequence U."""
        
        N = U.shape[0]
        X = np.zeros((N+1, self.n_x))
        X[0] = x0
        J = 0.0
        
        for t in range(N):
            X[t+1] = self.dynamics(X[t], U[t])
            J += self.cost(X[t], U[t])
        J += self.cost(X[-1], np.zeros(self.n_u), terminal=True)
        
        return X, J
    
    def plot(self, 
             X, 
             title_suffix=None, 
             do_headings=False, 
             surface_plot=False, 
             coupling_radius=1.0, 
             **kwargs):
        """Sets up a trajectory plot and renders the sub-objects on gca, passing 
           additional keywords to AgentCost.plot().
        """

        plt.clf()
        ax = plt.gca()
        
        title = self.dynamics.name
        if title_suffix is not None:
            title += title_suffix
        
        ax.set_title(title)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        
        self.dynamics.plot(X, do_headings, coupling_radius)
        self.cost.plot(surface_plot, **kwargs)
        
        # Only include unique labels in the legend.
        handles, labels = plt.gca().get_legend_handles_labels()
        leg_map = dict(zip(labels, handles))
        ax.legend(leg_map.values(), leg_map.keys())

        
class iLQR(BaseController):
    """
    iLQR solver.
    """
    
    DELTA_0 = 2.0 # initial regularization scaling
    MU_MIN = 1e-6 # regularization floor, below which is 0.0
    MU_MAX = 1e3 # regularization ceiling
    N_LS_ITER = 10 # number of line search iterations
    
    def __init__(self, dynamics, cost, N=10):
        super().__init__(dynamics, cost, N)

        self.μ = 1.0 # regularization
        self.Δ = self.DELTA_0 # regularization scaling
    
    def _forward_pass(self, X, U, K, d, α):
        """Forward pass to rollout the control gains K and d."""
        
        X_next = np.zeros((self.N+1, self.n_x))
        U_next = np.zeros((self.N, self.n_u))

        X_next[0] = X[0].copy()
        J = 0.0

        for t in range(self.N):
            δx = X_next[t] - X[t]
            δu = K[t]@δx + α*d[t]

            U_next[t] = U[t] + δu
            X_next[t+1] = self.dynamics(X_next[t], U_next[t])
            
            J += self.cost(X_next[t], U_next[t])
        J += self.cost(X_next[-1], np.zeros((self.n_u)), terminal=True)

        return X_next, U_next, J
    
    def _backward_pass(self, X, U):
        """Backward pass to compute gain matrices K and d from the trajectory."""
        
        K = np.zeros((self.N, self.n_u, self.n_x)) # linear feedback gain
        d = np.zeros((self.N, self.n_u)) # feedforward gain

        # self.μ = 0.0 # DBG
        reg = self.μ * np.eye(self.n_x)
        
        L_x, _, L_xx, _, _ = self.cost.quadraticize(X[-1], np.zeros(self.n_u), terminal=True)
        p = L_x
        P = L_xx

        for t in range(self.N-1, -1, -1):
            L_x, L_u, L_xx, L_uu, L_ux = self.cost.quadraticize(X[t], U[t])
            A, B = self.dynamics.linearize(X[t], U[t])
            
            Q_x = L_x + A.T @ p
            Q_u = L_u + B.T @ p
            Q_xx = L_xx + A.T @ P @ A
            Q_uu = L_uu + B.T @ (P + reg) @ B
            Q_ux = L_ux + B.T @ (P + reg) @ A

            K[t] = -np.linalg.solve(Q_uu, Q_ux)
            d[t] = -np.linalg.solve(Q_uu, Q_u)
            
            p = Q_x + K[t].T @ Q_uu @ d[t] + K[t].T @ Q_u + Q_ux.T @ d[t]
            P = Q_xx + K[t].T @ Q_uu @ K[t] + K[t].T @ Q_ux + Q_ux.T @ K[t]
            P = 0.5 * (P + P.T)
            
        return K, d
        
    def run(self, x0, U=None, n_lqr_iter=50, tol=1e-3):
        """Solve the OCP."""
        
        if U is None:
            # U = np.zeros((self.N, self.n_u))
            # U = np.full((self.N, self.n_u), 0.1)
            U = 1e-3 * np.random.uniform(-1, 1, (self.N, self.n_u))
        if U.shape != (self.N, self.n_u):
            raise ValueError
            
        # Reset regularization terms.
        self.μ = 1.0
        self.Δ = self.DELTA_0
        
        is_converged = False
        alphas = 1.1**(-np.arange(self.N_LS_ITER)**2)
        
        X, J_star = self._rollout(x0, U)
        
        print(f'0/{n_lqr_iter}\tJ: {J_star:g}')        
        for i in range(n_lqr_iter):
            accept = False
            
            # Backward recurse to compute gain matrices.
            K, d = self._backward_pass(X, U)
            alphas_ls = alphas
            
            # Conduct a line search to find a satisfactory trajectory where we
            # continually decrease α. We're effectively getting closer to the 
            # linear approximation in the LQR case.
            for α in alphas_ls:
                
                X_next, U_next, J = self._forward_pass(X, U, K, d, α)
                
                if J < J_star:
                    if np.abs((J_star - J) / J_star) < tol:
                        is_converged = True
                        
                    X = X_next
                    U = U_next
                    J_star = J
                    
                    # Decrease regularization to converge more slowly.
                    self.Δ = min(1.0, self.Δ) / self.DELTA_0
                    self.μ *= self.Δ
                    if self.μ <= self.MU_MIN:
                        self.μ = 0.0
                    
                    accept = True
                    break

            if not accept:
                
                # DBG: Regularization is pointless for quadratic costs.
                # print('[run] Failed line search.. giving up.')
                # break

                # Increase regularization if we're not converging.
                print('Failed line search.. increasing μ.')
                self.Δ = max(1.0, self.Δ) * self.DELTA_0
                self.μ = max(self.MU_MIN, self.μ*self.Δ)
                if self.μ >= self.MU_MAX:
                    print("Exceeded max regularization term...")
                    break

            if is_converged:
                break
            
            print(f'{i+1}/{n_lqr_iter}\tJ: {J_star:g}\tμ: {self.μ:g}\tΔ: {self.Δ:g}')

        return X, U, J
    
        
class LQR(BaseController):
    """
    Linear Quadratic Regulator that assumes linear dynamics and quadratic costs.
    """
    
    @property
    def Q(self):
        return self.cost.Q
    
    @property
    def R(self):
        return self.cost.R
        
    def _backward_pass(self, X, U):
        """Compute the optimal feedforward gain K."""
        
        K = np.zeros((self.N, self.n_u, self.n_x))
        P = np.zeros((self.N, self.n_x, self.n_x))
        P = self.Q.copy()
        
        for t in range(self.N-1, -1, -1):
            A, B = self.dynamics.linearize(X[t], U[t])
            # Feedforward gain [1] (30)
            K[t] = np.linalg.inv(self.R + B.T @ P @ B) @ B.T @ P @ A
            # Cost-to-go [1] (32)
            P = self.Q + (A.T @ P @ A) - A.T @ P.T @ B @ K[t]
        return K
    
    def _forward_pass(self, X, U, K):
        """Apply the feedforward gain to compute the next trajectory."""
        
        X_next = np.zeros((self.N+1, self.n_x))
        U_next = np.zeros((self.N, self.n_u))
        
        X_next[0] = X[0]
        J = 0.0

        for t in range(self.N):
            U_next[t] = -K[t] @ (X_next[t] - self.xf)
            X_next[t+1] = self.dynamics(X_next[t], U_next[t])
            J += self.cost(X_next[t], U_next[t])
        J += self.cost(X_next[-1], np.zeros((self.n_u)), terminal=True)

        return X_next, U_next, J
    
    def run(self, x0, U=None, n_iter=10):
        """Solve the LQR OCP."""
        
        if U is None:
            U = np.random.randn(self.N, self.n_u) * 1e-3
        if U.shape != (self.N, self.n_u):
            raise ValueError
            
        X, J0 = self._rollout(x0, U)

        K = self._backward_pass(X, U)
        X, U, Jf = self._forward_pass(X, U, K)
        print(f'J0: {J0:.3g}\tJf: {Jf:.3g}')

        return X, U, Jf

    
# Adaped from: 
#   https://github.com/anassinator/ilqr/blob/master/ilqr/controller.py
class RecedingHorizonController:
    """Receding horizon controller
    
    Attributes
    ----------
    _x : np.ndarray
        Current state
    _controller : BaseController
        Controller instance initialized with all necessary costs
    step_size : int, default=1
        Number of steps to take between controller fits
    n_runs : int
        Number of ``controller.run()`` executions
        
    """

    def __init__(self, x0, controller, step_size=1):
        self._x = x0
        self._controller = controller
        self.step_size = step_size

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, xn):
        self._x = xn
        
    @property
    def N(self):
        return self._controller.N
    
    def run(self, U_init, J_converge=1.0, **kwargs):
        """Optimize the system controls from the current state
        
        Parameters
        ----------
        U_init : np.ndarray
            Initial inputs to provide to the controller
        J_converge : float
            Cost defining convergence to the goal, which causes us to stop if 
            reached
        **kwargs
            Additional keyword arguments to pass onto the ``controller.run``.
            
        Returns
        -------
        X : np.ndarray
            Resulting trajectory computed by controller of shape (step_size, n_x)
        U : np.ndarray
            Control sequence applied of shape (step_size, n_u)
        J : float
            Converged cost value
        
        """

        U = U_init
        while True:
            # Fit the current state initializing with our control sequence.
            if U.shape != (self._controller.N, self._controller.n_u):
                raise RuntimeError
                
            X, U, J = self._controller.run(self.x, U, **kwargs)
            
            # Add random noise to the trajectory to add some realism.
            # X += np.random.normal(size=X.shape, scale=0.05)
            
            # Shift the state to our predicted value. NOTE: this can be 
            # updated externally for actual sensor feedback.
            self.x = X[self.step_size]

            yield X[:self.step_size], U[:self.step_size], J
            
            U = U[self.step_size:]
            U = np.vstack([U, np.zeros((self.step_size, self._controller.n_u))])
            
            if J < J_converge:
                print("Converged!")
                break
            
            
class NavigationProblem:
    
    """Splits a centralized optimal control problem into a decentralized one
    
    Attributes
    ----------
    planning_radius : float, default=10.0
        The splitting distance between agents. If two agents come within each 
        others' planning radii, we split the centralized problem into distinct
        sub-problems.
        
    """
    
    def __init__(self, planning_radius=10.0):
        self.planning_radius = planning_radius
        
    def delegate_subproblems(self, multi_dynamics, game_cost):
        raise NotImplementedError
        
        