# cost.py
# Implements various cost structures for creating the optimization surface of the LQ Game.

import abc

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, NoNorm
import numpy as np

from util import Point


EPS = np.finfo(float).eps


class Cost(abc.ABC):
    """
    Abstract base class for cost objects.
    """
    
    @abc.abstractmethod
    def __call__(self, *args):
        """Returns the cost evaluated at the given state and control."""
        pass
    
    @abc.abstractmethod
    def quadraticize(self, *args):
        """Compute the jacobians and the hessians around the current control and state."""
        pass
    
    
class NumericalDiffCost(Cost):
    """
    Computes the quadraticization via finite difference. TODO.
    """
    
    def __init__(self, *args):
        pass
    
    def quadraticize(self, x, u, terminal=False):
        pass
    

class ObstacleCost(Cost):
    """
    The cost of an operating point from a stationary obstacle.
    """
    
    def __init__(self, position_inds, point, max_distance):
        self.ix, self.iy = position_inds
        self.point = point
        self.max_distance = max_distance
    
    def __call__(self, x, _, __=None):
        distance = np.linalg.norm([x[self.ix] - self.point.x, x[self.iy] - self.point.y])
        return min(0, distance - self.max_distance)**2
    
    def quadraticize(self, x, u, _=None):
        n_x = x.shape[0]
        n_u = u.shape[0]

        L_u = np.zeros((n_u))
        L_uu = np.zeros((n_u, n_u))
        L_ux = np.zeros((n_u, n_x))
        L_xx = np.zeros((n_x, n_x))
        
        dx = x[self.ix] - self.point.x
        dy = x[self.iy] - self.point.y
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance > self.max_distance:
            L_x = np.zeros((n_x))
            return L_x, L_u, L_xx, L_uu, L_ux
        
        L_x = 2 * (distance - self.max_distance) / distance \
                * np.pad(np.array([dx, dy]), (0, n_x-2))
        
        L_xx[self.ix,self.ix] = (
            2*self.max_distance*dx**2 / distance**3
          - 2*self.max_distance / distance 
          + 2
        )
        L_xx[self.ix,self.iy] = L_xx[self.iy,self.ix] = (
            2*self.max_distance*dx*dy / np.sqrt(
                self.point.x**2 - 2*self.point.x*x[self.ix] + self.point.y**2 
              - 2*self.point.y*x[self.iy] + x[self.ix]**2 + x[self.iy]**2
            ) ** 3)
        L_xx[self.iy,self.iy] = (
            2*self.max_distance*dy**2 / distance**3
          - 2*self.max_distance / distance 
          + 2
        )
        
        return L_x, L_u, L_xx, L_uu, L_ux
    
    def plot(self):
        circle = plt.Circle(
            (self.point.x, self.point.y), self.max_distance, 
            color='k', fill=False, alpha=0.75, ls='--', lw=2
        )
        plt.gca().add_artist(circle)
        
        
class ReferenceCost(Cost):
    """
    The cost of a state and control from some reference trajectory.
    """
    
    def __init__(self, xf, Q, R, Qf=None):
        self.xf = xf.astype(float)
        self.Q = Q.astype(float)
        self.R = R.astype(float)
        
        if Qf is None:
            Qf = np.eye(Q.shape[0])
        self.Qf = Qf.astype(float)
               
        self.Q_plus_QT = self.Q + self.Q.T
        self.R_plus_RT = self.R + self.R.T
        
    def __call__(self, x, u, terminal=False):
        if not terminal:
            return (x - self.xf).T @ self.Q @ (x - self.xf) + u.T @ self.R @ u
        return (x - self.xf).T @ self.Qf @ (x - self.xf)
        
    def quadraticize(self, x, u, terminal=False):
        n_x = x.shape[0]
        n_u = u.shape[0]
        
        L_x = (x - self.xf).T @ self.Q_plus_QT
        L_u = u.T @ self.R_plus_RT
        L_xx = self.Q_plus_QT
        L_uu = self.R_plus_RT
        L_ux = np.zeros((n_u, n_x))
        
        if terminal:
            L_x = (x - self.xf).T @ (self.Qf + self.Qf.T)
            L_xx = self.Qf + self.Qf.T
            L_u = np.zeros((n_u))
            L_uu = np.zeros((n_u, n_u))
        
        return L_x, L_u, L_xx, L_uu, L_ux
    
    def plot(self):
        ax = plt.gca()
        if "1D" in ax.get_title():
            ax.axhline(self.xf[0], c='r', label='$x_f$: ' + str(self.xf))
            return
        ax.scatter(self.xf[0], self.xf[1], 80, 'r', 'x', label='$x_{goal}$: ' + str(self.xf))
        

class AgentCost(Cost):
    """
    Collects multiple costs for an agent in the game.
    """
    
    def __init__(self, costs, weights):
        assert len(costs) == len(weights)
        
        self.costs = costs
        self.weights = weights
        
    def __call__(self, x, u, terminal=False):
        
        full_cost = 0.0
        for cost, weight in zip(self.costs, self.weights):
            full_cost += weight * cost(x, u, terminal)
        return full_cost
    
    def quadraticize(self, x, u, terminal=False):
        n_x = x.shape[0]
        n_u = u.shape[0]
        
        L_x = np.zeros((n_x))
        L_u = np.zeros((n_u))
        L_xx = np.zeros((n_x, n_x))
        L_uu = np.zeros((n_u, n_u))
        L_ux = np.zeros((n_u, n_x))
        
        for cost, weight in zip(self.costs, self.weights):
            L_xi, L_ui, L_xxi, L_uui, L_uxi = cost.quadraticize(x, u, terminal)
            L_x += weight * L_xi
            L_u += weight * L_ui
            L_xx += weight * L_xxi
            L_uu += weight * L_uui
            L_ux += weight * L_uxi
        
        return L_x, L_u, L_xx, L_uu, L_ux
    
    def plot(self, surface_plot=False, x0=None, axis=None, log_colors=False):
        """Call the children cost.plot functions and overlay a sampled cost surface
           over the current axes.
        """
        
        for cost in self.costs:
            cost.plot()
            
        if not surface_plot:
            return
        
        assert x0 is not None
        STEP = 0.1
        imshow_kwargs = {}

        if axis is None:
            axis = (0, 10, 0, 10)
        if log_colors:
            imshow_kwargs = {'norm': LogNorm()}

        pts = np.mgrid[axis[0]:axis[1]:STEP, axis[2]:axis[3]:STEP].T.reshape(-1,2)
        u = np.zeros(2)

        costs = []
        for pt in pts:
            xi = np.resize(pt, x0.shape[0])
            costs.append(self(xi, u, False))

        side_len = round(np.sqrt(pts.shape[0]))
        costs = np.flip(np.array(costs).reshape(side_len, side_len), axis=0)
        # Q: why does NoNorm have weird behavior here?
        cost_h = plt.imshow(costs, extent=axis, interpolation='bilinear', **imshow_kwargs)
        
        # plt.title('Cost Surface')
        # plt.xlabel('x [m]')
        # plt.ylabel('y [m]')
        plt.colorbar(cost_h, label='Cost')
        
    