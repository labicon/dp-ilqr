# cost.py
# Implements various cost structures for creating the optimization surface of the LQ Game.
# With inspiration from 

import abc

import numpy as np

from util import Point


class Cost(abc.ABC):
    """Encapulates various cost objects used in navigation contexts.
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
    """Computes the quadraticization via finite difference."""
    
    def __init__(self, *args):
        pass
    
    def quadraticize(self, x, u, terminal=False):
        pass
    

class AgentCost(object):
    """Collects the cost 
    

class ObstacleCost(Cost):
    """The cost of an operating point from a stationary obstacle.
    """
    
    def __init__(self, position_inds, point, max_distance):
        self.x_ind, self.y_ind = position_inds
        self.point = point
        self.max_distance = max_distance
    
    def __call__(self, x):
        distance = np.linalg.norm([x[self.x_ind] - self.point.x, x[self.y_ind] - self.point.y])
        return min(distance, self.max_distance)
    
    def quadraticize(self, x, u, _=None):
        n_x = x.shape[0]
        n_u = u.shape[0]
        
        state_mask = np.zeros((n_x))
        state_mask[[self.x_ind, self.y_ind]] = -1
        
        x_o = np.zeros((n_x))
        x_o[self.x_ind] = self.point.x
        x_o[self.y_ind] = self.point.y
        
        L_x = 2*(x*state_mask - x_o).T
        L_u = np.zeros((n_u))
        L_xx = 2*np.diag(state_mask)
        L_uu = np.zeros((n_u, n_u))
        L_ux = np.zeros((n_u, n_x))
        
        return L_x, L_u, L_xx, L_uu, L_ux
        
        
class ReferenceCost(Cost):
    """The cost of a state and control from some reference trajectory.
    """
    
    def __init__(self, xf, Q, R, Qf=None):
        self.xf = xf.astype(float)
        self.Q = Q.astype(float)
        self.R = R.astype(float)
        
        if Qf is None:
            Qf = np.eye(Q.shape)
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
        # L_xu = L_ux.T
        
        if terminal:
            L_x = (x - self.xf).T @ (self.Qf + self.Qf.T)
            L_xx = self.Qf + self.Qf.T
            L_u = np.zeros((n_u))
            L_uu = np.zeros((n_u, n_u))
        
        return L_x, L_u, L_xx, L_uu, L_ux
    
 