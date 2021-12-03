# cost.py

import numpy as np


class QuadraticCost(object):
    """Defines the quadratic cost of a state and control from some 
       reference trajectory.
    """
    
    TERM_PENALTY = 10.0 # terminal state penalty
    
    def __init__(self, xf, Q, R):
        self.xf = xf
        # Q: Should normalize here?
        self.Q = Q / np.sum(Q) * Q.shape[0]
        self.R = R / np.sum(R) * R.shape[0]
        # self.Q = Q
        # self.R = R
        
    def __call__(self, x, u, terminal=False):
        """Return the quadratic cost around the operating point assuming identity Q & R."""
        if not terminal:
            return (x - self.xf).T @ self.Q @ (x - self.xf) + u.T @ self.R @ u
        return self.TERM_PENALTY * (x - self.xf).T @ self.Q @ (x - self.xf) 
        
    def quadraticize(self, x, u, terminal=False):
        """Compute the jacobians and hessians around the current control and state."""
        
        n_x = x.shape[0]
        n_u = u.shape[0]
        
        # Q: multiply by dt here?
        L_x = (x - self.xf).T @ (self.Q + self.Q.T)
        L_u = u.T @ (self.R + self.R.T)
        L_xx = self.Q + self.Q.T
        L_uu = self.R + self.R.T
        L_ux = np.zeros((n_u, n_x))
        # L_xu = L_ux.T
        
        if terminal:
            L_x *= self.TERM_PENALTY
            L_xx *= self.TERM_PENALTY
            L_u = np.zeros((n_u))
            L_uu = np.zeros((n_u, n_u))
        
        return L_x, L_u, L_xx, L_uu, L_ux
    
 