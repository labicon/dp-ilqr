# cost.py

import numpy as np


class QuadraticCost(object):
    """Defines the quadratic cost of a state and control from some 
       reference trajectory.
    """
    
    def __init__(self, xf, Q, R, term_penalty=10.0):
        self.xf = xf.astype(float)
        # NOTE: don't normalize here to maintain relative cost of state and control.
        self.Q = Q.astype(float)
        self.R = R.astype(float)
        self.term_penalty = term_penalty
        
    def __call__(self, x, u, terminal=False):
        """Return the quadratic cost around the operating point assuming identity Q & R."""
        if not terminal:
            return (x - self.xf).T @ self.Q @ (x - self.xf) + u.T @ self.R @ u
        return self.term_penalty * (x - self.xf).T @ self.Q @ (x - self.xf)
        
    def quadraticize(self, x, u, terminal=False):
        """Compute the jacobians and hessians around the current control and state."""
        
        n_x = x.shape[0]
        n_u = u.shape[0]
        
        L_x = (x - self.xf).T @ (self.Q + self.Q.T)
        L_u = u.T @ (self.R + self.R.T)
        L_xx = self.Q + self.Q.T
        L_uu = self.R + self.R.T
        L_ux = np.zeros((n_u, n_x))
        # L_xu = L_ux.T
        
        if terminal:
            L_x *= self.term_penalty
            L_xx *= self.term_penalty
            L_u = np.zeros((n_u))
            L_uu = np.zeros((n_u, n_u))
        
        return L_x, L_u, L_xx, L_uu, L_ux
    
 