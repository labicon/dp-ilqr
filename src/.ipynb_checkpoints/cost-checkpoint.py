# cost.py

import numpy as np


class QuadraticCost(object):
    """Defines the quadratic cost of a state and control from some 
       reference trajectory.
    """
    
    TERM_PENALTY = 10.0 # terminal state penalty
    
    def __init__(self, xf, Q, R):
        self.xf = xf
        self.Q = Q
        self.R = R
            
    def __call__(self, x, u, terminal=False):
        """Return the quadratic cost around the operating point assuming identity Q & R."""
        if not terminal:
            return 0.5 * ((x - self.xf).T @ self.Q @ (x - self.xf) + u.T @ self.R @ u)
        return self.TERM_PENALTY * 0.5 * (x - self.xf).T @ self.Q @ (x - self.xf) 
        
    def quadraticize(self, x, u, dt):
        """Compute the jacobians and hessians around the current control and state."""
        
        n_x = x.shape[0]
        n_u = u.shape[0]
        
        # Q: multiply by dt here?
        L_x = (x - self.xf).T @ (self.Q + self.Q.T) * dt
        L_u = u.T @ (self.R + self.R.T) * dt
        L_xx = (self.Q + self.Q.T) * dt
        L_uu = (self.R + self.R.T) * dt
        L_ux = np.zeros((n_u, n_x)) * dt
        # L_xu = L_ux.T
        
        return L_x, L_u, L_xx, L_uu, L_ux