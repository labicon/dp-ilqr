# cost.py
# Implements various cost structures for creating the optimization surface of the LQ Game.

import abc

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, NoNorm
import numpy as np

from util import Point


EPS = np.finfo(float).eps

# Indicies corresponding to the positional outputs of quadraticize_distance
IX = 0
IY = 1


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
        raise NotImplementedError
    
    def quadraticize(self, x, u, terminal=False):
        raise NotImplementedError

        
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
        return min(0.0, distance - self.max_distance)**2
    
    def state_point(self, x):
        """Reformat the position within a state into a Point object."""
        return Point(x[self.ix], x[self.iy])
    
    def quadraticize(self, x, u, _=None):
        n_x = x.shape[0]
        n_u = u.shape[0]
        
        L_x = np.zeros((n_x))
        L_u = np.zeros((n_u))
        L_xx = np.zeros((n_x, n_x))
        L_uu = np.zeros((n_u, n_u))
        L_ux = np.zeros((n_u, n_x))
        
        # Compute jacobians and hessians as 2x1 and 2x2 as a function of the positions,
        # then reformat by adding zeros to match the state dimensions.
        L_x_obs, L_xx_obs = \
            quadraticize_distance(self.state_point(x), self.point, self.max_distance)
        
        L_x[self.ix] = L_x_obs[IX]
        L_x[self.iy] = L_x_obs[IY]
        L_xx[self.ix,self.ix] = L_xx_obs[IX,IX]
        L_xx[self.iy,self.iy] = L_xx_obs[IY,IY]
        L_xx[self.ix,self.iy] = L_xx[self.iy,self.ix] = L_xx_obs[IX,IY]
        
        return L_x, L_u, L_xx, L_uu, L_ux
    
    def plot(self):
        circle = plt.Circle(
            (self.point.x, self.point.y), self.max_distance, 
            color='k', fill=False, alpha=0.75, ls='--', lw=2
        )
        plt.gca().add_artist(circle)
        
        
class CouplingCost(Cost):
    
    """
    Models the couplings between different agents in the MultiDynamicalModel sense, i.e.
    how should we penalize two agents in the aggregate state for their relative distance?
    
    NOTE: This logic assumes that interactions between agents are symmetric, such that we
    can add jacobians & hessians equally in both directions.
    """
    
    def __init__(self, pos_inds, max_distance):
        self.pos_inds = pos_inds
        self.max_distance = max_distance
        self.n_agents = len(pos_inds)
    
    def __call__(self, x):
        total_cost = 0.0
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                ix, iy = self.pos_inds[i]
                jx, jy = self.pos_inds[j]
                
                distance = (x[ix] - x[jx])**2 + (x[iy] - x[jy])**2
                total_cost += min(0.0, distance - self.max_distance)**2
        return total_cost
    
    def quadraticize(self, x):
        n_x = x.shape[0]
        L_x = np.zeros((n_x))
        L_xx = np.zeros((n_x, n_x))
        
        for i in range(self.n_agents):            
            for j in range(i+1, self.n_agents):
                L_xi = np.zeros((n_x))
                L_xxi = np.zeros((n_x, n_x))
                
                ix, iy = self.pos_inds[i]
                jx, jy = self.pos_inds[j]
                L_x_pair, L_xx_pair = quadraticize_distance(
                    Point(x[ix], x[iy]), 
                    Point(x[jx], x[jy]), 
                    self.max_distance
                )
                
                L_xi[ix] = L_xi[jx] = L_x_pair[IX]
                L_xi[iy] = L_xi[jy] = L_x_pair[IY]
                L_xxi[ix,ix] = L_xxi[jx,jx] = L_xx_pair[IX,IX]
                L_xxi[iy,iy] = L_xxi[jy,jy] = L_xx_pair[IY,IY]
                L_xxi[ix,iy] = L_xxi[iy,ix] = L_xxi[jx,jy] = L_xxi[jy,jx] = L_xx_pair[IX,IY]
                
                L_x += L_xi
                L_xx += L_xxi
                
        return L_x, L_xx
                

class AgentCost(Cost):
    """
    Collects a reference cost and potentially obstacle costs for one agent in the game.
    
    TODO: Formalize organization of sub-class costs so we know where Q, R, and xf are.
    """
    
    def __init__(self, costs, weights):
        assert len(costs) == len(weights)
        assert all(isinstance(cost, ReferenceCost) or isinstance(cost, ObstacleCost)
                   for cost in costs)
        
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

        print(axis)
        if axis is None:
            ax = plt.gca()
            axis = ax.get_xlim() + ax.get_ylim()
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
        

class GameCost(Cost):
    """
    Collects the costs for multiple agents as well as coupling costs between the agents.
    """
    
    def __init__(self, agent_costs, coupling_cost):
        # NOTE: we assume that agent_costs are passed in the same order as their respective 
        # dynamics in the MultiDynamicalModel
        
        self.agent_costs = agent_costs
        self.coupling_cost = coupling_cost
        
    def __call__(self, x, u, terminal=False):
        agent_total = sum(ac(x, u, terminal) for ac in self.agent_costs)
        coupling_total = self.coupling_cost(x)
        return agent_total + coupling_total
    
    def quadraticize(self, x, u, terminal=False):
        L_xs, L_us = [], []
        L_xxs, L_uus, L_uxs = [], [], []
        
        # Run agent quadraticization then incorporate coupling quadraticizations.
        for agent_cost in self.agent_costs:
            # ...but what about the dimensionality of x and u...
            L_xi, L_ui, L_xxi, L_uui, L_uxi = agent_cost.quadraticize(x, u, terminal)
            L_xs.append(L_xi)
            L_us.append(L_ui)
            L_xxs.append(L_xxi)
            L_uus.append(L_uui)
            L_uxs.append(L_uxi)
        
        L_x = block_diag(*L_xs)
        L_u = block_diag(*L_us)
        L_xx = block_diag(*L_xxs)
        L_uu = block_diag(*L_uus)
        L_ux = block_diag(*L_uxs)
        
        L_x_coup, L_xx_coup = self.coupling_cost.quadraticize(x)
        L_x += L_x_coup
        L_xx += L_xx_coup
        
        return L_x, L_u, L_xx, L_uu, L_ux
    
def quadraticize_distance(point_a, point_b, max_distance):
    """Quadraticize the distance between two points thresholded by a max distance,
       returning the corresopnding 2x1 jacobian and 2x2 hessian.
       
       NOTE: we assume that the distances are organized in matrix form as [x, y]
       rather than [y, x].
       NOTE: in the ObstacleCost.quadraticize, point_a is for the state and point_b
       is for the obstacle.
    """
    
    POS_DIM = 2
    L_x = np.zeros((POS_DIM))
    L_xx = np.zeros((POS_DIM, POS_DIM))
    
    dx = point_a.x - point_b.x
    dy = point_a.y - point_b.y
    distance = np.sqrt(dx**2 + dy**2)

    if distance > max_distance:
        return L_x, L_xx
    
    L_x = 2 * (distance - max_distance) / distance * np.array([dx, dy])

    L_xx[IX,IX] = (
        2*max_distance*dx**2 / distance**3
      - 2*max_distance / distance 
      + 2
    )
    L_xx[IX,IY] = L_xx[IY,IX] = \
        2*max_distance*dx*dy / np.sqrt(
            point_b.x**2 - 2*point_b.x*point_a.x + point_b.y**2 
          + point_a.x**2 - 2*point_b.y*point_a.y + point_a.y**2
        ) ** 3
    L_xx[IY,IY] = (
        2*max_distance*dy**2 / distance**3
      - 2*max_distance / distance 
      + 2
    )

    return L_x, L_xx