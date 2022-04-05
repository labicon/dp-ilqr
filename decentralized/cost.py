#!/usr/bin/env python

"""Implements various cost structures in the LQ Game"""

import abc

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, NoNorm
import numpy as np
from scipy.optimize import approx_fprime
from scipy.linalg import block_diag

from .util import Point


EPS = np.sqrt(np.finfo(float).eps)

# Indicies corresponding to the positional outputs of _quadraticize_distance
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
    
    def quadraticize(self, x, u, _=None):
        """Compute the jacobians and the hessians around the current control and state."""
        
        nx = x.shape[0]
        nu = u.shape[0]
        
        L_x = np.zeros((nx))
        L_u = np.zeros((nu))
        L_xx = np.zeros((nx, nx))
        L_uu = np.zeros((nu, nu))
        L_ux = np.zeros((nu, nx))
        
        return L_x, L_u, L_xx, L_uu, L_ux
    
    def plot(self, *args, **kwargs):
        """Visualize this cost object on plt.gca()."""
        pass
    

class NumericalDiffCost(Cost):

    """Finite difference approximated Instantaneous Cost.
    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.
    """

    def quadraticize(self, x, u, terminal=False):
        #returns L_x,L_u,L_xx,L_uu,L_ux
        nx = x.shape[0]
        nu = u.shape[0]
        
        L_x = np.vstack([
            approx_fprime(x, lambda x: self.l(x, u)[i], EPS) for i in range(self.n_x)
        ])
        
        L_u = np.vstack([
            approx_fprime(u, lambda u: self.f(x, u)[i], EPS) for i in range(self.n_u)
        ])
        
        
        L_xx = np.vstack([
            
            approx_fprime(x, lambda x: L_x[i], EPS) for i in range(self.n_x)
            
        ])
        
        L_uu = np.vstack([
            
            approx_fprime(u, lambda u: L_u[i], EPS) for i in range(self.n_u)
            
        ])
        
        
        L_ux = np.vstack([
            
            approx_fprime(x, lambda x: L_u[i], EPS) for i in range(self.n_x)
            
        ])
        
        return
        
        

        
class ReferenceCost(Cost):
    """
    The cost of a state and control from some reference trajectory.
    """
    
    def __init__(self, xf, Q, R, Qf=None, weight=1.0):
        self.xf = xf.astype(float)
        self.Q = Q.astype(float)
        self.R = R.astype(float)
        self.weight = weight
        
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
        nx = x.shape[0]
        nu = u.shape[0]
        
        L_x = (x - self.xf).T @ self.Q_plus_QT
        L_u = u.T @ self.R_plus_RT
        L_xx = self.Q_plus_QT
        L_uu = self.R_plus_RT
        L_ux = np.zeros((nu, nx))
        
        if terminal:
            L_x = (x - self.xf).T @ (self.Qf + self.Qf.T)
            L_xx = self.Qf + self.Qf.T
            L_u = np.zeros((nu))
            L_uu = np.zeros((nu, nu))
        
        return L_x, L_u, L_xx, L_uu, L_ux
    
    def plot(self):
        ax = plt.gca()
        if "1D" in ax.get_title():
            ax.axhline(self.xf[0], c='r', label='$x_f$: ' + str(self.xf))
            return
        ax.scatter(self.xf[0], self.xf[1], 80, 'r', 'x', label='$x_{goal}$')


class ObstacleCost(Cost):
    """
    The cost of an operating point from a stationary obstacle.
    """
    
    def __init__(self, position_inds, point, radius, weight=1.0):
        self.ix, self.iy = position_inds
        self.point = point
        self.radius = radius
        self.weight = weight
    
    def __call__(self, x, _, __=None):
        distance = np.sqrt((x[self.ix] - self.point.x)**2 + (x[self.iy] - self.point.y)**2)
        return min(0.0, distance - self.radius)**2
    
    def state_point(self, x):
        """Reformat the position within a state into a Point object."""
        return Point(x[self.ix], x[self.iy])
    
    def quadraticize(self, x, u, _=None):
        nx = x.shape[0]
        nu = u.shape[0]
        
        L_x = np.zeros((nx))
        L_u = np.zeros((nu))
        L_xx = np.zeros((nx, nx))
        L_uu = np.zeros((nu, nu))
        L_ux = np.zeros((nu, nx))
        
        # Compute jacobians and hessians as 2x1 and 2x2 as a function of the positions,
        # then reformat by adding zeros to match the state dimensions.
        L_x_obs, L_xx_obs = \
            _quadraticize_distance(self.state_point(x), self.point, self.radius)
        
        L_x[self.ix] = L_x_obs[IX]
        L_x[self.iy] = L_x_obs[IY]
        L_xx[self.ix,self.ix] = L_xx_obs[IX,IX]
        L_xx[self.iy,self.iy] = L_xx_obs[IY,IY]
        L_xx[self.ix,self.iy] = L_xx[self.iy,self.ix] = L_xx_obs[IX,IY]
        
        return L_x, L_u, L_xx, L_uu, L_ux
    
    def plot(self):
        circle = plt.Circle(
            (self.point.x, self.point.y), self.radius, 
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
    
    def __init__(self, pos_inds, radius, weight=1.0):
        self.pos_inds = pos_inds
        self.radius = radius
        self.n_agents = len(pos_inds)
        self.weight = weight
    
    def __call__(self, x):
        total_cost = 0.0
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                distance = np.linalg.norm(
                    x[..., self.pos_inds[i]] - x[..., self.pos_inds[j]])
                total_cost += min(0.0, distance - self.radius)**2
        return total_cost
    
    def quadraticize(self, x):
        nx = x.shape[0]
        L_x = np.zeros((nx))
        L_xx = np.zeros((nx, nx))
        
        for i in range(self.n_agents):            
            for j in range(i+1, self.n_agents):
                L_xi = np.zeros((nx))
                L_xxi = np.zeros((nx, nx))
                
                L_x_pair, L_xx_pair = _quadraticize_distance(
                    Point(*x[..., self.pos_inds[i]]), 
                    Point(*x[..., self.pos_inds[j]]), 
                    self.radius
                )
                
                ix, iy = self.pos_inds[i]
                jx, jy = self.pos_inds[j]
                
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
    Collects a reference and obstacle costs for one agent in the game.
    """
    
    def __init__(self, costs):
        assert all(isinstance(cost, ReferenceCost) 
                or isinstance(cost, ObstacleCost) for cost in costs)
        self.costs = costs
        
    def __call__(self, x, u, terminal=False):
        full_cost = 0.0
        for cost in self.costs:
            full_cost += cost.weight * cost(x, u, terminal)
        return full_cost
    
    def quadraticize(self, x, u, terminal=False):
        nx = x.shape[0]
        nu = u.shape[0]
        
        L_x = np.zeros((nx))
        L_u = np.zeros((nu))
        L_xx = np.zeros((nx, nx))
        L_uu = np.zeros((nu, nu))
        L_ux = np.zeros((nu, nx))
        
        for cost in self.costs:
            L_xi, L_ui, L_xxi, L_uui, L_uxi = cost.quadraticize(x, u, terminal)
            L_x += cost.weight * L_xi
            L_u += cost.weight * L_ui
            L_xx += cost.weight * L_xxi
            L_uu += cost.weight * L_uui
            L_ux += cost.weight * L_uxi
        
        return L_x, L_u, L_xx, L_uu, L_ux
    
    def plot(self):
        """Call the children cost.plot functions"""
        for cost in self.costs:
            cost.plot()
        

class GameCost(Cost):
    """
    Collects the costs for multiple agents as well as coupling costs between the agents.
    """
    
    def __init__(self, 
                 agent_costs, 
                 coupling_costs, 
                 x_dims, 
                 u_dims):
        # NOTE: we assume that agent_costs are passed in the same order as their respective 
        # dynamics in the MultiDynamicalModel
        
        self.agent_costs = agent_costs
        self.coupling_costs = coupling_costs
        # TODO: only store these in one place.
        self.x_dims = x_dims
        self.u_dims = u_dims
        
    def __call__(self, x, u, terminal=False):
        x_split = np.split(x, np.cumsum(self.x_dims[:-1]))
        u_split = np.split(u, np.cumsum(self.u_dims[:-1]))
        
        agent_total = sum(agent_cost(xi, ui, terminal)
                          for agent_cost, xi, ui in zip(self.agent_costs, x_split, u_split))
        coupling_total = sum(coupling_cost.weight * coupling_cost(x) 
                             for coupling_cost in self.coupling_costs)
        
        return agent_total + coupling_total
    
    def quadraticize(self, x, u, terminal=False):
        L_xs, L_us = [], []
        L_xxs, L_uus, L_uxs = [], [], []
        
        x_split = np.split(x, np.cumsum(self.x_dims[:-1]))
        u_split = np.split(u, np.cumsum(self.u_dims[:-1]))
        
        # Compute agent quadraticizations in individual state spaces.
        for agent_cost, xi, ui in zip(self.agent_costs, x_split, u_split):
            L_xi, L_ui, L_xxi, L_uui, L_uxi = agent_cost.quadraticize(xi, ui, terminal)
            L_xs.append(L_xi)
            L_us.append(L_ui)
            L_xxs.append(L_xxi)
            L_uus.append(L_uui)
            L_uxs.append(L_uxi)
        
        L_x = np.hstack(L_xs)
        L_u = np.hstack(L_us)
        L_xx = block_diag(*L_xxs)
        L_uu = block_diag(*L_uus)
        L_ux = block_diag(*L_uxs)
        
        # Incorporate coupling costs in full cartesian state space.
        for coupling_cost in self.coupling_costs:
            L_x_coup, L_xx_coup = coupling_cost.quadraticize(x)
            L_x += coupling_cost.weight * L_x_coup
            L_xx += coupling_cost.weight * L_xx_coup
        
        return L_x, L_u, L_xx, L_uu, L_ux
    
    def plot(self, 
             surface_plot=False, 
             agent_ind=0, 
             couple_ind=0,
             axis=None, 
             log_colors=False):
        """Call the agent_cost.plot functions and overlay a sampled cost surface
           over the current axes.
        """
        
        for agent_cost in self.agent_costs:
            agent_cost.plot()
            
        if not surface_plot:
            return
        
        STEP = 0.1
        imshow_kwargs = {}
        
        # Hope that the current axes are square if none supplied.
        if not axis:
            ax = plt.gca()
            axis = ax.get_xlim() + ax.get_ylim()
        
        if log_colors:
            imshow_kwargs['norm'] = LogNorm()

        # Sample the positional state space over axis.
        pts = np.mgrid[axis[0]:axis[1]:STEP, axis[2]:axis[3]:STEP].T.reshape(-1,2)
        u = np.zeros(self.u_dims[agent_ind])

        costs = []
        for pt in pts:
            xi = np.resize(pt, self.x_dims[agent_ind])
            costs.append(self.agent_costs[agent_ind](xi, u, False))

        side_len = round(np.sqrt(pts.shape[0]))
        costs = np.flip(np.array(costs).reshape(side_len, side_len), axis=0)
        # Q: why does NoNorm have weird behavior here?
        cost_h = plt.imshow(costs, extent=axis, interpolation='bilinear', **imshow_kwargs)
        
        # plt.title('Cost Surface')
        # plt.xlabel('x [m]')
        # plt.ylabel('y [m]')
        plt.colorbar(cost_h, label='Cost')
        
        # Plot coupling costs
        if self.coupling_costs:
            self.coupling_costs[couple_ind].plot()
    
    
def _quadraticize_distance(point_a, point_b, max_distance):
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

