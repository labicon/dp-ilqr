from tkinter import N
import numpy as np
from casadi import *
import do_mpc
import decentralized as dec
import util

def baseline_drone_model(xf, x_dims, Q, R, Qf, n_agents, n_dims, radius): 
    #input arguments: np.ndarray
    Qs = [Q] * n_agents
    Rs = [R] * n_agents
    Qfs = [Qf] * n_agents

    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    x = model.set_variable(var_type='_x', var_name='x', shape=(6, 1))
    u = model.set_variable(var_type='_u', var_name='u', shape=(3, 1))
    print(f'Shape of x is {x.shape}')
    print(f'Shape of u is {u.shape}')
    #x = p_x,p_y,p_z,v_x,v_y,v_z
    #u = theta, phi, tau
    g = 9.81
    model.set_rhs('x', vertcat(x[3], x[4], x[5], g*np.tan(u[0]), -g*np.tan(u[1]), u[2]-g))
    
    #6-D model of a quadrotor with 6 states and 3 inputs:
    """
    p_x_dot = v_x
    p_y_dot = v_y
    p_z_dot = v_z
    v_x_dot = g*tan(theta)
    v_y_dot = -g*tan(phi)
    v_z_dot = tau-g
    
    The inputs are [theta,phi,tau]
    The states are [p_x,p_y,p_z,v_x,v_y,v_z]

    """

    """Distributed cost functions:"""

    
    total_stage_cost= [(x-xf_i.T).T@Qi@(x-xf_i.T) + u.T@Ri@u
    for xf_i, Qi, Ri, in zip(
        dec.split_agents(xf.reshape(1,-1), x_dims), Qs, Rs 
    )]
 

    total_terminal_cost = [(x-xf_i.T).T@Qfi@(x-xf_i.T)

    for xf_i, Qfi in zip(
        dec.split_agents(xf.reshape(1,-1), x_dims), Qfs
    )]

    model.set_expression('total_stage_cost',np.sum(total_stage_cost))
    model.set_expression('total_terminal_cost',np.sum(total_terminal_cost))
    

    #collision avoidance will later be handled through constraints rather than a quadratic cost!

    """Constraints:"""
    
    
    #TODO: The prox_cost is not correct; needs a compatible expression
    
    distances = util.compute_pairwise_distance_Sym(xf,x_dims,n_d=3) 
    prox_costs = SX(np.fmin(np.zeros(1), distances - radius) ** 2 * 100)
    
    model.set_expression('total_prox_cost',prox_costs)

    model.setup()

    return model


