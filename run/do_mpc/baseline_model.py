import numpy as np
from casadi import *
import do_mpc
import decentralized as dec


def baseline_drone_model(xf, x_dims, ids, Q, R, Qf, n_agents, n_dims, radius): 
    #input arguments: np.ndarray
    Qs = [Q] * n_agents
    Rs = [R] * n_agents
    Qfs = [Qf] * n_agents

    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    x = model.set_variable(var_type='_x', var_name='x', shape=(6, 1))
    u = model.set_variable(var_type='_u', var_name='u', shape=(3, 1))
    g = 9.81

    #x = p_x,p_y,p_z,v_x,v_y,v_z
    #u = theta, phi, tau

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

    """Costs:"""
    goal_costs = [dec.ReferenceCost(xf_i, Qi, Ri, Qfi, id_)
    for xf_i, id_, x_dim, Qi, Ri, Qfi in zip(
        dec.split_agents_gen(xf, x_dims), ids, x_dims, Qs, Rs, Qfs
    )]

    for m in goal_costs:

        model.set_expression('goal_cost', m)

    #collision avoidance will now be handled through constraints rather than a quadratic cost!

    """Constraints:"""
    
    prox_cost = dec.ProximityCost(x_dims, radius, n_dims)

    model.setup()

    return model


