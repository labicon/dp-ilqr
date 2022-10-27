from tkinter import N
import numpy as np
from casadi import *
import do_mpc
import decentralized as dec
import util
#******************************************************
#Below is the centralized set-up

def baseline_drone_model(xf, Q, R, Qf, x_baseline, x_dims): 
    
    """
    Below is the distributed set-up
    xf: concatenated goal state vector for all agents
    
    """
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)
    
    #x = p_x,p_y,p_z,v_x,v_y,v_z
    #u = theta, phi, tau
    
    #Concantenated states of all agents
    x = model.set_variable(var_type='_x', var_name='x', shape=(18, 1))
    #Concatenated inputs of all agents
    u = model.set_variable(var_type='_u', var_name='u', shape=(9, 1))
    
    # print(f'Shape of x is {x.shape}')
    # print(f'Shape of u is {u.shape}')
    
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
    g = 9.81
    model.set_rhs('x', vertcat(x[3], x[4], x[5], g*np.tan(u[0]), -g*np.tan(u[1]), u[2]-g,\
                               x[9], x[10], x[11], g*np.tan(u[3]), -g*np.tan(u[4]), u[5]-g,\
                              x[15], x[16], x[17], g*np.tan(u[4]), -g*np.tan(u[5]), u[6]-g))
    

    total_stage_cost = (x-xf).T@Q@(x-xf) + u.T@R@u 
    total_terminal_cost = (x-xf).T@Qf@(x-xf)
    
    model.set_expression('total_stage_cost',total_stage_cost)
    model.set_expression('total_terminal_cost',total_terminal_cost)
    
    radius = 0.5
    #x_baseline is concatenated states of all agents
    distances = util.compute_pairwise_distance_Sym(x_baseline,x_dims,n_d=3) 
    
    distances = SX(distances)
    # prox_cost = sum1(SX(np.fmin(np.zeros(1), distances - radius) ** 2))
    # model.set_expression('proximity_cost',prox_cost) 
    
    model.set_expression('collision_constraint',distances)
    
    #The prox_cost should be handled by a centralized processor since xf only contains the state
    #a single agent
    #Set this up in the mpc controller

    model.setup()

    return model
























#******************************************************
#Below is the decentralized set-up
# def baseline_drone_model(xf, Q, R, Qf, x_baseline, x_dims): 
#     """
#     Below is the distributed set-up
    
#     """
#     #input arguments: np.ndarray
#     # Qs = [Q] * n_agents
#     # Rs = [R] * n_agents
#     # Qfs = [Qf] * n_agents

#     model_type = 'continuous' # either 'discrete' or 'continuous'
#     model = do_mpc.model.Model(model_type)
    
    
#     #x = p_x,p_y,p_z,v_x,v_y,v_z
#     #u = theta, phi, tau
#     x = model.set_variable(var_type='_x', var_name='x', shape=(6, 1))
#     u = model.set_variable(var_type='_u', var_name='u', shape=(3, 1))
#     # print(f'Shape of x is {x.shape}')
#     # print(f'Shape of u is {u.shape}')
    
    
#     g = 9.81
#     model.set_rhs('x', vertcat(x[3], x[4], x[5], g*np.tan(u[0]), -g*np.tan(u[1]), u[2]-g))
    
#     #6-D model of a quadrotor with 6 states and 3 inputs:
#     """
#     p_x_dot = v_x
#     p_y_dot = v_y
#     p_z_dot = v_z
#     v_x_dot = g*tan(theta)
#     v_y_dot = -g*tan(phi)
#     v_z_dot = tau-g
    
#     The inputs are [theta,phi,tau]
#     The states are [p_x,p_y,p_z,v_x,v_y,v_z]

#     """
    
#     # total_stage_cost= [(x-xf_i.T).T@Qi@(x-xf_i.T) + u.T@Ri@u
#     # for xf_i, Qi, Ri, in zip(
#     #     dec.split_agents(xf.reshape(1,-1), x_dims), Qs, Rs 
#     # )]
#     total_stage_cost = (x-xf).T@Q@(x-xf) + u.T@R@u 
#     #stage for 1 agent
 
#     total_terminal_cost = (x-xf).T@Qf@(x-xf)
    
#     model.set_expression('total_stage_cost',total_stage_cost)
#     model.set_expression('total_terminal_cost',total_terminal_cost)
    
#     radius = 0.5
#     #x_baseline is concatenated states of all agents
#     distances = util.compute_pairwise_distance_Sym(x_baseline,x_dims,n_d=3) 
#     prox_cost = sum1(SX(np.fmin(np.zeros(1), distances - radius) ** 2))
#     model.set_expression('proximity_cost',prox_cost) 
#     #The prox_cost should be handled by a centralized processor since xf only contains the state
#     #a single agent
#     #Set this up in the mpc controller

#     model.setup()

#     return model


