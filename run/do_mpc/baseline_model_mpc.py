import numpy as np
from casadi import *
import do_mpc
import decentralized as dec
from baseline_model import baseline_drone_model
import util

# 1-drone MPC setup:
# def baseline_drone_mpc(model, n_agents, x_baseline, x_dims, v_max, theta_max, phi_max, tau_max):
#     """ 
#     x_baseline : current concatenated states of all agents
    
#     """
#     mpc = do_mpc.controller.MPC(model)

#     setup_mpc = {
#         'n_horizon': 15,
#         'n_robust': 0,
#         'open_loop': 0,
#         't_step': 0.1,
#         'state_discretization': 'collocation',
#         'collocation_type': 'radau',
#         'collocation_deg': 2,
#         'store_full_solution': True
#     }

#     mpc.set_param(**setup_mpc)

#     mterm = model.aux['total_terminal_cost']
#     lterm = model.aux['total_stage_cost']
    
#     mpc.set_objective(mterm=mterm,lterm=lterm)

#     max_input = np.array([[theta_max], [phi_max], [tau_max]])

#     mpc.bounds['lower', '_u', 'u'] = -max_input
#     mpc.bounds['upper', '_u', 'u'] = max_input
    
#     mpc.set_rterm(u=np.array([[1],[1],[1]])) 
#     #seems like this is needed for reasonably short simulation runtime...weird!
    
#     mpc.setup()
    
# #     opt_labels = mpc.x.labels()
# #     labels_lb_viol =np.array(opt_labels)[np.where(lb_viol)[0]]
# #     labels_ub_viol =np.array(opt_labels)[np.where(lb_viol)[0]]
    
#     return mpc






#*****************************
#Below is for centralized set-up for 3 drones:
def baseline_drone_mpc(model, n_agents, x_baseline, x_dims, v_max, theta_max, phi_max, tau_max):
    """ 
    x_baseline : current concatenated states of all agents
    
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 15,
        'n_robust': 1,
        'open_loop': 0,
        't_step': 0.05,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 2,
        'store_full_solution': True
    }

    mpc.set_param(**setup_mpc)
     
    #We want the pairwise distance > 0.5
    mpc.set_nl_cons('collision1', -model.aux['collision_avoidance1'], -0.5) 
    mpc.set_nl_cons('collision2', -model.aux['collision_avoidance2'], -0.5)
    mpc.set_nl_cons('collision3', -model.aux['collision_avoidance3'], -0.5)
    #in this case we want the collision avoidance cost inccured to be 0, which means their 
    #pairwise distances must be > radius
    mterm = model.aux['total_terminal_cost']
    lterm = model.aux['total_stage_cost']
    
    mpc.set_objective(mterm=mterm,lterm=lterm)

    max_input = np.array([[theta_max], [phi_max], [tau_max], \
                          [theta_max], [phi_max], [tau_max], \
                          [theta_max], [phi_max], [tau_max]])
    
    min_input = np.array([[theta_max], [phi_max], [0], \
                          [theta_max], [phi_max], [0], \
                          [theta_max], [phi_max], [0]])
    
    #Note: final error is much smaller when inputs are not constrained?!
    
    mpc.bounds['lower', '_u', 'u'] = -min_input
    mpc.bounds['upper', '_u', 'u'] = max_input
    
    max_state_upper = np.array([[5], [5], [6], [v_max],[v_max], [v_max],\
                          [5], [5], [6], [v_max],[v_max], [v_max],\
                          [5], [5], [6], [v_max],[v_max], [v_max]])
    
    max_state_lower = np.array([[5], [5], [0], [v_max],[v_max], [v_max],\
                          [5], [5], [0], [v_max],[v_max], [v_max],\
                          [5], [5], [0], [v_max],[v_max], [v_max]])
    
    # v_max refers to the max velocity in each direction

    mpc.bounds['lower','_x', 'x'] = -max_state_lower
    mpc.bounds['upper','_x', 'x'] = max_state_upper

    mpc.set_rterm(u=np.array([[0],[0],[0],\
                             [0],[0],[0],\
                             [0],[0],[0]])) 
    
    # mpc.set_rterm(u=np.array([[1],[1],[1],[1],[1],[1],\
    #                          [1],[1],[1],[1],[1],[1],\
    #                          [1],[1],[1],[1],[1],[1]])) 
    
    mpc.setup()
    
#     opt_labels = mpc.x.labels()
#     labels_lb_viol =np.array(opt_labels)[np.where(lb_viol)[0]]
#     labels_ub_viol =np.array(opt_labels)[np.where(lb_viol)[0]]
    
    return mpc










#**********************************
#Below is for decentralized set-up:
# def baseline_drone_mpc(model, n_agents, x_baseline, x_dims, v_max, theta_max, phi_max, tau_max):
#     """ 
#     x_baseline : current concatenated states of all agents
    
#     """
#     mpc = do_mpc.controller.MPC(model)

    
#     setup_mpc = {
#         'n_horizon': 5,
#         'n_robust': 0,
#         'open_loop': 0,
#         't_step': 0.1,
#         'state_discretization': 'collocation',
#         'collocation_type': 'radau',
#         'collocation_deg': 2,
#         'collocation_ni': 2,
#         'store_full_solution': True
#     }

#     mpc.set_param(**setup_mpc)
     
#     mpc.set_nl_cons('collision_constraint', -model.aux['proximity_cost'],-1)
#     #in this case we want the collision avoidance cost inccured to be 0, which means their 
#     #pairwise distances must be > radius

#     """
#     TODO: how to separate the goal_costs and terminal_costs for each agent and add them to mpc.set_objective?
#     mterm: terminal cost
#     lterm: stage cost
#     """
#     mterm = model.aux['total_terminal_cost']
#     lterm = model.aux['total_stage_cost']
    
#     mpc.set_objective(mterm=mterm,lterm=lterm)
    
#     mpc.set_rterm(u=np.array([[0],[0],[0]]))

#     max_input = np.array([[theta_max], [phi_max], [tau_max]])

#     mpc.bounds['lower', '_u', 'u'] = -max_input
#     mpc.bounds['upper', '_u', 'u'] = max_input
    
#     max_state = np.array([[6.5], [6.5], [6.5], [v_max],[v_max], [v_max]]) 
#     #v_max refers to the max velocity in each direction
#     #constraints on x,y,z position is set as 6.5 because the actual flying arena has a limited space
#     mpc.bounds['lower','_x', 'x'] = -max_state
#     mpc.bounds['upper','_x', 'x'] = max_state

    
    
#     mpc.setup()
    
#     """
#     Retrieve the labels from the optimization variables and find those that are violating the constraints:
#     """

#     return mpc


    