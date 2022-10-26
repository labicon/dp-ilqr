import numpy as np
from casadi import *
import do_mpc
import decentralized as dec
from baseline_model import baseline_drone_model
import util

def baseline_drone_mpc(model, n_agents, x_baseline, x_dims, v_max, theta_max, phi_max, tau_max):
    """ 
    x_baseline : current concatenated states of all agents
    
    """
    mpc = do_mpc.controller.MPC(model)

    
    setup_mpc = {
        'n_horizon': 5,
        'n_robust': 0,
        'open_loop': 0,
        't_step': 0.1,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 2,
        'store_full_solution': True
    }

    mpc.set_param(**setup_mpc)
    
    mpc.set_nl_cons('collision_constraint', -model.aux['proximity_cost'],-0.)
    #in this case we want the collision avoidance cost inccured to be 0, which means their 
    #pairwise distances must be > radius

    """
    TODO: how to separate the goal_costs and terminal_costs for each agent and add them to mpc.set_objective?
    mterm: terminal cost
    lterm: stage cost
    """
    mterm = model.aux['total_terminal_cost']
    lterm = model.aux['total_stage_cost']
    
    mpc.set_objective(mterm=mterm,lterm=lterm)
    
    mpc.set_rterm(u=np.array([[0],[0],[0]]))

    max_input = np.array([[theta_max], [phi_max], [tau_max]])

    mpc.bounds['lower', '_u', 'u'] = -max_input
    mpc.bounds['upper', '_u', 'u'] = max_input
    
    max_state = np.array([[6.5], [6.5], [6.5], [v_max],[v_max], [v_max]]) 
    #v_max refers to the max velocity in each direction
    #constraints on x,y,z position is set as 6.5 because the actual flying arena has a limited space
    mpc.bounds['lower','_x', 'x'] = -max_state
    mpc.bounds['upper','_x', 'x'] = max_state

    mpc.setup()


    return mpc


    