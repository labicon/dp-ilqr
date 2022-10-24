import numpy as np
from casadi import *
import do_mpc
import decentralized as dec

def baseline_drone_mpc(model, v_max, theta_max, phi_max, tau_max):

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

    mpc.set_nl_cons('collision',model.aux['total_prox_cost'],0) 
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

    max_input = np.array([[theta_max], [phi_max], [tau_max]])

    mpc.bounds['lower', '_u', 'u'] = -max_input
    mpc.bounds['upper', '_u', 'u'] = max_input
    
    # mpc.bounds['lower','_x', 'x'] = -v_max
    # mpc.bounds['upper','_x', 'x'] = v_max

    mpc.setup()


    return mpc


    