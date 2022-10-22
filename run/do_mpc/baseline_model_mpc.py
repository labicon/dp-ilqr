import numpy as np
from casadi import *
import do_mpc
import decentralized as dec

def baseline_drone_mpe(model, v_max, theta_max, phi_max, tau_max, \
                      xf, x_dims, ids, Q, R, Qf, n_agents, n_dims, radius):

    mpc = do_mpc.controller.MPC(model)

    """Costs:"""
    
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

    mpc.set_nl_cons('collision',model.aux['collision_constraint'],0) 
    #in this case we want the collision avoidance cost inccured between any 2 agents to be 0, which means their 
    #pairwise distances must be > radius

    """
    TODO: how to separate the goal_costs and terminal_costs for each agent and add them to mpc.set_objective?
    mterm: terminal cost
    lterm: stage cost
    """
    mterm = model.aux['total_terminal_cost']
    lterm = model.aux['total_stage_costs']
    
    mpc.set_objective(mterm=mterm,lterm=lterm)

    max_input = np.array([[v_max], [v_max], [v_max], [theta_max], [phi_max], [tau_max]])

    mpc.bounds['lower', '_u', 'u'] = -max_input
    mpc.bounds['upper', '_u', 'u'] = max_input

    mpc.setup()


    return mpc


    