import numpy as np
from casadi import *
import do_mpc
import decentralized as dec

def baseline_drone_mpe(model,xf, x_dims, ids, Q, R, Qf, n_agents, n_dims, radius):

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

    
    

  
    mpc.set_objective(mterm=mterm,lterm=lterm)

    model.set_expression('collision_constraint',prox_cost)