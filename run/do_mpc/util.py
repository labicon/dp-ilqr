import numpy as np
import decentralized as dec
import itertools
from casadi import *
# import do_mpc

def paper_setup_3_quads():
    x0 = np.array([[0.5, 1.5, 1, 0, 0, 0,
                    2.5, 1.5, 1, 0, 0, 0,
                    1.5, 1.3, 1, 0, 0, 0]], 
                     dtype=float).T
    xf = np.array([[2.5, 1.5, 1, 0, 0, 0, 
                    0.5, 1.5, 1, 0, 0, 0, 
                    1.5, 2.2, 1, 0, 0, 0]]).T
    # x0[dec.pos_mask([6]*3, 3)] += 0.01*np.random.randn(9, 1)
    # xf[dec.pos_mask([6]*3, 3)] += 0.01*np.random.randn(9, 1)
    return x0, xf


def paper_setup_4_quads():
    x0 = np.array([[0.5, 1.5, 1, 0, 0, 0,
                    2.5, 1.5, 1, 0, 0, 0,
                    1.5, 1.3, 1, 0, 0, 0,
                    -1.1, 1.5, 1, 0, 0, 0]], 
                     dtype=float).T
    xf = np.array([[2.5, 1.5, 1, 0, 0, 0, 
                    0.5, 1.5, 1, 0, 0, 0, 
                    1.5, 2.2, 1, 0, 0, 0,
                   0.5,  0.2, 1, 0, 0, 0,]]).T
    # x0[dec.pos_mask([6]*3, 3)] += 0.01*np.random.randn(9, 1)
    # xf[dec.pos_mask([6]*3, 3)] += 0.01*np.random.randn(9, 1)
    return x0, xf



def paper_setup_1_quad():
    x0 = np.array([[0.5, 1.5, 1, 0, 0, 0]],dtype=float).T
    xf = np.array([[2.5, 1.5, 1, 0, 0, 0]]).T
    return x0, xf


def compute_pairwise_distance_Sym(X, x_dims, n_d=3):
    """Compute the distance between each pair of agents"""
    assert len(set(x_dims)) == 1

    n_agents = len(x_dims)
    n_states = x_dims[0]

    if n_agents == 1:
        raise ValueError("Can't compute pairwise distance for one agent.")  
    
    pair_inds = np.array(list(itertools.combinations(range(n_agents), 2)))
    
    X_agent = reshape(X,(n_agents, n_states))
    distances = []
    
    if n_agents == 2:
        dX=X_agent[0,0:3]-X_agent[1,0:3]
        distances.append(sqrt(dX[0]**2+dX[1]**2+dX[2]**2))
        
    else:
        dX = X_agent[:n_d, pair_inds[:, 0]] - X_agent[:n_d, pair_inds[:, 1]]
        for j in range(dX.shape[1]):
            distances.append(sqrt(dX[0,j]**2+dX[1,j]**2+dX[2,j]**2))
            
    return distances #this is a list of symbolic pariwise distances


