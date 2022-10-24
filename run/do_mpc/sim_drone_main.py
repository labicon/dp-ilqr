import time
import concurrent.futures
import numpy as np
from casadi import *
import do_mpc
import util

# import sys

from baseline_model import *
from baseline_model_mpc import *
from baseline_model_simulator import *


n_agents = 3
n_states = 6
n_inputs = 3

def setup_baseline(x_baseline, x_baseline_f, v_max, theta_max, phi_max, tau_max,\
                    x_dims, Q, R, Qf, n_agents, n_dims, radius):
    model_baseline = baseline_drone_model(x_baseline_f, x_dims, Q, R, Qf, n_agents, n_dims, radius)
    mpc_baseline = baseline_drone_mpc(model_baseline, v_max, theta_max, phi_max, tau_max)
    simulator_baseline = baseline_drone_simulator(model_baseline)
    simulator_baseline.x0['x'] = x_baseline
    mpc_baseline.x0 = x_baseline

    u_init_baseline = np.full((n_agents*n_inputs,1), 0.0)
    mpc_baseline.u0 = u_init_baseline
    simulator_baseline.u0 = u_init_baseline
    mpc_baseline.set_initial_guess()
    
    u0_baseline = mpc_baseline.make_step(x_baseline)
    x_baseline_next = simulator_baseline.make_step(u0_baseline)
    # print(mpc_baseline.data._lam_g_num)
    return u0_baseline, x_baseline_next, mpc_baseline.data._lam_g_num

def run_sim():
    
    theta_max = np.pi/6
    phi_max = np.pi/6
    tau_max = 5
    v_max = 5

    x_dims = [n_agents]*n_states

    Q = np.eye(n_states)*10
    Qf = np.eye(n_states)*1e3
    R = np.eye(n_inputs)

    radius = 0.5
    n_dims = [3,3,3]
    episode=200
    x_baseline_init, x_baseline_f = util.paper_setup_3_quads()

    x_baseline1 = x_baseline_init

    states_list = np.zeros((episode+1,9)) #positions of each drone
    states_list[0,:] = np.array([x_baseline1[0],x_baseline1[1],x_baseline1[2],\
                            x_baseline1[6],x_baseline1[7],x_baseline1[8],\
                            x_baseline1[12],x_baseline1[13],x_baseline1[14]]).flatten()

    # vx = 0
    # vy = 0
    # vz = 0
    # x_prev = x_baseline_init[0]
    # y_prev = x_baseline_init[1]
    # z_prev = x_baseline_init[2]
    time_start = time.perf_counter()
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for k in range(episode):
    
            f = executor.submit(setup_baseline(x_baseline1, x_baseline_f, v_max, theta_max, phi_max, tau_max,\
                    x_dims, Q, R, Qf, n_agents, n_dims, radius), x_baseline1, x_baseline_f, 1.8)

            _, x_baseline1, la_mul = f.result()

            print("Lagrange Multiplier: ", la_mul)
            print("Length of Lagrange Multiplier: ", len(la_mul[0]))
        # ------------------------------------------------------------

            x = np.array(x_baseline1)
            x.flatten()
            x = x.ravel()
            states_list[k+1] = np.array([x[0], x[1], x[2], x[6], x[7], x[8], x[12], x[13], x[14]]) 
            #pos updates of 3 drones

            """
            TODO: how to separate the velocity vectors of each agent?
            """
            # vx = -(x_prev - x_baseline1[0])/0.1
            # vy = -(y_prev - x_baseline1[1])/0.1
            # vz = -(z_prev - x_baseline1[2])/0.1
            # x_prev = x_joint[0]
            # y_prev = x_joint[1]
            # z_prev = x_joint[2]

    time_finish = time.perf_counter()
    print("Total time: ", time_finish - time_start)
    np.save('drone_sim_data', states_list)

if __name__ == '__main__':
    run_sim()