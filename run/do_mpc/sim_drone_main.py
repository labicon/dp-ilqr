import time
import concurrent.futures
import numpy as np
from casadi import *
import do_mpc
# import sys

from baseline_model import *
from baseline_model_mpc import *
from baseline_model_simulator import *

def setup_baseline(x_baseline, x_baseline_f, v_max, theta_max, phi_max, tau_max,\
                    x_dims, Q, R, Qf, n_agents, n_dims, radius):
    model_baseline = baseline_drone_model(model_baseline, x_baseline_f, x_dims, Q, R, Qf, n_agents, n_dims, radius)
    mpc_baseline = baseline_drone_mpc(model_baseline, v_max, theta_max, phi_max, tau_max)
    simulator_baseline = baseline_drone_simulator(model_baseline)
    simulator_baseline.x0['x'] = x_baseline
    mpc_baseline.x0 = x_baseline

    u_init_baseline = np.full((9,1), 0.0)
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

    n_agents = 3
    n_states = 6
    n_inputs = 3

    x_dims = [n_agents]*n_states

    Q = np.eye(n_states)*10
    Qf = np.eye(n_states)*1e3
    R = np.eye(n_states)

    radius = 0.5

    episode=200
    states_list=np.zeros((episode+1, n_agents*n_states))

    # x_baseline1_init = np.array([0.0, 2.0, 0, 0, 0, 0])
    x_baseline_init = np.random.rand(n_agents*n_states)

    x_joint = x_baseline_init

    x_baseline_f = np.zeros(18)

    x_baseline1 = x_baseline_init

    # states_list[0] = np.array([x_joint[0], x_joint[1], x_joint[2], x_joint[6], x_joint[7], x_joint[8]])
    states_list = np.array([*x_joint])
    vx = 0
    vy = 0
    vz = 0
    x_prev = x_baseline_init[0]
    y_prev = x_baseline_init[1]
    z_prev = x_baseline_init[2]

    # states_list[0] = np.array([x_joint[0], x_joint[1], x_joint[2], x_joint[6], x_joint[7], x_joint[8]])

    time_start = time.perf_counter()
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for k in range(episode):
    
            f = executor.submit(setup_baseline, x_baseline1, x_baseline_f, 1.8)

            _, x_joint = f.result()
            _, x_baseline1, la_mul = f.result()
            # la_mul_array = [x for x in la_mul]
            print("Lagrange Multiplier: ", la_mul)
            print("Length of Lagrange Multiplier: ", len(la_mul[0]))
        # ------------------------------------------------------------
            x_joint[6] = x_baseline1[0]
            x_joint[7] = x_baseline1[1]
            x_joint[8] = x_baseline1[2]
            x_joint[9] = x_baseline1[3]
            x_joint[10] = x_baseline1[4]
            x_joint[11] = x_baseline1[5]

            x = np.array(x_joint)
            x.flatten()
            x = x.ravel()
            states_list[k+1] = np.array([*x])

            vx = -(x_prev - x_joint[0])/0.1
            vy = -(y_prev - x_joint[1])/0.1
            vz = -(z_prev - x_joint[2])/0.1
            x_prev = x_joint[0]
            y_prev = x_joint[1]
            z_prev = x_joint[2]

    time_finish = time.perf_counter()
    print("Total time: ", time_finish - time_start)
    np.save('two_drone_sine_sim_data', states_list)

if __name__ == '__main__':
    run_sim()