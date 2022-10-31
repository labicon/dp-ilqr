import time
import concurrent.futures
import numpy as np
from casadi import *
import do_mpc
import util
import decentralized as dec

# import sys

from baseline_model import *
from baseline_model_mpc import *
from baseline_model_simulator import *


#1-drone MPC:
# def setup_baseline(x_baseline, x_baseline_f, v_max, theta_max, phi_max, tau_max,\
#                     x_dims, Q, R, Qf, n_agents, n_inputs, n_dims):
#     model_baseline = baseline_drone_model(x_baseline_f, Q, R, Qf, x_baseline, x_dims)  

    
#     mpc_baseline = baseline_drone_mpc(model_baseline,n_agents,x_baseline,\
#                                       x_dims, v_max, theta_max, phi_max,tau_max)

    
#     simulator_baseline = baseline_drone_simulator(model_baseline) 

        
#     simulator_baseline.x0['x'] = x_baseline
#     mpc_baseline.x0 = x_baseline

#     u_init_baseline = np.full((n_inputs,1), 0.0)
#     mpc_baseline.u0 = u_init_baseline
#     simulator_baseline.u0 = u_init_baseline
#     mpc_baseline.set_initial_guess()
    
#     u0_baseline = mpc_baseline.make_step(x_baseline)
#     x_baseline_next = simulator_baseline.make_step(u0_baseline)
#     # print(mpc_baseline.data._lam_g_num)
#     return u0_baseline, x_baseline_next, mpc_baseline.data._lam_g_num
    

# def run_sim():
    
#     n_agents = 1
#     n_states = 6
#     n_inputs = 3

#     theta_max = np.pi/6
#     phi_max = np.pi/6
#     tau_max = 5
#     v_max = 5
    
#     Q = np.eye(n_states)*10
#     Qf = np.eye(n_states)*1e3
#     R = np.eye(n_inputs)
#     x_dims = [3]#dummy variable since 1 drone set-up does not use it
#     n_dims = [3]#dummy variable since 1 drone set-up does not use it
#     episode= 200
#     x_baseline_init, x_baseline_f = util.paper_setup_1_quad()

#     x_baseline1 = x_baseline_init #concatenated states of all agents

#     states_list = np.zeros((episode+1,6*n_agents)) #positions of each drone
#     states_list[0] = x_baseline1.flatten()

#     time_start = time.perf_counter()
    
#     results = []

#     with concurrent.futures.ProcessPoolExecutor(100) as executor:

#         for k in range(episode):
#             f = executor.submit(setup_baseline,x_baseline1, x_baseline_f, v_max, theta_max, phi_max, tau_max,\
#                         x_dims,  Q, R, Qf, n_agents, n_inputs, n_dims)
          
#             results.append(f)

  
#             #update positions of each drone:
#             states_list[k+1] = f.result()[1][:].flatten() #collecting all the states of all agents
            
#             # print("Lagrange Multiplier: ", la_mul)
#             # print("Length of Lagrange Multiplier: ", len(la_mul[0]))
#         # ---------------------------------------------------------
#             #position update:
#             x_baseline1  = states_list[k+1].reshape(-1,1) 
#             #velocity update via finite-diff:
 
        
#     time_finish = time.perf_counter()
#     print("Total time: ", time_finish - time_start)
#     np.save('drone_sim_data', states_list)


# if __name__ == '__main__':
#     run_sim()




#**********************************************
#Below is centralized set-up for 3 drones:

def setup_baseline(x_baseline, x_baseline_f, v_max, theta_max, phi_max, tau_max,\
                    x_dims, Q, R, Qf, n_agents, n_inputs, n_dims):
    model_baseline = baseline_drone_model(x_baseline_f, Q, R, Qf, x_baseline, x_dims)  

    
    mpc_baseline = baseline_drone_mpc(model_baseline,n_agents,x_baseline,\
                                      x_dims, v_max, theta_max, phi_max,tau_max)

    
    simulator_baseline = baseline_drone_simulator(model_baseline) 

        
    simulator_baseline.x0['x'] = x_baseline
    mpc_baseline.x0 = x_baseline
    
    # u_init_baseline = np.full((n_inputs,1), 0)
    # u_init_baseline = np.random.rand(n_inputs,1)*0.01
    
    u_init_baseline = np.array([0,0,9.81,0,0,9.81,0,0,9.81]) #hover condition
    mpc_baseline.u0 = u_init_baseline
    simulator_baseline.u0 = u_init_baseline
    mpc_baseline.set_initial_guess()
    
    u0_baseline = mpc_baseline.make_step(x_baseline)
    x_baseline_next = simulator_baseline.make_step(u0_baseline)
    # print(mpc_baseline.data._lam_g_num)
    return u0_baseline, x_baseline_next, mpc_baseline.data._lam_g_num
    

def run_sim():
    
    n_agents = 3
    n_states = 18
    n_inputs = 9

    theta_max = np.pi/6
    phi_max = np.pi/6
    tau_max = 15
    v_max = 10
    
    # Q = np.eye(n_states)
    Q = np.diag([1,1,1,5,5,5,1,1,1,5,5,5,1,1,1,5,5,5])
    Qf = np.eye(n_states)*1e3
    # Qf = np.eye(n_states)
    R = np.diag([0, 1, 1, 0, 1, 1, 0, 1, 1])

    n_dims = [3,3,3]
    x_dims = [6,6,6]

    episode= 50
    x_baseline_init, x_baseline_f = util.paper_setup_3_quads()

    x_baseline1 = x_baseline_init #concatenated states of all agents

    states_list = np.zeros((episode+1,6*n_agents)) #states of each drone
    states_list[0] = x_baseline1.flatten()

    time_start = time.perf_counter()
    
    results = []

    with concurrent.futures.ProcessPoolExecutor(100) as executor:
            
            for k in range(episode):
       
                f = executor.submit(setup_baseline,x_baseline1, x_baseline_f, v_max, theta_max, phi_max, tau_max,\
                            x_dims,  Q, R, Qf, n_agents, n_inputs, n_dims)

                results.append(f)

                #update positions of each drone:
                states_list[k+1] = f.result()[1][:].flatten() #collecting all the states of all agents
            
                # print("Lagrange Multiplier: ", la_mul)
                # print("Length of Lagrange Multiplier: ", len(la_mul[0]))
            # ---------------------------------------------------------
                #position update:
                x_baseline1  = states_list[k+1].reshape(-1,1) 
                    
                if dec.compute_pairwise_distance(x_baseline1,x_dims).all() < 0.1:
                    print("simulation converged to goal!")
                    break
            
        
    time_finish = time.perf_counter()
    print("Total time: ", time_finish - time_start)
    print(f'initial positions of all drones are {x_baseline_init}')
    print(f'final positions of all drones are {x_baseline_f}')
    print(f'sum of final position error of all drones is {np.linalg.norm(x_baseline1-x_baseline_f)} [m]')
    np.save('drone_sim_data', states_list)
    
    

if __name__ == '__main__':
    run_sim()


    
    

#*********************************
#Below is for decentralized set-up
# def setup_baseline(x_baseline, x_baseline_f, v_max, theta_max, phi_max, tau_max,\
#                     x_dims, u_dims, Q, R, Qf, n_agents, n_dims, radius):
#     model_baseline = [baseline_drone_model(dec.split_agents(x_baseline_f.reshape(1,-1),x_dims)[i].flatten(), Q, \
#                      R, Qf, x_baseline, x_dims) for i in range(n_agents)] 
#     #a list of baseline models for each agent
    
#     mpc_baseline = [baseline_drone_mpc(model_i,n_agents,x_baseline, x_dims, v_max, theta_max, phi_max,\
#                    tau_max) for model_i in model_baseline]
#     #a list of baseline mpc controllers for each agent
    
#     simulator_baseline = [baseline_drone_simulator(model_i) for model_i in model_baseline]
    
#     #splitting the states of each agent
#     split_states = dec.split_agents(x_baseline.reshape(1,-1),x_dims)

#     for m in range(len(simulator_baseline)):
        
#         simulator_baseline[m].x0['x'] = split_states[m].T #dimension mismatch here?
#         mpc_baseline[m].x0 = split_states[m].T

#     u_init_baseline = np.full((n_agents*n_inputs,1), 0.0)
#     split_inputs = dec.split_agents(u_init_baseline.reshape(1,-1),u_dims)
    
#     u0_baseline = []
#     x_baseline_next = []
#     for m in range(len(mpc_baseline)):
#         mpc_baseline[m].u0 = split_inputs[m].T
#         simulator_baseline[m].u0 = split_inputs[m].T
#         mpc_baseline[m].set_initial_guess()
#         u0_baseline.append(mpc_baseline[m].make_step(split_states[m].T))
#         x_baseline_next.append(simulator_baseline[m].make_step(split_inputs[m].T))

#     return u0_baseline, x_baseline_next, [mpc_baseline[i].data._lam_g_num for i in range(len(mpc_baseline))]
#     #last _lam_g_num is the Lagrange multiplier
    

# def run_sim():
    
#     theta_max = np.pi/6
#     phi_max = np.pi/6
#     tau_max = 5
#     v_max = 5

#     x_dims = [n_states]*n_agents
#     u_dims = [n_inputs]*n_agents

#     Q = np.eye(n_states)*10
#     Qf = np.eye(n_states)*1e3
#     R = np.eye(n_inputs)

#     radius = 0.5
#     n_dims = [3,3,3]
#     episode= 10
#     x_baseline_init, x_baseline_f = util.paper_setup_3_quads()

#     x_baseline1 = x_baseline_init #concatenated states of all agents

#     states_list = np.zeros((episode+1,3*n_agents)) #positions of each drone
#     states_list[0,:] = np.array([x_baseline1[0],x_baseline1[1],x_baseline1[2],\
#                         x_baseline1[6],x_baseline1[7],x_baseline1[8],\
#                         x_baseline1[12],x_baseline1[13],x_baseline1[14]]).flatten()
#     """
#     TODO: Velocity updates for each drone
#     """
    
#     #Initialize position vectors

#     pos_prev = np.array([x_baseline1[0], x_baseline1[1], x_baseline1[2],\
#                         x_baseline1[6], x_baseline1[7], x_baseline1[8], \
#                         x_baseline1[12], x_baseline1[13], x_baseline1[14]])
    
#     #Initialize velcoity vectors

#     velocity_prev = np.zeros((n_agents*3,1))
    
#     time_start = time.perf_counter()
    
#     results = []

#     with concurrent.futures.ProcessPoolExecutor(100) as executor:

#         for k in range(episode):
#             f = executor.submit(setup_baseline,x_baseline1, x_baseline_f, v_max, theta_max, phi_max, tau_max,\
#                         x_dims, u_dims, Q, R, Qf, n_agents, n_dims, radius)
          
#             results.append(f)

#             for m in range(n_agents):
#                 #update positions of each drone:
#                 states_list[k+1,m*3:(m+1)*3] = f.result()[1][m][0:3].flatten() 
#                 #velocity components will be estimated through finite-difference approx.

#             # print("Lagrange Multiplier: ", la_mul)
#             # print("Length of Lagrange Multiplier: ", len(la_mul[0]))
#         # ------------------------------------------------------------
               
#             for j in range(n_agents):
#                 #position update:
#                 x_baseline1[(j+1-1)*3+j*3:(j+1)*3+j*3]  = states_list[k+1,j*3:(j+1)*3].reshape(-1,1) 
#                 #velocity update via finite-diff:
#                 x_baseline1[(j+1)*3+j*3:(j+2)*3+j*3]  = -(pos_prev[j*3:(j+1)*3] - states_list[k+1,j*3:(j+1)*3].reshape(-1,1) ) / 0.1
            
#             for n in range(n_agents):
#                 pos_prev[n*3:(n+1)*3] = x_baseline1[(n+1-1)*3+n*3:(n+1)*3+n*3]
        
        
#     time_finish = time.perf_counter()
#     print("Total time: ", time_finish - time_start)
#     np.save('drone_sim_data', states_list)

# if __name__ == '__main__':
#     run_sim()