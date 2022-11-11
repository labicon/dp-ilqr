import numpy as np
from casadi import *
import do_mpc
import decentralized as dec
import util

# 1 drone MPC:

# def baseline_drone_model(xf, Q, R, Qf, x_baseline, x_dims):

#     """
#     Below is the distributed set-up
#     xf: concatenated goal state vector for all agents

#     """
#     model_type = 'continuous' # either 'discrete' or 'continuous'
#     model = do_mpc.model.Model(model_type)

#     #x = p_x,p_y,p_z,v_x,v_y,v_z
#     #u = theta, phi, tau

#     #Concantenated states of all agents
#     x = model.set_variable(var_type='_x', var_name='x', shape=(6, 1))
#     #Concatenated inputs of all agents
#     u = model.set_variable(var_type='_u', var_name='u', shape=(3, 1))

#     # print(f'Shape of x is {x.shape}')
#     # print(f'Shape of u is {u.shape}')

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
#     g = 9.81
#     model.set_rhs('x', vertcat(x[3], x[4], x[5], g*np.tan(u[0]), -g*np.tan(u[1]), u[2]-g))


#     total_stage_cost = (x-xf).T@Q@(x-xf) + u.T@R@u
#     total_terminal_cost = (x-xf).T@Qf@(x-xf)

#     model.set_expression('total_stage_cost',total_stage_cost)
#     model.set_expression('total_terminal_cost',total_terminal_cost)


#     model.setup()

#     return model


# ******************************************************
# Below is the centralized set-up for 3 drones

# def baseline_drone_model_3(xf, Q, R, Qf, x_baseline, x_dims):
#     """
#     Below is the distributed set-up
#     xf: concatenated goal state vector for all agents

#     """
#     model_type = 'continuous'  # either 'discrete' or 'continuous'
#     model = do_mpc.model.Model(model_type)

#     #x = p_x,p_y,p_z,v_x,v_y,v_z
#     #u = theta, phi, tau

#     # Concantenated states of all agents
#     x = model.set_variable(var_type='_x', var_name='x', shape=(18, 1))
#     # Concatenated inputs of all agents
#     u = model.set_variable(var_type='_u', var_name='u', shape=(9, 1))

#     # x = model.set_variable(var_type='_x', var_name='x', shape=(18, 1))
#     # u = model.set_variable(var_type='_u', var_name='u', shape=(18, 1))

#     # model.set_rhs('x', vertcat(u[0], u[1], u[2], u[3], u[4], u[5],\
#     #                           u[6], u[7], u[8], u[9], u[10], u[11],\
#     #                           u[12], u[13], u[14], u[15], u[16], u[17])) #a simpler model

#     # print(f'Shape of x is {x.shape}')
#     # print(f'Shape of u is {u.shape}')

#     # 6-D model of a quadrotor with 6 states and 3 inputs:
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
#     g = 9.81
#     # use tan function from Casadi:
#     eps = 1e-6
#     model.set_rhs('x', vertcat(x[3], x[4], x[5], g*tan(u[0]+eps), -g*tan(u[1]+eps), u[2]-g,
#                                x[9], x[10], x[11], g *tan(u[3]+eps), -g*tan(u[4]+eps), u[5]-g,
#                                x[15], x[16], x[17], g*tan(u[6]+eps), -g*tan(u[7]+eps), u[8]-g))

#     u_ref = np.array([0, 0, g+eps, 0, 0, g+eps, 0, 0, g+eps])

#     total_stage_cost = 0
#     for i in range(x.shape[0]):
#         total_stage_cost += (x[i]-xf[i])*Q[i, i]*(x[i]-xf[i])

#     for j in range(u.shape[0]):
#         total_stage_cost += (u[j]-u_ref[j])*R[j, j]*(u[j]-u_ref[j])

#     total_terminal_cost = 0
#     for m in range(x.shape[0]):
#         total_terminal_cost += (x[m]-xf[m])*Qf[m, m]*(x[m]-xf[m])

#     model.set_expression('total_stage_cost', total_stage_cost)
#     model.set_expression('total_terminal_cost', total_terminal_cost)

#     # x_baseline is concatenated states of all agents
#     # distances = SX(util.compute_pairwise_distance_Sym(x_baseline,x_dims,n_d=3))

#     distance_1 = sqrt((x[0]-x[6])**2+(x[1]-x[7])**2+(x[2]-x[8])**2)
#     distance_2 = sqrt((x[0]-x[12])**2+(x[1]-x[13])**2+(x[2]-x[14])**2)
#     distance_3 = sqrt((x[6]-x[12])**2+(x[7]-x[13])**2+(x[8]-x[14])**2)

#     model.set_expression('collision_avoidance1', distance_1)
#     model.set_expression('collision_avoidance2', distance_2)
#     model.set_expression('collision_avoidance3', distance_3)

#     model.setup()

#     return model


# ****************************************************
# Below is the centralized set-up for 4 drones:
def baseline_drone_model_4(xf, Q, R, Qf, x_baseline, x_dims):
    """
    Below is the distributed set-up
    xf: concatenated goal state vector for all agents

    """
    
    model_type = 'continuous'  # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    #x = p_x,p_y,p_z,v_x,v_y,v_z
    #u = theta, phi, tau

    # Concantenated states of all agents
    x = model.set_variable(var_type='_x', var_name='x', shape=(24, 1))
    # Concatenated inputs of all agents
    u = model.set_variable(var_type='_u', var_name='u', shape=(12, 1))

    # 6-D model of a quadrotor with 6 states and 3 inputs:
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
    # use tan function from Casadi:
    eps = 1e-6
    model.set_rhs('x', vertcat(x[3], x[4], x[5], g*tan(u[0]+eps), -g*tan(u[1]+eps), u[2]-g,
                               x[9], x[10], x[11], g *tan(u[3]+eps), -g*tan(u[4]+eps), u[5]-g,
                               x[15], x[16], x[17], g *tan(u[6]+eps), -g*tan(u[7]+eps), u[8]-g,
                               x[18], x[19], x[20], g*tan(u[9]+eps), -g*tan(u[10]+eps), u[11]-g))

    u_ref = np.array([0, 0, g+eps,\
                     0, 0, g+eps, \
                     0, 0, g+eps, \
                    0, 0, g+eps])

    total_stage_cost = 0
    for i in range(x.shape[0]):
        total_stage_cost += (x[i]-xf[i])*Q[i, i]*(x[i]-xf[i])

    for j in range(u.shape[0]):
        total_stage_cost += (u[j]-u_ref[j])*R[j, j]*(u[j]-u_ref[j])

    total_terminal_cost = 0
    for m in range(x.shape[0]):
        total_terminal_cost += (x[m]-xf[m])*Qf[m, m]*(x[m]-xf[m])

    model.set_expression('total_stage_cost', total_stage_cost)
    model.set_expression('total_terminal_cost', total_terminal_cost)

    distances = util.compute_pairwise_distance_Sym(x, x_dims)
    
    for j in range(len(distances)):

        model.set_expression(f'collision_avoidance{j}', distances[j])   

    model.setup()

    return model


# ******************************************************
# Below is the decentralized set-up for multiple drones
# def baseline_drone_model_decentralized(xf, Q, R, Qf, x_baseline, x_dims,n_agents,n_states,n_inputs):
#     """
#     Below is the distributed set-up

#     """

#     model_type = 'continuous' # either 'discrete' or 'continuous'
    
#     models = []
#     for m in range(n_agents):
#         models.append(do_mpc.model.Model(model_type))
        
#     x = [i.set_variable(var_type='_x', var_name=f'x{j}', shape=(n_states, 1)) for i,j in zip(models,range(len(models)))]
#     u = [j.set_variable(var_type='_u', var_name=f'u{k}', shape=(n_inputs, 1)) for j,k in zip(models,range(len(models)))]
    
#    #6-D model of a quadrotor with 6 states and 3 inputs:
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
#     eps = 1e-6
#     g = 9.81
#     for n,p in zip(models,range(n_agents)):
#         n.set_rhs('x', vertcat(x[p][3], x[p][4], x[p][5], g*tan(u[p][0]+eps), -g*tan(u[p][1]+eps), u[p][2]-g))
        
#     u_ref =np.array([0, 0, g+eps])
                   
    
#     total_stage_cost=[]
#     total_terminal_cost=[]
    
#     for i in range(len(x)):
#         for j in range(x[i].shape[0]):
#             total_stage_cost.append((x[i][j]-xf[j])*Q[j, j]*(x[i][j]-xf[j]))
#             total_terminal_cost.append((x[i][j]-xf[j])*Qf[j, j]*(x[i][j]-xf[j]))
            
#     for i in range(len(u)):
#         for j in range(u[i].shape[0]):
#             total_stage_cost[i] += (u[i][j]-u_ref[j])*R[j, j]*(u[i][j]-u_ref[j])

#     for i in range(len(models)):    
#         models[i].set_expression(f'total_stage_cost{i}', sum(total_stage_cost[i*n_states:(i+1)*n_states]))
#         models[i].set_expression(f'total_terminal_cost{i}',sum(total_terminal_cost[i*n_states:(i+1)*n_states]))
    
        
#     distances = []
#     distances.append(util.compute_pairwise_distance_Sym(x[i], x_dims) for i in range(len(models)))
    
    
#     #setting pairwise distance constraints:
#     for i in range(len(models)):
#         for j in range(len(distances)):

#             models[i].set_expression(f'collision_avoidance{j}', distances[j])   
    
    
#     model.setup()

#     return model
