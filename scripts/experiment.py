#!/usr/bin/env python

import argparse
import atexit
import csv
import datetime
import signal
import sys
import time
from time import perf_counter as pc

import matplotlib.pyplot as plt
import numpy as np

import dpilqr as dec
from dpilqr import plot_solve, split_agents
import crazyflie_py as crazy

plt.ion()

# Assemble filename for logged data
datetimeString = datetime.datetime.now().strftime("%m%d%y-%H:%M:%S")
csv_filename = "experiment_data/" + datetimeString + "-data.csv"

# Enable or disable data logging
LOG_DATA = False

TAKEOFF_Z = 1.0
TAKEOFF_DURATION = 1.0

# Used to tune aggresiveness of low-level controller
GOTO_DURATION = 1.0

# Defining takeoff and experiment start position
start_pos_list = [[0.5, 1.0, 1.0], [3.0, 2.5, 1.0], [1.5, 1.0, 1.0]]
goal_pos_list = [[2.5, 1.5, 1.0], [0.5, 1.5, 1.0], [1.5, 2.2, 1.0]]


def go_home_callback(swarm, timeHelper, start_pos_list):
    """Tell all quadcopters to go to their starting positions when program exits"""
    print("Program exit: telling quads to go home...")
    swarm.allcfs.goToAbsolute(start_pos_list, yaw=0.0, duration=GOTO_DURATION)
    timeHelper.sleep(4.0)
    swarm.allcfs.land(targetHeight=0.05, duration=3.0)
    timeHelper.sleep(4.0)


"""
The states of the quadcopter are: px, py ,pz, vx, vy, vz
"""
def perform_experiment(listener, centralized=False, sim=False):

    fig1 = plt.figure()
    fig2 = plt.figure()
    
    if not sim:
        # Wait for button press for take off
        input("##### Press Enter to Take Off #####")
    
        swarm.allcfs.takeoff(targetHeight=TAKEOFF_Z, duration=1.0+TAKEOFF_Z)
        timeHelper.sleep(TAKEOFF_DURATION)
        swarm.allcfs.goToAbsolute(start_pos_list)
    
        # Wait for button press to begin experiment
        input("##### Press Enter to Begin Experiment #####")

    n_agents = 3
    n_states = 6
    n_controls = 3
    n_dims = [3] * n_agents
    x_dims = [n_states] * n_agents
    # u_dims = [n_controls] * n_agents
    
    x = np.hstack([start_pos_list,np.zeros((n_agents,3))]).flatten() 
    x_goal = np.hstack([goal_pos_list,np.zeros((n_agents,3))]).flatten()

    dt = 0.1
    N = 40

    ids = [100 + i for i in range(n_agents)]
    model = dec.QuadcopterDynamics6D
    dynamics = dec.MultiDynamicalModel([model(dt, id_) for id_ in ids])
    Q = 1.0 * np.diag([10, 10, 10, 1, 1, 1])
    Qf = 1000.0 * np.eye(Q.shape[0])
    R = 1.0 * np.diag([0, 1, 1])

    d_converge = 0.05
    d_prox = 0.2
    
    goal_costs = [dec.ReferenceCost(x_goal_i, Q.copy(), R.copy(), Qf.copy(), id_) 
                for x_goal_i, id_ in zip(split_agents(x_goal.T, x_dims), ids)]
    prox_cost = dec.ProximityCost(x_dims, d_prox, n_dims)
    game_cost = dec.GameCost(goal_costs, prox_cost)

    prob = dec.ilqrProblem(dynamics, game_cost)
    centralized_solver = dec.ilqrSolver(prob, N)
    
    xi = x.reshape(1, -1)
    U = np.zeros((N, n_controls*n_agents))
    ids = prob.ids.copy()

    step_size = 1
    
    X_full = np.zeros((0, n_states*n_agents))
    U_full = np.zeros((0, n_controls*n_agents))
    X = np.tile(xi,(N+1, 1))
    
    t_kill = N*dt
    
    while not np.all(dec.distance_to_goal(xi,x_goal,n_agents,n_states,3) <= d_converge):
        t0 = pc()
        # How to feed state back into decentralization?
        #  1. Only decentralize at the current state.
        #  2. Offset the predicted trajectory by the current state.
        # Go with 2. and monitor the difference between the algorithm and VICON.
        if centralized:
            X, U, J, _ = dec.solve_centralized(
                centralized_solver, xi, U, ids, verbose=False
            )
        else:
            X, U, J, _ = dec.solve_distributed(
                prob, X, U, d_prox, pool=None, verbose=False, t_kill=t_kill
                )
   
        tf = pc()
        print(f"Solve time: {tf-t0}")
        
        # Record which steps were taken for plotting.
        X_full = np.r_[X_full, X[:step_size]]
        U_full = np.r_[U_full, U[:step_size]]

        # Seed the next iteration with the last state.
        X = np.r_[X[step_size:], np.tile(X[-1], (step_size, 1))]
        U = np.r_[U[step_size:], np.zeros((step_size, n_controls*n_agents))]

        # x, y, z coordinates from the solved trajectory X.
        xd = X[step_size].reshape(n_agents, n_states)[:, :3]
        if not sim:
            swarm.allcfs.goToAbsolute(xd, duration=1.5)
            # Position update from VICON
            pos_cfs = swarm.allcfs.position
            print(pos_cfs.shape)
            # Velocity update from VICON
            # vel_cfs = [cf.velocity() for cf in swarm.allcfs.crazyflies]
            vel_cfs = np.zeros_like(pos_cfs)
        else:
            xi = X[step_size]

        state_error = np.abs(X[0] - xi)
        print(f"CF states: \n{xi.reshape(n_agents, n_states)}\n")
        print(f"Predicted state error: {state_error}")

        plt.figure(fig1.number)
        plt.clf()
        plot_solve(X_full, J, x_goal, x_dims, n_d=3)
        plt.title("Path Taken")
        plt.gca().set_zlim(0, 2)

        plt.figure(fig2.number)
        plt.clf()
        plot_solve(X, J, x_goal, x_dims, n_d=3)
        plt.title("Path Planned")
        plt.gca().set_zlim(0, 2)

        fig1.canvas.draw()
        fig2.canvas.draw()
        plt.pause(1)

        # # Replace the currently predicted states with the actual ones.
        # X[0, pos_mask(x_dims, 3)] = xi[pos_mask(x_dims, 3)]
        # # TODO: see if this velocity makes sense here.
        # X[0, ~pos_mask(x_dims, 3)] = xi[~pos_mask(x_dims, 3)]
        X = np.tile(xi, (N+1,1))

        if LOG_DATA:
            timestampString = str(time.time())
            csvwriter.writerow([timestampString] + pos_cfs + vel_cfs)

        # rate.sleep()
    
    if not sim:
        input("##### Press Enter to Go Back to Origin #####")
        swarm.allcfs.goToAbsolute(start_pos_list, duration=GOTO_DURATION*3)
        timeHelper.sleep(4.0)

        swarm.allcfs.land(targetHeight=0.05, duration=GOTO_DURATION)
        timeHelper.sleep(4.0)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--centralized", action="store_true", default=False)
    parser.add_argument("-s", "--sim", action="store_true", default=False)
    args = parser.parse_args()

    swarm = crazy.Crazyswarm()
    # rate = rospy.Rate(2)
   
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # TODO: succeed without collision avoidance.
    # swarm.allcfs.setParam("colAv/enable", 1) 

    # Exit on CTRL+C. 
    signal.signal(signal.SIGINT, lambda *_: sys.exit(-1))

    # Tell the quads to go home when we're done.
    if not args.sim:
        atexit.register(go_home_callback, swarm, timeHelper, start_pos_list)

    if LOG_DATA:
        num_cfs = len(swarm.allcfs.crazyflies)
        print("### Logging data to file: " + csv_filename)
        with open(csv_filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')

            csvwriter.writerow(['# CFs', str(num_cfs)])
            csvwriter.writerow(
                ["Timestamp [s]"] 
                + num_cfs*["x_d", "y_d", "z_d", " x", "y", "z", "qw", "qx", "qy", "qz"]
                )

    perform_experiment(args.centralized, args.sim)
