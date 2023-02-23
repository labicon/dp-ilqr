#!/usr/bin/env python
import rclpy
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

# Allow a little extra time for the quads to arrive at their goals since
# GOTO_DURATION isn't guaranteed.
N_MAX_SLEEPS = 1

# Defining takeoff and experiment start position
# start_pos_list = [[0.5, 1.0, 1.0], [3.0, 2.5, 1.0], [1.5, 1.0, 1.0]]
# goal_pos_list = [[2.5, 1.5, 1.0], [0.5, 1.5, 1.0], [1.5, 2.2, 1.0]]

start_pos_list = [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]]
goal_pos_list = [[3.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]


def go_home_callback(swarm, timeHelper, start_pos_list):
    """Tell all quadcopters to go to their starting positions when program exits"""
    print("Program exit: telling quads to go home...")
    goToAbsolute(swarm.allcfs, start_pos_list, yaw=0.0, duration=GOTO_DURATION)
    timeHelper.sleep(5.0)
    swarm.allcfs.land(targetHeight=0.05, duration=GOTO_DURATION)
    timeHelper.sleep(1.0)


def goToAbsolute(cfs, goals, yaw=0.0, duration=2.0):
    """Send goTo's to all crazyflies at once"""
    for pos, cf in zip(goals, cfs.crazyflies):
        # cf.goTo(pos, yaw, duration)
        cf.cmdPosition(pos, yaw)

def goToBlock(cfs, goals, timeHelper, *args, **kwargs):
    # Define the distance at which we have 'arrived'.
    D_ARRIVE = 0.1
    # Recommended rate for using cmdPosition for streaming setpoints.
    CMD_HZ = 10.0
    xf = goals.ravel()
    xi = cfs.position
    while np.linalg.norm(xi, xf) >= D_ARRIVE:
        goToAbsolute(cfs, goals, *args, **kwargs)
        timeHelper.sleepForRate(1 / CMD_HZ)
        xi = cfs.position


def vicon_measurement(swarm):
    """Position update from VICON."""
    pos_cfs = swarm.allcfs.position
    vel_cfs = np.zeros_like(pos_cfs)
    return np.c_[pos_cfs, vel_cfs].ravel()


def perform_experiment(centralized=False, sim=False):

    fig = plt.figure(figsize=(12, 4), layout="constrained")
    
    if not sim:
        # Wait for button press for take off
        input("##### Press Enter to Take Off #####")
    
        swarm.allcfs.takeoff(targetHeight=TAKEOFF_Z, duration=1.0+TAKEOFF_Z)
        timeHelper.sleep(TAKEOFF_DURATION)
        swarm.allcfs.goToAbsolute(start_pos_list)
    
        # Wait for button press to begin experiment
        input("##### Press Enter to Begin Experiment #####")

    n_states = 6
    n_controls = 3
    n_agents = 4
    x_dims = [n_states] * n_agents
    # u_dims = [n_controls] * n_agents
    n_dims = [3] * n_agents

    # x = np.hstack([start_pos_list,np.zeros((n_agents,3))]).flatten() 
    x_goal = np.hstack([goal_pos_list,np.zeros((n_agents,3))]).flatten()

    dt = 0.1
    N = 40

    ids = [100 + i for i in range(n_agents)]
    model = dec.DoubleIntDynamics6D
    dynamics = dec.MultiDynamicalModel([model(dt, id_) for id_ in ids])
    # Q = np.eye(n_states)
    Q = 1.0 * np.diag([10, 10, 10, 1, 1, 1])
    Qf = 1000.0 * np.eye(n_states)
    R = np.eye(3)

    d_converge = 0.2
    d_prox = 0.6
    
    goal_costs = [dec.ReferenceCost(x_goal_i, Q.copy(), R.copy(), Qf.copy(), id_) 
                for x_goal_i, id_ in zip(split_agents(x_goal.T, x_dims), ids)]
    prox_cost = dec.ProximityCost(x_dims, d_prox, n_dims)
    game_cost = dec.GameCost(goal_costs, prox_cost)

    prob = dec.ilqrProblem(dynamics, game_cost)
    centralized_solver = dec.ilqrSolver(prob, N)

    U = np.zeros((N, n_controls*n_agents))
    ids = prob.ids.copy()

    step_size = 20
    
    X_alg = np.zeros((0, n_states*n_agents))
    U_alg = np.zeros((0, n_controls*n_agents))
    X_meas = np.zeros_like(X_alg)
    
    t_kill = N * dt
    X_meas = vicon_measurement(swarm).reshape(1, -1)
    
    while not np.all(dec.distance_to_goal(X_meas[-1], x_goal, n_agents, n_states, 3) <= d_converge):
        # How to feed state back into decentralization?
        #  1. Only decentralize at the current state.
        #  2. Offset the predicted trajectory by the current state.
        # Go with 2. and monitor the difference between the algorithm and VICON.
        X_meas = np.r_[X_meas, vicon_measurement(swarm)[np.newaxis]]
        xi = X_meas[-1]

        t0 = pc()
        if centralized:
            X, U, J, _ = dec.solve_centralized(
                centralized_solver, xi, U, ids, verbose=False, t_kill=t_kill,
            )
        else:
            X, U, J, _ = dec.solve_distributed(
                prob, xi[np.newaxis], U, d_prox, pool=None, verbose=False, t_kill=t_kill,
                )
   
        tf = pc()
        print(f"Solve time: {tf-t0}")
        
        # Record which steps were taken for plotting.
        X_alg = np.r_[X_alg, X[np.newaxis, step_size]]
        U_alg = np.r_[U_alg, U[np.newaxis, step_size]]

        # state_error = np.abs(X[0] - xi)
        # print(f"CF states: \n{xi.reshape(n_agents, n_states)}\n")
        # print(f"Predicted state error: {state_error}")

        # Seed the next iteration with the last state.
        # X = np.r_[X[step_size:], np.tile(X[-1], (step_size, 1))]
        # U = np.r_[U[step_size:], np.tile(U[-1], (step_size, 1))]
    
        # x, y, z coordinates from the solved trajectory X.
        xd = X[step_size].reshape(n_agents, n_states)[:, :3]
        if not sim:
            goToAbsolute(swarm.allcfs, xd, duration=GOTO_DURATION)
            # Position update from VICON.
            xi = vicon_measurement(swarm)

            for _ in range(N_MAX_SLEEPS):
                if np.allclose(xi[dec.pos_mask(x_dims,3)], xd.flatten(), atol=d_converge):
                    break
                # print(f'current state mismatch is {(xi[dec.pos_mask(x_dims,3)] - xd.flatten())}')
                xi = vicon_measurement(swarm)
                timeHelper.sleep(0.01)
            X_meas = np.r_[X_meas, xi[np.newaxis]]
                
        else:
            xi = X[step_size]
        
        # Ensure the algorithm runs and updates are sent at a consistent
        # interval >= max solve time.
        timeHelper.sleepForRate(1/GOTO_DURATION)

        fig.clf()
        ax = fig.add_subplot(1, 3, 1, projection="3d")
        plot_solve(X_alg, J, x_goal, x_dims, n_d=3, ax=ax)
        ax.set_title("Path Taken")
        ax.set_zlim(0, 2)

        ax = fig.add_subplot(1, 3, 2, projection="3d")
        plot_solve(X, J, x_goal, x_dims, n_d=3, ax=ax)
        ax.set_title("Path Planned")
        ax.set_zlim(0, 2)

        ax = fig.add_subplot(1, 3, 3, projection="3d")
        plot_solve(X_meas, J, x_goal, x_dims, n_d=3, ax=ax)
        ax.set_title("Measured Path")
        ax.set_zlim(0, 2)

        fig.canvas.draw()
        plt.pause(0.01)

        # # Replace the currently predicted states with the actual ones.
        # X[0, pos_mask(x_dims, 3)] = xi[pos_mask(x_dims, 3)]
        # # TODO: see if this velocity makes sense here.
        # X[0, ~pos_mask(x_dims, 3)] = xi[~pos_mask(x_dims, 3)]
        # X = np.tile(xi, (N+1,1))

        if LOG_DATA:
            timestampString = str(time.time())
            csvwriter.writerow([timestampString] + xi)

        print(f"Distance left: {dec.distance_to_goal(xi,x_goal,n_agents,n_states,3)}\n{d_converge}")
        # rate.sleep()
    
    if not sim:
        input("##### Press Enter to Go Back to Origin #####")
        goToAbsolute(swarm.allcfs, start_pos_list, duration=GOTO_DURATION)
        timeHelper.sleep(4.0)

        swarm.allcfs.land(targetHeight=0.05, duration=GOTO_DURATION)
        timeHelper.sleep(4.0)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--centralized", action="store_true", default=False)
    parser.add_argument("-s", "--sim", action="store_true", default=False)
    args = parser.parse_args()

    swarm = crazy.Crazyswarm()
    # rate = rospy.Rate(2) this does not exist in ROS2
   
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
    