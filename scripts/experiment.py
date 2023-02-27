#!/usr/bin/env python
import argparse
import atexit
import datetime
from pathlib import Path
import signal
import sys
from time import perf_counter as pc, strftime

import matplotlib.pyplot as plt
import numpy as np
import rclpy

from crazyflie_py.crazyflie import CrazyflieServer, TimeHelper
import dpilqr as dec
from dpilqr import plot_solve, split_agents

plt.ion()

TAKEOFF_Z = 1.0
TAKEOFF_DURATION = 3.0

# Location of repository.
repopath = Path("/home/iconlab/Documents/zjw4/cswarm2/src/crazyswarm2/")

# Defining takeoff and experiment start position
# ==============================================
# --- 3 DRONE SCENARIO ---
# start_pos_list = [[0.5, 1.0, 1.0], [3.0, 2.5, 1.0], [1.5, 1.0, 1.0]]
# goal_pos_list = [[2.5, 1.5, 1.0], [0.5, 1.5, 1.0], [1.5, 2.2, 1.0]]

# --- 4 DRONE EXCHANGE ---
# Trial1:
start_pos_list = [[-1.0, 0.0, 0.9], [0.0, 0.0, 1.3], [1.0, 0.0, 0.8], [2.0, 0.0, 1.0]]
goal_pos_list = [[2.0, 1.5, 1.4], [-1.0, 1.0, 1.0], [0.0, 1.0, 1.1], [1.0, 1.0, 1.0]]

# Trial2:
# start_pos_list = [[-2.1, 0.15, 0.9], [-1.0, 0.1, 1.3], [0.2, 0.0, 0.8], [1.35, 0.0, 1.0]]
# goal_pos_list = [[1.0, 1.5, 1.4], [-2.0, 1.0, 1.0], [-1.0, 1.0, 1.1], [0.0, 1.0, 1.0]]

# Trial3:
# start_pos_list = [[-1.85, 0.5, 0.9], [-1.4, -0.1, 1.3], [0.8, 0.0, 0.8], [1.5, 0.0, 1.0]]
# goal_pos_list = [[1.0, 1.5, 1.4], [-2.0, 1.0, 1.0], [-1.0, 1.0, 1.1], [0.0, 1.0, 1.0]]

# --- 5 DRONE SCENARIO ---
# start_pos_list = [[0.0, -1.0, 0.95],[0.0, 0.0, 1.0] ,[-1.5, 0.0, 0.95] ,[0.7, 0.7, 1.05], [1.5, 0.3, 1.0]]
# goal_pos_list = [[-1.4, 0.0, 1.1], [-1.0, -1.0, 1.0], [0.0, -1.0, 1.0], [1.5, 0.4, 1.0], [1.0, 1.0, 1.0]]


class CrazyflieServerCustom(CrazyflieServer):
    """Adds a few additional useful functions to CrazyflieServer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.setParam("colAv/enable", 1)

    def goToAbsolute(self, goals, yaw=0.0, duration=2.0):
        """Send goTo's to all crazyflies at once
        
        NOTE: cmdPosition usually results in strange behavior and shouldn't be 
        trusted. There aren't much docs on how to go back and forth between
        high <---> low level motion commands either.
        
        """
        for pos, cf in zip(goals, self.crazyflies):
            cf.goTo(pos, yaw, duration)
            # cf.cmdPosition(pos, yaw)

    def goToBlock(self, goals, time_helper, *args, **kwargs):
        """goToAbsolute, but block until we get there
        
        NOTE: this doesn't seem to work all the time. While the docs talk about
        using this as a streaming setpoint, sometimes the crazyflies just don't
        move after continuously setting this parameter for some reason.
        """
        # Define the distance at which we have 'arrived'.
        D_ARRIVE = 0.1
        # Recommended rate for using cmdPosition for streaming setpoints.
        CMD_HZ = 10.0

        xd = np.array(goals)
        while np.any(np.linalg.norm(self.position.reshape(-1, 3) - xd, axis=1) >= D_ARRIVE):
            print("Block error: ", np.linalg.norm(self.position - xd, axis=1))
            self.goToAbsolute(goals, *args, **kwargs)
            time_helper.sleepForRate(CMD_HZ)


class ExperimentRunner:

    # The distance at which we've defined arrival and/or convergence.
    D_CONVERGE = 0.2
    
    # The proximity radius in penalizing for collisions.
    D_PROX = 0.6

    # Used to tune aggresiveness of high-level controller.
    GOTO_DURATION = 4.5

    # Allow a little extra time for the quads to arrive at their goals since
    # self.GOTO_DURATION isn't guaranteed.
    N_MAX_SLEEPS = 1

    # This needs to be tuned to allow the waypoints to be sufficiently ahead of
    # the current position to keep things moving.
    STEP_SIZE = 5

    N_RANGE = (10, 80)
    b = 5.8

    def __init__(self, cf_server, time_helper, dim_info, *args, **kwargs):
        self.server = cf_server
        
        self.time_helper = time_helper

        self.n_states, self.n_controls, self.n_agents, self.n_d = dim_info
        self.fig = plt.figure(figsize=(12, 4), layout="constrained")

        self.x_dims = [n_states] * n_agents
        self.u_dims = [n_controls] * n_agents
        self.n_dims = [n_d] * n_agents

        # Assemble filename for logged data
        timestamp = strftime('%Y-%m-%d_%H.%M.%S')
        results_path = repopath / "crazyflie" / "scripts" / "experiment-data"
        if not results_path.is_dir():
            results_path.mkdir()
        self.output_fname = str(results_path / f"{timestamp}-results.npz")

        self._setup_problem(*args, **kwargs)

    @property
    def t_kill(self):
        return self.N * self.dt

    @property
    def GOTO_RATE(self):
        return 1 / self.GOTO_DURATION
    
    def has_arrived(self, xi):
        return np.all(self.distance_remaining(xi) <= self.D_CONVERGE)
    
    def distance_remaining(self, xi):
        return dec.distance_to_goal(xi, self.x_goal, self.n_agents, self.n_states, self.n_d)

    def _setup_problem(self, dt=0.1, N=40, d_prox=0.6):
        """Setup necessary dp-ilqr bits to run this experiment"""

        self.dt = dt
        self.N = N
        self.d_prox = d_prox

        # x = np.hstack([start_pos_list,np.zeros((n_agents,3))]).flatten() 
        self.x_goal = np.hstack([goal_pos_list, np.zeros((n_agents,3))]).flatten()

        ids = [100 + i for i in range(n_agents)]
        model = dec.DoubleIntDynamics6D
        dynamics = dec.MultiDynamicalModel([model(dt, id_) for id_ in ids])

        # Q = np.eye(n_states)
        Q = 1.0 * np.diag([10, 10, 10, 1, 1, 1])
        Qf = 1000.0 * np.eye(n_states)
        R = np.eye(3)

        goal_costs = [dec.ReferenceCost(x_goal_i, Q.copy(), R.copy(), Qf.copy(), id_) 
                    for x_goal_i, id_ in zip(split_agents(self.x_goal.T, self.x_dims), ids)]
        prox_cost = dec.ProximityCost(self.x_dims, d_prox, self.n_dims)
        game_cost = dec.GameCost(goal_costs, prox_cost)

        self.prob = dec.ilqrProblem(dynamics, game_cost)
        self.centralized_solver = dec.ilqrSolver(self.prob, N)

    def run(self, centralized=False, sim=False):

        if not sim:
            # Wait for button press for take off
            input("##### Press Enter to Take Off #####")

            self.server.takeoff(targetHeight=TAKEOFF_Z, duration=1.0+TAKEOFF_Z)
            self.time_helper.sleep(TAKEOFF_DURATION)
            self.server.goToAbsolute(start_pos_list)
            # self.server.goToBlock(start_pos_list, self.time_helper)

            # Wait for button press to begin experiment
            input("##### Press Enter to Begin Experiment #####")

        U = np.zeros((self.N, n_controls*n_agents))
        ids = self.prob.ids.copy()
        
        X_alg = np.zeros((0, n_states*n_agents))
        U_alg = np.zeros((0, n_controls*n_agents))
        X_meas = self._vicon_measurement()[np.newaxis]
        d_init_max = self.distance_remaining(X_meas[0]).max()
        
        while not self.has_arrived(X_meas[-1]):
            X_meas = np.r_[X_meas, self._vicon_measurement()[np.newaxis]]
            xi = X_meas[-1]

            t0 = pc()
            if centralized:
                X, U, J, _ = dec.solve_centralized(
                    self.centralized_solver, xi, U, ids, verbose=False, t_kill=self.t_kill,
                )
            else:
                X, U, J, _ = dec.solve_distributed(
                    self.prob, xi[np.newaxis], U, self.d_prox, pool=None, verbose=False, t_kill=self.t_kill,
                    )
    
            tf = pc()
            print(f"Solve time: {tf-t0}")
            
            # Record which steps were taken for plotting.
            X_alg = np.r_[X_alg, X[np.newaxis, self.STEP_SIZE]]
            U_alg = np.r_[U_alg, U[np.newaxis, self.STEP_SIZE]]

            # x, y, z coordinates from the solved trajectory X.
            xd = X[self.STEP_SIZE].reshape(n_agents, n_states)[:, :self.n_d]
            if not sim:
                self.server.goToAbsolute(xd, duration=self.GOTO_DURATION)
                # self.server.goToBlock(xd, self.time_helper, duration=self.GOTO_DURATION)
                xi = self._vicon_measurement()

                for _ in range(self.N_MAX_SLEEPS):
                    if np.allclose(
                        xi[dec.pos_mask(self.x_dims, self.n_d)], xd.flatten(), 
                        atol=self.D_CONVERGE
                    ):
                        break

                    xi = self._vicon_measurement()
                    self.time_helper.sleep(0.01)
                X_meas = np.r_[X_meas, xi[np.newaxis]]
                    
            else:
                xi = X[self.STEP_SIZE]
            
            # Ensure the algorithm runs and updates are sent at a consistent
            # interval >= max solve time.
            self.time_helper.sleepForRate(self.GOTO_RATE)

            # Try to understand how far we've come.
            self._visualize_progress(X, X_alg, X_meas, J)

            d_left = self.distance_remaining(xi)
            print(f"Distance left: {d_left}")

            # Tune the aggressiveness based on proximity to the goal.
            m = (self.N_RANGE[1] - self.N_RANGE[0]) / (d_init_max - self.D_CONVERGE)
            N_lin = int(self.b +  m * d_left.max())
            N_i = min(max(N_lin, self.N_RANGE[0]), self.N_RANGE[1])
            print("Current horizon: ", N_i)

            # Update (overwrite) current results.
            self._save_results(X_alg, X_meas, U_alg)

        if not sim:
            input("##### Press Enter to Go Back to Origin #####")
            go_home_callback(self.server, self.time_helper, start_pos_list)

    def _vicon_measurement(self):
        """Position update from VICON."""
        pos_cfs = self.server.position
        vel_cfs = np.zeros_like(pos_cfs)
        return np.c_[pos_cfs, vel_cfs].ravel()
    
    def _visualize_progress(self, X, X_alg, X_meas, J):
        self.fig.clf()
        ax = self.fig.add_subplot(1, 3, 1, projection="3d")
        plot_solve(X_alg, J, self.x_goal, self.x_dims, n_d=self.n_d, ax=ax)
        ax.set_title("Path Taken")
        ax.set_zlim(0, 2)

        ax = self.fig.add_subplot(1, 3, 2, projection="3d")
        plot_solve(X, J, self.x_goal, self.x_dims, n_d=self.n_d, ax=ax)
        ax.set_title("Path Planned")
        ax.set_zlim(0, 2)

        ax = self.fig.add_subplot(1, 3, 3, projection="3d")
        plot_solve(X_meas, J, self.x_goal, self.x_dims, n_d=self.n_d, ax=ax)
        ax.set_title("Measured Path")
        ax.set_zlim(0, 2)

        self.fig.canvas.draw()
        plt.pause(0.01)

    def _save_results(self, X_alg, X_meas, U_alg):
        np.savez(self.output_fname, X_alg=X_alg, U_alg=U_alg, X_meas=X_meas)


def go_home_callback(cf_server, timeHelper, start_pos_list):
    """Tell all quadcopters to go to their starting positions when program exits"""
    print("Program exit: telling quads to go home...")
    GO_HOME_DURATION = 5.0
    cf_server.goToAbsolute(start_pos_list, yaw=0.0, duration=GO_HOME_DURATION)
    timeHelper.sleep(GO_HOME_DURATION)
    cf_server.land(targetHeight=0.05, duration=GO_HOME_DURATION)
    timeHelper.sleep(1.0)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--centralized", action="store_true", default=False)
    parser.add_argument("-s", "--sim", action="store_true", default=False)
    args = parser.parse_args()

    n_states = 6
    n_controls = 3
    n_agents = 4
    # n_agents = 5
    n_d = 3
    dim_info = (n_states, n_controls, n_agents, n_d)

    dt = 0.1
    N = 20

    # cf_server = crazy.Crazyswarm().allcfs
    rclpy.init()
    cf_server = CrazyflieServerCustom()
    time_helper = TimeHelper(cf_server)
    runner = ExperimentRunner(cf_server, time_helper, dim_info, dt=dt, N=N)

    # Exit on CTRL+C. 
    signal.signal(signal.SIGINT, lambda *_: sys.exit(-1))

    # Tell the quads to go home when we're done.
    if not args.sim:
        atexit.register(go_home_callback, cf_server, time_helper, start_pos_list)

    # Fingers crossed...
    runner.run(args.centralized, args.sim)
