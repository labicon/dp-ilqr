#!/usr/bin/env python

"""Collection of examples that demonstrate the functionality of
distributed-potential-ilqr in various scenarios.

Including:
 - single unicycle
 - single quadcopter (6D)
 - two quads with one human
 - random multi-agent simulation

"""


import numpy as np
import matplotlib.pyplot as plt

from dpilqr import split_agents, plot_solve
import dpilqr
import scenarios

π = np.pi
g = 9.80665


def single_unicycle():
    dt = 0.05
    N = 50

    x = np.array([-10, 10, 10, 0], dtype=float)
    x_goal = np.zeros((4, 1), dtype=float).T

    dynamics = dpilqr.UnicycleDynamics4D(dt)

    Q = np.diag([1.0, 1, 0, 0])
    Qf = 1000 * np.eye(Q.shape[0])
    R = np.eye(2)
    cost = dpilqr.ReferenceCost(x_goal, Q, R, Qf)

    prob = dpilqr.ilqrProblem(dynamics, cost)
    ilqr = dpilqr.ilqrSolver(prob, N)
    X, _, J = ilqr.solve(x)

    plt.clf()
    plot_solve(X, J, x_goal)
    plt.show()


def single_quad6d():
    dt = 0.1
    N = 40
    n_d = 3

    x = np.array([2, 2, 0.5, 0, 0, 0], dtype=float)
    xf = np.zeros((6, 1), dtype=float).T

    dynamics = dpilqr.QuadcopterDynamics6D(dt)

    Q = np.eye(6)
    Qf = 100 * np.eye(Q.shape[0])
    R = np.diag([0, 1, 1])
    cost = dpilqr.ReferenceCost(xf, Q, R, Qf)

    prob = dpilqr.ilqrProblem(dynamics, cost)
    ilqr = dpilqr.ilqrSolver(prob, N)

    X, _, J = ilqr.solve(x)

    plt.clf()
    plot_solve(X, J, xf, n_d=n_d)
    plt.show()


def two_quads_one_human():
    n_agents = 3
    n_states = 6
    n_controls = 3

    x_dims = [n_states] * n_agents
    n_dims = [3, 3, 2]

    dt = 0.05
    N = 50
    radius = 0.3

    x0, xf = scenarios.q2h1_passthrough_setup()

    Q = np.diag([1, 1, 1, 5, 5, 5])
    R = np.diag([1, 1, 1])
    Qf = 1e3 * np.eye(n_states)

    Q_human = np.diag([1, 1, 1, 0, 0, 0])
    R_human = np.diag([1, 1, 1e-9])
    Qf_human = 1e3 * np.eye(Q.shape[0])

    Qs = [Q, Q, Q_human]
    Rs = [R, R, R_human]
    Qfs = [Qf, Qf, Qf_human]

    models = [
        dpilqr.QuadcopterDynamics6D,
        dpilqr.QuadcopterDynamics6D,
        dpilqr.HumanDynamics6D,
    ]
    ids = [100 + i for i in range(n_agents)]
    dynamics = dpilqr.MultiDynamicalModel(
        [model(dt, id_) for id_, model in zip(ids, models)]
    )

    goal_costs = [
        dpilqr.ReferenceCost(xf_i, Qi, Ri, Qfi, id_)
        for xf_i, id_, x_dim, Qi, Ri, Qfi in zip(
            dpilqr.split_agents_gen(xf, x_dims), ids, x_dims, Qs, Rs, Qfs
        )
    ]
    prox_cost = dpilqr.ProximityCost(x_dims, radius, n_dims)
    game_cost = dpilqr.GameCost(goal_costs, prox_cost)

    problem = dpilqr.ilqrProblem(dynamics, game_cost)
    solver = dpilqr.ilqrSolver(problem, N)

    U0 = np.c_[np.tile([g, 0, 0], (N, 2)), np.ones((N, n_controls))]
    X, _, J = solver.solve(x0, U0)

    plt.figure()
    plot_solve(X, J, xf, x_dims, True, 3)

    plt.figure()
    dpilqr.plot_pairwise_distances(X, x_dims, n_dims, radius)

    plt.show()


def random_multiagent_simulation():

    n_states = 4
    n_controls = 2
    n_agents = 7
    x_dims = [n_states] * n_agents
    u_dims = [n_controls] * n_agents
    n_dims = [2] * n_agents

    n_d = n_dims[0]

    x0, xf = dpilqr.random_setup(
        n_agents,
        n_states,
        is_rotation=False,
        rel_dist=2.0,
        var=n_agents / 2,
        n_d=2,
        random=True,
    )

    dpilqr.eyeball_scenario(x0, xf, n_agents, n_states)
    plt.show()

    dt = 0.05
    N = 60

    tol = 1e-6
    ids = [100 + i for i in range(n_agents)]

    model = dpilqr.UnicycleDynamics4D
    dynamics = dpilqr.MultiDynamicalModel([model(dt, id_) for id_ in ids])

    Q = np.eye(4)
    R = np.eye(2)
    Qf = 1e3 * np.eye(n_states)
    radius = 0.5

    goal_costs = [
        dpilqr.ReferenceCost(xf_i, Q.copy(), R.copy(), Qf.copy(), id_)
        for xf_i, id_, x_dim, u_dim in zip(
            dpilqr.split_agents_gen(xf, x_dims), ids, x_dims, u_dims
        )
    ]
    prox_cost = dpilqr.ProximityCost(x_dims, radius, n_dims)
    goal_costs = [
        dpilqr.ReferenceCost(xf_i, Q.copy(), R.copy(), Qf.copy(), id_)
        for xf_i, id_ in zip(split_agents(xf.T, x_dims), ids)
    ]
    prox_cost = dpilqr.ProximityCost(x_dims, radius, n_dims)
    game_cost = dpilqr.GameCost(goal_costs, prox_cost)

    problem = dpilqr.ilqrProblem(dynamics, game_cost)
    solver = dpilqr.ilqrSolver(problem, N)

    X, _, J = solver.solve(x0, tol=tol, t_kill=None)

    plt.clf()
    plot_solve(X, J, xf.T, x_dims, True, n_d)

    plt.figure()
    dpilqr.plot_pairwise_distances(X, x_dims, n_dims, radius)

    plt.show()

    dpilqr.make_trajectory_gif(f"{n_agents}-unicycles.gif", X, xf, x_dims, radius)


def _3d_integrators():
    n_states = 6
    n_controls = 3
    n_agents = 5
    x_dims = [n_states] * n_agents
    u_dims = [n_controls] * n_agents
    n_dims = [3] * n_agents

    n_d = n_dims[0]

    # x0, xf = scenarios.paper_setup_5_quads()
    # x0, xf = scenarios.four_quads_passthrough()
    x0, xf = scenarios.five_quads_figure1()

    dpilqr.eyeball_scenario(x0, xf, n_agents, n_states)
    plt.show()

    dt = 0.05
    N = 60

    tol = 1e-3
    ids = [100 + i for i in range(n_agents)]

    model = dpilqr.DoubleIntDynamics6D
    dynamics = dpilqr.MultiDynamicalModel([model(dt, id_) for id_ in ids])

    Q = np.eye(6)
    R = np.eye(3)
    Qf = 1e3 * np.eye(n_states)
    radius = 0.6

    goal_costs = [
        dpilqr.ReferenceCost(xf_i, Q.copy(), R.copy(), Qf.copy(), id_)
        for xf_i, id_ in zip(dpilqr.split_agents_gen(xf, x_dims), ids)
    ]
    prox_cost = dpilqr.ProximityCost(x_dims, radius, n_dims)
    goal_costs = [
        dpilqr.ReferenceCost(xf_i, Q.copy(), R.copy(), Qf.copy(), id_)
        for xf_i, id_ in zip(split_agents(xf.T, x_dims), ids)
    ]
    prox_cost = dpilqr.ProximityCost(x_dims, radius, n_dims)
    game_cost = dpilqr.GameCost(goal_costs, prox_cost)

    problem = dpilqr.ilqrProblem(dynamics, game_cost)
    solver = dpilqr.ilqrSolver(problem, N)

    X, _, J = solver.solve(x0, tol=tol, t_kill=None)

    plt.clf()
    plot_solve(X, J, xf.T, x_dims, True, n_d)
    plt.gca().set_zlim([0, 2])

    plt.figure()
    dpilqr.plot_pairwise_distances(X, x_dims, n_dims, radius)

    plt.show()

    # dpilqr.make_trajectory_gif(f"{n_agents}-quads.gif", X, xf, x_dims, radius)


def nquads_mhumans():
    
    n = 2
    m = 2
    n_agents = n + m
    n_states = 6
    n_controls = 3

    x_dims = [n_states] * n_agents
    n_dims = [3, 3, 2, 2]
    # n_dims = [3, 3, 3, 3]

    dt = 0.05
    N = 60
    radius = 1.0

    x0, xf = scenarios.q3h2_qcross()
    x0, xf = scenarios.q2h2_hcross()

    Q = np.eye(6)
    R = 0.1*np.eye(3)
    Qf = 1e4 * np.eye(n_states)

    Q_human = Q.copy()
    R_human = R.copy()
    Qf_human = Qf.copy()
    # Q_human = np.diag([1, 1, 0, 0, 0, 0])
    # R_human = np.diag([1, 1, 1])
    # Qf_human = 1e3 * np.eye(Q.shape[0])

    Qs = [Q] * n + [Q_human] * m
    Rs = [R] * n + [R_human] * m
    Qfs = [Qf] * n + [Qf_human] * m

    models = [dpilqr.QuadcopterDynamics6D] * n + [dpilqr.HumanDynamicsLin6D] * m
    # models = [dpilqr.QuadcopterDynamics6D] * n + [dpilqr.QuadcopterDynamics6D] * m
    ids = [100 + i for i in range(n_agents)]
    dynamics = dpilqr.MultiDynamicalModel(
        [model(dt, id_) for id_, model in zip(ids, models)]
    )

    goal_costs = [
        dpilqr.ReferenceCost(xf_i, Qi, Ri, Qfi, id_)
        for xf_i, id_, x_dim, Qi, Ri, Qfi in zip(
            dpilqr.split_agents_gen(xf, x_dims), ids, x_dims, Qs, Rs, Qfs
        )
    ]
    prox_cost = dpilqr.ProximityCost(x_dims, radius, n_dims)
    game_cost = dpilqr.GameCost(goal_costs, prox_cost)

    h_ids = ids[-2:]
    problem = dpilqr.ilqrProblem(dynamics, game_cost)
    solver = dpilqr.ilqrSolver(problem, N)

    # U0 = np.zeros((N, n_controls*n_agents))
    U0 = problem.selfish_warmstart(x0, N)
    # X, _, J = solver.solve(x0, U0)
    X, _, J, solve_info = dpilqr.solve_distributed(problem, x0.T, U0, radius, h_ids, pool=None, verbose=True)
    igraph = {k: v[1] for k, v in solve_info.items()}
    dpilqr.plot_interaction_graph(igraph)

    plt.figure()
    plot_solve(X, J, xf, x_dims, True, 3)
    plt.gca().set_zlim([0, 2])

    plt.figure()
    dpilqr.plot_pairwise_distances(X, x_dims, n_dims, radius)

    plt.show()



def main():

    # single_unicycle()
    # single_quad6d()
    # two_quads_one_human()
    # random_multiagent_simulation()
    # _3d_integrators()
    nquads_mhumans()

if __name__ == "__main__":
    main()
