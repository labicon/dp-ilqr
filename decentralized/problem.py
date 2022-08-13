#!/usr/bin/env python

"""Logic to combine dynamics and cost in one framework to simplify decentralization"""

from time import perf_counter as pc

import numpy as np

from .control import ilqrSolver
from .cost import ReferenceCost, GameCost
from .dynamics import DynamicalModel, MultiDynamicalModel
from .util import split_agents_gen


class ilqrProblem:
    """Centralized optimal control problem that combines dynamics and cost"""

    def __init__(self, dynamics, cost):
        self.dynamics = dynamics
        self.game_cost = cost
        self.n_agents = 1
        
        if isinstance(cost, GameCost):
            self.n_agents = len(cost.ref_costs)

    @property
    def ids(self):
        if not isinstance(self.dynamics, MultiDynamicalModel):
            raise NotImplementedError(
                "Only MultiDynamicalModel's have an 'ids' attribute"
            )
        if not self.dynamics.ids == self.game_cost.ids:
            raise ValueError(f"Dynamics and cost have inconsistent ID's: {self}")
        return self.dynamics.ids.copy()

    def split(self, graph):
        """Split up this centralized problem into a list of decentralized
        sub-problems.
        """

        split_dynamics = self.dynamics.split(graph)
        split_costs = self.game_cost.split(graph)

        return [
            ilqrProblem(dynamics, cost)
            for dynamics, cost in zip(split_dynamics, split_costs)
        ]

    def extract(self, X, U, id_):
        """Extract the state and controls for a particular agent id_ from the
        concatenated problem state/controls
        """

        if id_ not in self.ids:
            raise IndexError(f"Index {id_} not in ids: {self.ids}.")

        # NOTE: Assume uniform dynamical models.
        ext_ind = self.ids.index(id_)
        x_dim = self.game_cost.x_dims[0]
        u_dim = self.game_cost.u_dims[0]
        Xi = X[:, ext_ind * x_dim : (ext_ind + 1) * x_dim]
        Ui = U[:, ext_ind * u_dim : (ext_ind + 1) * u_dim]

        return Xi, Ui

    def selfish_warmstart(self, x0, N):
        """Compute a 'selfish' warmstart by ignoring other agents"""

        print("=" * 80 + "\nComputing warmstart...")
        # Split out the full problem into separate problems for each agent.
        x0 = x0.reshape(1, -1)
        selfish_graph = {id_: [id_] for id_ in self.ids}
        subproblems = self.split(selfish_graph)

        U_warm = np.zeros((N, self.dynamics.n_u))
        t0_all = pc()
        for problem, x0i, id_ in zip(
            subproblems, split_agents_gen(x0, self.game_cost.x_dims), self.ids
        ):
            t0 = pc()
            solver = ilqrSolver(problem, N)
            _, Ui, _ = solver.solve(x0i)
            print(f"Problem {id_}: Took {pc() - t0} seconds\n" + "=" * 60)

            nu_i = problem.dynamics.n_u
            ind_i = self.ids.index(id_)
            U_warm[:, nu_i * ind_i : nu_i * (ind_i + 1)] = Ui

        print(f"All: {self.ids}\nTook {pc() - t0_all} seconds\n" + "=" * 80)

        return U_warm

    def __repr__(self):
        return f"ilqrProblem(\n\t{self.dynamics},\n\t{self.game_cost}\n)"


def solve_subproblem(args, **kwargs):
    """Solve the sub-problem and extract results for this agent"""

    subproblem, x0, U, id_ = args
    N = U.shape[0]

    subsolver = ilqrSolver(subproblem, N)
    Xi, Ui, _ = subsolver.solve(x0, U, **kwargs)
    return *subproblem.extract(Xi, Ui, id_), id_


def solve_subproblem_starmap(subproblem, x0, U, id_):
    """Package up the input arguments for compatiblity with mp.imap()."""
    return solve_subproblem((subproblem, x0, U, id_))


def _reset_ids():
    """Set each of the agent specific ID's to zero for understandability"""
    DynamicalModel._reset_ids()
    ReferenceCost._reset_ids()
