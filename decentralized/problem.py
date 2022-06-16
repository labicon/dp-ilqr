#!/usr/bin/env python

"""Logic to combine dynamics and cost in one framework"""


def decentralize_ocp(dynamics, cost, planning_radii, planning_horizon):
    pass


class NavigationProblem:
    """Centralized optimal control problem that combines dynamics and cost"""

    def __init__(self, dynamics, game_cost):
        self.dynamics = dynamics
        self.game_cost = game_cost

    def split(self, X, planning_radii, planning_horizon):
        """Split up this centralized problem into a list of decentralized
           sub-problems.
        """
        raise NotImplementedError
