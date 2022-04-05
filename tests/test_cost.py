#!/usr/bin/env python

"""Unit testing for the cost module

TODO

"""

import unittest

import numpy as np

from decentralized import (
    NumericalDiffCost, ReferenceCost, ObstacleCost, CouplingCost, AgentCost, 
    GameCost, _quadraticize_distance
)


class ReferenceCostDiff(NumericalDiffCost, ReferenceCost):
    pass


class TestReferenceCost(unittest.TestCase):
    
    def setUp(self):
        self.n, self.m = 3, 2
        self.xf = np.zeros(self.n)
        self.Q = np.eye(self.n)
        self.R = np.eye(self.m)
        self.Qf = np.diag([1, 1, 0])
        #self.ref_cost = ReferenceCost(self.xf, self.Q, self.R, self.Qf)
        # self.ref_cost = ReferenceCostDiff(self.xf, self.Q, self.R, self.Qf)
        
        self.ref_cost = NumericalDiffCost(self.xf, self.Q, self.R, self.Qf)
        
        self.x0 = np.random.randint(0, 10, (self.n,))
        self.u = np.random.randint(0, 10, (self.m,))

    def test_call(self):
        expected = np.sum(np.linalg.norm(self.x0)**2) \
                 + np.sum(np.linalg.norm(self.u)**2)
        self.assertAlmostEqual(expected, self.ref_cost(self.x0, self.u))

    def test_terminal_call(self):
        expected = np.sum(np.linalg.norm(self.x0[:-1])**2)
        self.assertAlmostEqual(expected, 
                               self.ref_cost(self.x0, self.u, terminal=True))

    def test_quadraticize(self):
        Q_plus_Q_T = 2*np.eye(self.n)
        R_plus_R_T = 2*np.eye(self.m)
        L_x_expect = self.x0.T @ Q_plus_Q_T,
        L_u_expect = self.u.T @ R_plus_R_T,
        L_xx_expect = Q_plus_Q_T,
        L_uu_expect = R_plus_R_T,
        L_ux_expect = np.zeros((self.m, self.n))
        
        L_x, L_u, L_xx, L_uu, L_ux = self.ref_cost.quadraticize(self.x0, self.u)
        self.assertTrue(np.allclose(L_x, L_x_expect))
        self.assertTrue(np.allclose(L_u, L_u_expect))
        self.assertTrue(np.allclose(L_xx, L_xx_expect))
        self.assertTrue(np.allclose(L_uu, L_uu_expect))
        self.assertTrue(np.allclose(L_ux, L_ux_expect))


if __name__ == "__main__":
    unittest.main()

