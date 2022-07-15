#!/usr/bin/env python

"""Unit testing for the cost module

TODO

"""

import unittest

import numpy as np

from decentralized import (
    ReferenceCost, CouplingCost, GameCost, quadraticize_distance
)


class TestCouplingCost(unittest.TestCase):

    def test_single(self):
        cost = CouplingCost([2], 10.0)([1, 2])
        self.assertAlmostEqual(cost, 0.0)

    def test_call_2(self):
        r = 10.0
        x = np.array([0, 0, 0, 1, 2, 0])
        cost = CouplingCost([3, 3], r)(x)
        expected = (np.hypot(1, 2) - r)**2
        self.assertAlmostEqual(cost, expected)

    def test_quadraticize_2(self):
        r = 10.0
        x = np.array([0, 0, 0, 1, 2, 0])
        cost = CouplingCost([3, 3], r)
        Lx, *_ = cost.quadraticize(x)
        
        dx = 1
        dy = 2
        dist = np.hypot(dx, dy)
        Lx_half = 2 * (r - dist) / dist * np.array([dx, dy, 0])
        Lx_expect = np.r_[Lx_half, -Lx_half]

        self.assertTrue(np.allclose(Lx, Lx_expect))


@unittest.skip("TODO: switch to analytical")
class TestReferenceCost(unittest.TestCase):
    
    def setUp(self):
        self.n, self.m = 3, 2
        self.xf = np.zeros(self.n)
        self.Q = np.eye(self.n)
        self.R = np.eye(self.m)
        self.Qf = np.diag([1, 1, 0])

        self.ref_cost = ReferenceCost(self.xf, self.Q, self.R, self.Qf)
        
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
        
        L_x_expect = self.x0.T @ Q_plus_Q_T
        L_u_expect = self.u.T @ R_plus_R_T
        L_xx_expect = Q_plus_Q_T
        L_uu_expect = R_plus_R_T
        L_ux_expect = np.zeros((self.m, self.n))
        
        L_x, L_u, L_xx, L_uu, L_ux = self.ref_cost.quadraticize(self.x0, self.u)
        
        self.assertTrue(np.allclose(L_x, L_x_expect))
        self.assertTrue(np.allclose(L_u, L_u_expect))
        self.assertTrue(np.allclose(L_xx, L_xx_expect))
        self.assertTrue(np.allclose(L_uu, L_uu_expect))
        self.assertTrue(np.allclose(L_ux, L_ux_expect))


if __name__ == "__main__":
    unittest.main()

