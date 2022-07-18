#!/usr/bin/env python

"""Unit test for dynamical models"""

import unittest

import numpy as np
import torch

from decentralized import UnicycleDynamics4D, UnicycleDynamics4dSymbolic, linearize_finite_difference


class _TestUnicycleDynamics4D:
    def _test_integrate(self, x0, u, X_truth):
        x = x0.copy()
        for x_expect in X_truth:
            self.assertTrue(np.allclose(x, x_expect, atol=0.1))
            x = self.model(x, u)


class TestUnicycleAnalytical(_TestUnicycleDynamics4D, unittest.TestCase):
    def setUp(self):
        self.mm = np
        self.model = UnicycleDynamics4dSymbolic(0.5)

    def test_straight(self):
        x0 = np.zeros(4)
        u = np.array([1, 0])
        X_truth = np.array([
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 2, 0],
            [3, 0, 3, 0]

        ])
        super()._test_integrate(x0, u, X_truth)

    def test_curve(self):
        v = np.pi
        r = 10
        omega = v / r
        theta0 = np.pi/2 + omega/2
        x0 = np.array([r, 0, v, theta0])
        u = np.array([0, omega])
        theta = np.arange(0, 2*np.pi + omega, omega).reshape(-1,1)
        X_truth = np.hstack([
            r * np.cos(theta),
            r * np.sin(theta),
            np.full(theta.shape, v),
            theta0 + theta
        ])
        super()._test_integrate(x0, u, X_truth)
    
    def test_linearize(self):
        x = 10 * np.random.randn(4)
        u = 10 * np.random.randn(2)
        A, B = self.model.linearize(x, u)
        A_diff, B_diff = linearize_finite_difference(self.model.__call__, x, u)

        self.assertTrue(np.allclose(A, A_diff, atol=1e-3))
        self.assertTrue(np.allclose(B, B_diff, atol=1e-3))


class TestUnicycleDynamicsAutodiff(_TestUnicycleDynamics4D, unittest.TestCase):
    def setUp(self):
        self.mm = torch
        self.model = UnicycleDynamics4D(1.0)

    def test_integrate(self):
        x0 = torch.zeros(4)
        u = torch.tensor([1.0, 0.0])
        X_truth = torch.tensor([
            [0, 0, 1, 0],
            [1, 0, 2, 0],
            [3, 0, 3, 0]
        ])
        super()._test_integrate(x0, u, X_truth)


if __name__ == "__main__":
    unittest.main()
