#!/usr/bin/env python

"""Unit test for dynamical models"""

import unittest

import numpy as np

import decentralized as dec


class _TestDynamics:
    def _test_integrate(self, x0, u, X_truth):
        x = x0.copy()
        for x_expect in X_truth:
            self.assertTrue(np.allclose(x, x_expect, atol=0.1))
            x = self.model(x, u)

    def _test_linearize(self, x0, u, **kwargs):
        A, B = self.model.linearize(x0, u)
        A_diff, B_diff = dec.linearize_finite_difference(self.model.__call__, x0, u)

        self.assertTrue(np.allclose(A, A_diff, **kwargs))
        self.assertTrue(np.allclose(B, B_diff, **kwargs))


class TestDoubleInt4D(_TestDynamics, unittest.TestCase):
    def setUp(self):
        self.model = dec.DoubleIntDynamics4D(0.5)

    def test_call(self):
        x = np.array([0, 2, 0, -2])
        u = np.array([0, 2])
        X_truth = np.array([
            [0, 2,   0, -2],
            [0, 1,   0, -1],
            [0, 0.5, 0,  0],
            [0, 0.5, 0,  1],
            [0, 1,   0,  2]
        ])
        super()._test_integrate(x, u, X_truth)

    def test_integrate(self):
        x = np.random.rand(4)
        u = np.random.rand(2)
        super()._test_linearize(x, u)


class TestCarDynamics3D(_TestDynamics, unittest.TestCase):
    def setUp(self):
        self.model = dec.CarDynamics3D(0.5)

    def test_call(self):
        x0 = np.array([0, 0, np.pi/4])
        u = np.array([1, 0])
        X_truth = np.c_[
            self.model.dt * np.sqrt(2)/2 * np.array([
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3]
            ]),
            np.full((4,1), np.pi/4)
        ]
        super()._test_integrate(x0, u, X_truth)

    def test_linearize(self):
        x0 = np.random.rand(3)
        u = np.random.randn(2)
        super()._test_linearize(x0, u)
        

class TestUnicycle4D(_TestDynamics, unittest.TestCase):
    def setUp(self):
        self.model = dec.UnicycleDynamics4D(1.0)

    def test_straight(self):
        x0 = np.zeros(4)
        u = np.array([1, 0])
        X_truth = self.model.dt * np.array([
            [              0, 0, 0, 0],
            [              0, 0, 1, 0],
            [  self.model.dt, 0, 2, 0],
            [3*self.model.dt, 0, 3, 0]
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
        super()._test_linearize(x, u, atol=1e-3)


class TestBikeDynamics5D(_TestDynamics, unittest.TestCase):
    def setUp(self):
        self.model = dec.BikeDynamics5D(0.5)

    def test_linearize(self):
        x = np.random.rand(5)
        u = np.random.rand(2)
        super()._test_linearize(x, u)


if __name__ == "__main__":
    unittest.main()
