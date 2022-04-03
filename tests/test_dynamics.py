#!/usr/bin/env python

"""Unit test for dynamical models

TODO

"""

import unittest

import numpy as np

from decentralized import (
    DoubleInt1dDynamics, DoubleInt2dDynamics, CarDynamics, UnicycleDynamics
)


class TestDoubleInt1dDynamics(unittest.TestCase):
    
    def setUp(self):
        self.model = DoubleInt1dDynamics(1.0)

    def test_call(self):
        x0 = np.array()


class TestDoubleInt2dDynamics(unittest.TestCase):

    def setUp(self):
        self.model = DoubleInt2dDynamics(1.0)


class TestCarDynamics(unittest.TestCase):

    def setUp(self):
        self.model = CarDynamics(1.0)


class TestUnicycleDynamics(unittest.TestCase):

    def setUp(self):
        self.model = UnicycleDynamics(1.0)

        