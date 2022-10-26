#!/usr/bin/env python

"""Assortment of archived initial (x0) and final(xf) positions that serve as scenarios 
  to simulate
"""

import numpy as np

from dpilqr import pos_mask, π


def potential_ilqr_setup():
    """Hardcoded example with reasonable consistency eyeballed from
       Potential-iLQR paper
    """
    x0 = np.array([[0.5, 1.5, 0, 0.1,
                    2.5, 1.5, 0, π,
                    1.5, 1.3, 0, π/2]]).T
    xf = np.array([[2.5, 1.5, 0, 0,
                    0.5, 1.5, 0, π,
                    1.5, 2.2, 0, π/2]]).T
    return x0, xf


def paper_setup_3_quads():
    x0 = np.array([[0.5, 1.5, 1, 0.5, 0.0, 0,
                    2.5, 1.5, 1, -0.5, 0.0, 0,
                    1.5, 1.3, 1, 0, 0.2, 0]]).T
    xf = np.array([[2.5, 1.5, 1, 0, 0, 0,
                    0.5, 1.5, 1, 0, 0, 0,
                    1.5, 2.2, 1, 0, 0, 0]]).T
    x0[pos_mask([6]*3, 3)] += 0.1*np.random.randn(9, 1)
    xf[pos_mask([6]*3, 3)] += 0.1*np.random.randn(9, 1)
    return x0, xf


def paper_setup_5_quads():
    x0 = np.array([[0.5, 1.5, 1, 0.1, 0.0, 0,
                    2.5, 1.5, 1, -0.1, 0.0, 0,
                    1.5, 1.3, 1, 0, 0.1, 0,
                    0.5, 1.0, 1, 0.1, 0, 0,
                    1.2, -0.5, 1, 0, 0, 0]]).T
    xf = np.array([[2.5, 1.5, 1, 0, 0, 0,
                    0.5, 1.5, 1, 0, 0, 0,
                    1.5, 2.2, 1, 0, 0, 0,
                    -0.5, -0.6, 1, 0, 0, 0,
                    0.7, 1.0, 1, 0, 0, 0]]).T
    x0[pos_mask([6]*5, 3)] += 0.1*np.random.randn(15, 1)
    xf[pos_mask([6]*5, 3)] += 0.1*np.random.randn(15, 1)
    return x0, xf



def paper_setup_7_quads():
    x0 = np.array([[0.5, 1.5, 1, 0.1, 0.0, 0,
                    2.5, 1.5, 1, -0.1, 0.0, 0,
                    1.5, 1.3, 1, 0, 0.1, 0,
                    0.5, 1.0, 1, 0.1, 0, 0,
                    1.2, -0.5, 1, 0, 0, 0,
                    1.7, 1.4, 1, 0, 0, 0,
                    -1.5, 1.1, 1, 0.1, 0, 0]]).T
    xf = np.array([[2.5, 1.5, 1, 0, 0, 0,
                    0.5, 1.5, 1, 0, 0, 0,
                    1.5, 2.2, 1, 0, 0, 0,
                    -0.5, -0.6, 1, 0, 0, 0,
                    0.7, 1.0, 1, 0, 0, 0,
                    2.0, 2.1, 1, 0, 0, 0,
                    -0.6, 0.6, 1, 0, 0, 0]]).T
    x0[pos_mask([6]*7, 3)] += 0.1*np.random.randn(21, 1)
    xf[pos_mask([6]*7, 3)] += 0.1*np.random.randn(21, 1)
    return x0, xf


def two_quads_one_human_setup():
    x0 = np.array([[-1.5, 0.1, 1, 0, 0, 0,
                    1.5, 0, 1, 0, 0, 0,
                    0, -1, 1.5, 0, 0, 0]]).T
    xf = np.array([[1.5, 0, 2, 0, 0, 0,
                   -1.5, 0, 2, 0, 0, 0,
                   0.0, 2, 1.5, 0, 0, 0]]).T
    return x0, xf
