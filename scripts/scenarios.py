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


def four_quads_exchange():
    x0 = np.array([[0.0, 0, 1, 0, 0, 0,
                    1.0, 0, 1, 0, 0, 0,
                    2.0, 0, 1, 0, 0, 0,
                    3.0, 0, 1, 0, 0, 0
                    ]]).T
    xf = np.array([[3.0, 1, 1, 0, 0, 0,
                    0.0, 1, 1, 0, 0, 0,
                    1.0, 1, 1, 0, 0, 0,
                    2.0, 1, 1, 0, 0, 0,
                    ]]).T
    x0[pos_mask([6]*4, 3)] += 0.1*np.random.randn(12, 1)
    xf[pos_mask([6]*4, 3)] += 0.1*np.random.randn(12, 1)
    return x0, xf

def four_quads_passthrough():
    x0 = np.array([[-0.117, 0.179, 0.963, 0.0, 0.0, 0.0, 
                    0.9, 0.118, 1.111, 0.0, 0.0, 0.0, 
                    1.943, 0.065, 0.987, 0.0, 0.0, 0.0, 
                    3.14, -0.077, 1.083, 0.0, 0.0, 0.0]]).T
    xf = np.array([[2.989, 1.046, 0.986, 0.0, 0.0, 0.0, 
                    -0.054, 0.979, 1.077, 0.0, 0.0, 0.0, 
                    1.051, 1.121, 1.111, 0.0, 0.0, 0.0, 
                    2.038, 1.006, 0.767, 0.0, 0.0, 0.0]]).T
    return x0, xf


def four_quads_box_exchange():
    """
    1_________2

        

    3_________4


    """
    
    
    x0 = np.array([[-2.5, 2.5, 1.0, 0, 0, 0,
                    2.5, 2.5, 1.0, 0, 0, 0,
                    -2.5, -2.5, 1.0, 0, 0, 0,
                    2.5, -2.5, 1.0, 0, 0, 0,       
                    ]]).T
    
    xf = np.array([[2.5, -2.5, 1.0, 0, 0, 0,
                    -2.5, -2.5, 1.0, 0, 0, 0,
                    2.5, 2.5, 1.0, 0, 0, 0,
                    -2.5, 2.5, 1.0, 0, 0,
                    ]]).T
    
    # x0[pos_mask([6]*4, 3)] += 0.1*np.random.randn(12, 1)
    # xf[pos_mask([6]*4, 3)] += 0.1*np.random.randn(12, 1)
    
    return x0, xf
