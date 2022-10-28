import numpy as np
from casadi import *
import do_mpc

def baseline_drone_simulator(model):
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step = 0.05)
    simulator.setup()

    return simulator