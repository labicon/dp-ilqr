import numpy as np
from casadi import *
import do_mpc


def baseline_drone_mpe(model, v_max = 6.0, )