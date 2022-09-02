# distutils: language = c++

from enum import Enum, auto

import numpy as np


class Model(Enum):
    DoubleInt4D = 0
    Car3D = auto()
    Unicycle4D = auto()
    Quadcopter6D = auto()
    Human6D = auto()
    Quadcopter12D = auto()


ctypedef void (*f_func)(double x[], double u[], double x_dot[])

cdef extern from "bbdynamics.cpp":

    void rk4(
        f_func dxdt, double dt, double x0[], double u[], size_t n_x, double x_new[]
    )

    void f_double_int_4d(double x[], double u[], double x_dot[])
    void linearize_double_int_4d(double dt, double A[], double B[])

    void f_car_3d(double x[], double u[], double x_dot[])
    void linearize_car_3d(double x[], double u[], double dt, double A[], double B[])

    void f_unicycle_4d(double x[], double u[], double x_dot[])
    void linearize_unicycle_4d(double x[], double u[], double dt, double A[], double B[])

    void f_human_6d(double x[], double u[], double x_dot[])
    void linearize_human_6d(double x[], double u[], double dt, double A[], double B[])

    void f_quad_6d(double x[], double u[], double x_dot[])
    void linearize_quad_6d(double x[], double u[], double dt, double A[], double B[])

    void f_quad_12d(double x[], double u[], double x_dot[])
    void linearize_quad_12d(double x[], double u[], double dt, double A[], double B[])


def _common_validation(model, x, u):
    if not isinstance(model, Model):
        raise ValueError()

    # TODO: remove when confident.
    if not x.flags["C_CONTIGUOUS"] or not u.flags["C_CONTIGUOUS"]:
        raise ValueError("Contiguous")


def f(x, u, model):
    _common_validation(model, x, u)

    cdef size_t n_x = x.shape[0]
    x_dot = np.empty(n_x, dtype=np.double)

    cdef double[::1] x_view = x
    cdef double[::1] u_view = u
    cdef double[::1] x_dot_view = x_dot

    cdef f_func f
    if model is Model.DoubleInt4D:
        f = f_double_int_4d
    elif model is Model.Car3D:
        f = f_car_3d
    elif model is Model.Unicycle4D:
        f = f_unicycle_4d
    elif model is Model.Human6D:
        f = f_human_6d
    elif model is Model.Quadcopter6D:
        f = f_quad_6d
    elif model is Model.Quadcopter12D:
        f = f_quad_12d

    f(&x_view[0], &u_view[0], &x_dot_view[0])
    return x_dot


def integrate(x, u, double dt, model):
    _common_validation(model, x, u)

    cdef size_t n_x = x.shape[0]
    x_new = np.empty(n_x, dtype=np.double)

    cdef double[::1] x_view = x
    cdef double[::1] u_view = u
    cdef double[::1] x_new_view = x_new

    # TODO: Figure out how to get rid of this duplicate branching logic.
    cdef f_func f;
    if model is Model.DoubleInt4D:
        f = f_double_int_4d
    elif model is Model.Car3D:
        f = f_car_3d
    elif model is Model.Unicycle4D:
        f = f_unicycle_4d
    elif model is Model.Human6D:
        f = f_human_6d
    elif model is Model.Quadcopter6D:
        f = f_quad_6d
    elif model is Model.Quadcopter12D:
        f = f_quad_12d

    rk4(f, dt, &x_view[0], &u_view[0], n_x, &x_new_view[0])
    return x_new

def linearize(x, u, double dt, model):

    if not isinstance(model, Model):
        raise ValueError()

    # TODO: remove when confident.
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    if not u.flags["C_CONTIGUOUS"]:
        u = np.ascontiguousarray(u)

    # Pre-allocate the flat output Jacobians.
    cdef size_t nx = x.shape[0]
    cdef size_t nu = u.shape[0]
    A = np.empty((nx*nx), dtype=np.double)
    B = np.empty((nx*nu), dtype=np.double)

    cdef double[::1] x_view = x
    cdef double[::1] u_view = u
    cdef double[::1] A_view = A
    cdef double[::1] B_view = B

    if model is Model.DoubleInt4D:
        linearize_double_int_4d(dt, &A_view[0], &B_view[0])
    elif model is Model.Car3D:
        linearize_car_3d(&x_view[0], &u_view[0], dt, &A_view[0], &B_view[0])
    elif model is Model.Unicycle4D:
        linearize_unicycle_4d(&x_view[0], &u_view[0], dt, &A_view[0], &B_view[0])
    elif model is Model.Human6D:
        linearize_human_6d(&x_view[0], &u_view[0], dt, &A_view[0], &B_view[0])
    elif model is Model.Quadcopter6D:
        linearize_quad_6d(&x_view[0], &u_view[0], dt, &A_view[0], &B_view[0])
    elif model is Model.Quadcopter12D:
        linearize_quad_12d(&x_view[0], &u_view[0], dt, &A_view[0], &B_view[0])

    return A.reshape((nx, nx)), B.reshape((nx, nu))
