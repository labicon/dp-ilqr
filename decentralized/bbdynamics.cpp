/*
 * bbdynamics.cpp
 * Barebones dynamics library for faster integration and slightly faster linearization.
 */

#include <math.h>
#include <stdio.h>
#include <iostream>

// Acceleration due to gravity.
#define g 9.80665

typedef void (*f_ptr)(double x[], double u[], double x_dot[]);

static void sum_scale_array(double a[], double b[], double c[], double scale, size_t size)
/* Compute the sum of two arrays with a scaled addend. */
{
    for (uint16_t i = 0; i < size; ++i)
        c[i] = a[i] + scale * b[i];
}

static void copy_array(double a[], double b[], size_t size)
/* Copy array a into array b */
{
    for (uint16_t i = 0; i < size; ++i)
        b[i] = a[i];
}

static void print_array(double arr[], size_t size)
/* Print an array to stdout */
{
    std::cout << "[ ";
    for (uint16_t i = 0; i < size; ++i)
        std::cout << arr[i] << " ";
    std::cout << "]" << std::endl;
}

// Adapted from: https://people.sc.fsu.edu/~jburkardt/cpp_src/rk4/rk4.cpp
void rk4(
    void dxdt(double x[], double u[], double x_dot[]),
    double dt, double x[], double u[], size_t n_x, double x_new[])
/* Integrate the ODE x_dot = f(x, u) for a duration of time of dt over steps dh for
   finer granularity.
*/
{
    uint16_t i, j;

    // Take smaller steps from the initial condition.
    int N = 5;
    double dh = dt / N;

    double *k0 = new double[n_x];
    double *k1 = new double[n_x];
    double *k2 = new double[n_x];
    double *k3 = new double[n_x];
    double *x0 = new double[n_x];
    double *x1 = new double[n_x];
    double *x2 = new double[n_x];
    double *x3 = new double[n_x];

    copy_array(x, x_new, n_x);
    for (j = 0; j < N; ++j)
    {
        copy_array(x_new, x0, n_x);
        dxdt(x0, u, k0);

        sum_scale_array(x0, k0, x1, dh / 2.0, n_x);
        dxdt(x1, u, k1);

        sum_scale_array(x0, k1, x2, dh / 2.0, n_x);
        dxdt(x2, u, k2);

        sum_scale_array(x0, k2, x3, dh, n_x);
        dxdt(x3, u, k3);

        for (i = 0; i < n_x; ++i)
        {
            x_new[i] += dh * (k0[i] + 2.0 * k1[i] + 2.0 * k2[i] + k3[i]) / 6.0;
        }
    }

    // Free heap memory.
    delete[] k0;
    delete[] k1;
    delete[] k2;
    delete[] k3;
    delete[] x0;
    delete[] x1;
    delete[] x2;
    delete[] x3;

    return;
}

static void euler_method_discretization(double dt, double A[], double B[], int n_x, int n_u)
{
    // Apply Newton Euler integration across the flattened jacobians.
    for (uint16_t i = 0; i < n_x * n_x; ++i)
    {
        A[i] *= dt;
        if (i % (n_x + 1) == 0)
            A[i] += 1;
    }
    for (uint16_t i = 0; i < n_x * n_u; ++i)
        B[i] *= dt;
}

static void f_double_int_4d(double x[], double u[], double x_dot[])
{
    /* x: [px, py, vx, vy]
       u: [ax, ay]
    */
    x_dot[0] = x[2];
    x_dot[1] = x[3];
    x_dot[2] = u[0];
    x_dot[3] = u[1];
}

static void linearize_double_int_4d(double dt, double A[], double B[])
{
    A[0] = 0;
    A[1] = 0;
    A[2] = 1;
    A[3] = 0;
    A[4] = 0;
    A[5] = 0;
    A[6] = 0;
    A[7] = 1;
    A[8] = 0;
    A[9] = 0;
    A[10] = 0;
    A[11] = 0;
    A[12] = 0;
    A[13] = 0;
    A[14] = 0;
    A[15] = 0;

    B[0] = 0;
    B[1] = 0;
    B[2] = 0;
    B[3] = 0;
    B[4] = 1;
    B[5] = 0;
    B[6] = 0;
    B[7] = 1;

    euler_method_discretization(dt, A, B, 4, 2);
}

static void f_car_3d(double x[], double u[], double x_dot[])
{
    /* x: [px, py, theta]
       u: [v, omega]
    */

    x_dot[0] = u[0] * cos(x[2]);
    x_dot[1] = u[0] * sin(x[2]);
    x_dot[2] = u[1];
}

static void linearize_car_3d(double x[], double u[], double dt, double A[], double B[])
{

    A[0] = 0;
    A[1] = 0;
    A[2] = -u[0] * sin(x[2]);
    A[3] = 0;
    A[4] = 0;
    A[5] = u[0] * cos(x[2]);
    A[6] = 0;
    A[7] = 0;
    A[8] = 0;

    B[0] = cos(x[2]);
    B[1] = 0;
    B[2] = sin(x[2]);
    B[3] = 0;
    B[4] = 0;
    B[5] = 1;

    euler_method_discretization(dt, A, B, 3, 2);
}

static void f_unicycle_4d(double x[], double u[], double x_dot[])
{
    /* x: [px, py, v, theta]
       u: [a, omega]
    */

    x_dot[0] = x[2] * cos(x[3]);
    x_dot[1] = x[2] * sin(x[3]);
    x_dot[2] = u[0];
    x_dot[3] = u[1];
}

static void linearize_unicycle_4d(double x[], double u[], double dt, double A[], double B[])
{

    A[0] = 0;
    A[1] = 0;
    A[2] = cos(x[3]);
    A[3] = -x[2] * sin(x[3]);
    A[4] = 0;
    A[5] = 0;
    A[6] = sin(x[3]);
    A[7] = x[2] * cos(x[3]);
    A[8] = 0;
    A[9] = 0;
    A[10] = 0;
    A[11] = 0;
    A[12] = 0;
    A[13] = 0;
    A[14] = 0;
    A[15] = 0;

    B[0] = 0;
    B[1] = 0;
    B[2] = 0;
    B[3] = 0;
    B[4] = 1;
    B[5] = 0;
    B[6] = 0;
    B[7] = 1;

    euler_method_discretization(dt, A, B, 4, 2);
}

static void f_unicycle_human(double x[], double u[], double x_dot[])
{

    /* x: [px, py, pz, theta]
       u: [v, omega]
       NOTE: the human agent is modelled as a simple unicycle with a constant height, so pz_dot = 0
    */

    x_dot[0] = u[0] * cos(x[3]);
    x_dot[1] = u[0] * sin(x[3]);
    x_dot[2] = 0;
    x_dot[3] = u[1];
}

static void linearize_unicycle_human(double x[], double u[], double dt, double A[], double B[])
{

    A[0] = 0;
    A[1] = 0;
    A[2] = 0;
    A[3] = -u[0] * sin(x[3]);
    A[4] = 0;
    A[5] = 0;
    A[6] = 0;
    A[7] = u[0] * cos(x[3]);
    A[8] = 0;
    A[9] = 0;
    A[10] = 0;
    A[11] = 0;
    A[12] = 0;
    A[13] = 0;
    A[14] = 0;
    A[15] = 0;

    B[0] = cos(x[3]);
    B[1] = 0;
    B[2] = sin(x[3]);
    B[3] = 0;
    B[4] = 0;
    B[5] = 0;
    B[6] = 0;
    B[7] = 1;

    euler_method_discretization(dt, A, B, 4, 2);
}

static void f_quad_6d(double x[], double u[], double x_dot[])
{
    /* x: [px, py, pz, vx, vy, vz]
       u: [tau, phi, theta]
    */

    x_dot[0] = x[3];
    x_dot[1] = x[4];
    x_dot[2] = x[5];
    x_dot[3] = g * tan(u[2]);
    x_dot[4] = -g * tan(u[1]);
    x_dot[5] = u[0] - g;
}

static void linearize_quad_6d(double x[], double u[], double dt, double A[], double B[])
{

    A[0] = 0;
    A[1] = 0;
    A[2] = 0;
    A[3] = 1;
    A[4] = 0;
    A[5] = 0;
    A[6] = 0;
    A[7] = 0;
    A[8] = 0;
    A[9] = 0;
    A[10] = 1;
    A[11] = 0;
    A[12] = 0;
    A[13] = 0;
    A[14] = 0;
    A[15] = 0;
    A[16] = 0;
    A[17] = 1;
    A[18] = 0;
    A[19] = 0;
    A[20] = 0;
    A[21] = 0;
    A[22] = 0;
    A[23] = 0;
    A[24] = 0;
    A[25] = 0;
    A[26] = 0;
    A[27] = 0;
    A[28] = 0;
    A[29] = 0;
    A[30] = 0;
    A[31] = 0;
    A[32] = 0;
    A[33] = 0;
    A[34] = 0;
    A[35] = 0;

    B[0] = 0;
    B[1] = 0;
    B[2] = 0;
    B[3] = 0;
    B[4] = 0;
    B[5] = 0;
    B[6] = 0;
    B[7] = 0;
    B[8] = 0;
    B[9] = 0;
    B[10] = 0;
    B[11] = g * pow(tan(u[2]), 2) + g;
    B[12] = 0;
    B[13] = -g * pow(tan(u[1]), 2) - g;
    B[14] = 0;
    B[15] = 1;
    B[16] = 0;
    B[17] = 0;

    euler_method_discretization(dt, A, B, 6, 3);
}
