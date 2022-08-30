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

static void f_human_6d(double x[], double u[], double x_dot[])
{

    /* x: [px, py, pz, theta, 0, 0]
       u: [v, omega, 0]
       NOTE: the human agent is modelled as a simple unicycle with a constant height, so pz_dot = 0
       NOTE: we zero-pad some of the states and controls to more closely line up with the dimensions
             of the 6-dimensional quadcopter model
    */

    x_dot[0] = u[0] * cos(x[3]);
    x_dot[1] = u[0] * sin(x[3]);
    x_dot[2] = 0;
    x_dot[3] = u[1];
    x_dot[4] = 0;
    x_dot[5] = 0;
}

void linearize_human_6d(double x[], double u[], double dt, double A[], double B[])
{

    A[0] = 0;
    A[1] = 0;
    A[2] = 0;
    A[3] = -u[0] * sin(x[3]);
    A[4] = 0;
    A[5] = 0;
    A[6] = 0;
    A[7] = 0;
    A[8] = 0;
    A[9] = u[0] * cos(x[3]);
    A[10] = 0;
    A[11] = 0;
    A[12] = 0;
    A[13] = 0;
    A[14] = 0;
    A[15] = 0;
    A[16] = 0;
    A[17] = 0;
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

    B[0] = cos(x[3]);
    B[1] = 0;
    B[2] = 0;
    B[3] = sin(x[3]);
    B[4] = 0;
    B[5] = 0;
    B[6] = 0;
    B[7] = 0;
    B[8] = 0;
    B[9] = 0;
    B[10] = 1;
    B[11] = 0;
    B[12] = 0;
    B[13] = 0;
    B[14] = 0;
    B[15] = 0;
    B[16] = 0;
    B[17] = 0;

    euler_method_discretization(dt, A, B, 6, 3);
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

static void f_quad_12d(double x[], double u[], double f_sym[])
{
    /* x: [px, py, pz, psi, theta, phi, vx, vy, vz, wx, wy, wz]
       u: [tau_x, tau_y, tau_z, f_z]
    */

    f_sym[0] = x[6] * cos(x[3]) * cos(x[4]) + x[7] * (sin(x[5]) * sin(x[4]) * cos(x[3]) - sin(x[3]) * cos(x[5])) + x[8] * (sin(x[5]) * sin(x[3]) + sin(x[4]) * cos(x[5]) * cos(x[3]));
    f_sym[1] = x[6] * sin(x[3]) * cos(x[4]) + x[7] * (sin(x[5]) * sin(x[3]) * sin(x[4]) + cos(x[5]) * cos(x[3])) + x[8] * (-sin(x[5]) * cos(x[3]) + sin(x[3]) * sin(x[4]) * cos(x[5]));
    f_sym[2] = -x[6] * sin(x[4]) + x[7] * sin(x[5]) * cos(x[4]) + x[8] * cos(x[5]) * cos(x[4]);
    f_sym[3] = x[10] * sin(x[5]) / cos(x[4]) + x[11] * cos(x[5]) / cos(x[4]);
    f_sym[4] = x[10] * cos(x[5]) - x[11] * sin(x[5]);
    f_sym[5] = x[9] + x[10] * sin(x[5]) * tan(x[4]) + x[11] * cos(x[5]) * tan(x[4]);
    f_sym[6] = x[7] * x[11] - x[8] * x[10] + g * sin(x[4]);
    f_sym[7] = -x[6] * x[11] + x[8] * x[9] - g * sin(x[5]) * cos(x[4]);
    f_sym[8] = (2000.0 / 63.0) * u[3] + x[6] * x[10] - x[7] * x[9] - g * cos(x[5]) * cos(x[4]);
    f_sym[9] = (625000000000000000.0 / 10982593196059.0) * u[0] - 85899976080679.0 / 175721491136944.0 * x[10] * x[11];
    f_sym[10] = (5000000000000000000.0 / 92848985528431.0) * u[1] + (95876456000597.0 / 185697971056862.0) * x[9] * x[11];
    f_sym[11] = (10000000000000000000.0 / 271597947137541.0) * u[2] - 9976479919918.0 / 271597947137541.0 * x[9] * x[10];
}

static void linearize_quad_12d(double x[], double u[], double dt, double A[], double B[])
{

    A[0] = 0;
    A[1] = 0;
    A[2] = 0;
    A[3] = -x[6] * sin(x[3]) * cos(x[4]) + x[7] * (-sin(x[5]) * sin(x[3]) * sin(x[4]) - cos(x[5]) * cos(x[3])) + x[8] * (sin(x[5]) * cos(x[3]) - sin(x[3]) * sin(x[4]) * cos(x[5]));
    A[4] = -x[6] * sin(x[4]) * cos(x[3]) + x[7] * sin(x[5]) * cos(x[3]) * cos(x[4]) + x[8] * cos(x[5]) * cos(x[3]) * cos(x[4]);
    A[5] = x[7] * (sin(x[5]) * sin(x[3]) + sin(x[4]) * cos(x[5]) * cos(x[3])) + x[8] * (-sin(x[5]) * sin(x[4]) * cos(x[3]) + sin(x[3]) * cos(x[5]));
    A[6] = cos(x[3]) * cos(x[4]);
    A[7] = sin(x[5]) * sin(x[4]) * cos(x[3]) - sin(x[3]) * cos(x[5]);
    A[8] = sin(x[5]) * sin(x[3]) + sin(x[4]) * cos(x[5]) * cos(x[3]);
    A[9] = 0;
    A[10] = 0;
    A[11] = 0;
    A[12] = 0;
    A[13] = 0;
    A[14] = 0;
    A[15] = x[6] * cos(x[3]) * cos(x[4]) + x[7] * (sin(x[5]) * sin(x[4]) * cos(x[3]) - sin(x[3]) * cos(x[5])) + x[8] * (sin(x[5]) * sin(x[3]) + sin(x[4]) * cos(x[5]) * cos(x[3]));
    A[16] = -x[6] * sin(x[3]) * sin(x[4]) + x[7] * sin(x[5]) * sin(x[3]) * cos(x[4]) + x[8] * sin(x[3]) * cos(x[5]) * cos(x[4]);
    A[17] = x[7] * (-sin(x[5]) * cos(x[3]) + sin(x[3]) * sin(x[4]) * cos(x[5])) + x[8] * (-sin(x[5]) * sin(x[3]) * sin(x[4]) - cos(x[5]) * cos(x[3]));
    A[18] = sin(x[3]) * cos(x[4]);
    A[19] = sin(x[5]) * sin(x[3]) * sin(x[4]) + cos(x[5]) * cos(x[3]);
    A[20] = -sin(x[5]) * cos(x[3]) + sin(x[3]) * sin(x[4]) * cos(x[5]);
    A[21] = 0;
    A[22] = 0;
    A[23] = 0;
    A[24] = 0;
    A[25] = 0;
    A[26] = 0;
    A[27] = 0;
    A[28] = -x[6] * cos(x[4]) - x[7] * sin(x[5]) * sin(x[4]) - x[8] * sin(x[4]) * cos(x[5]);
    A[29] = x[7] * cos(x[5]) * cos(x[4]) - x[8] * sin(x[5]) * cos(x[4]);
    A[30] = -sin(x[4]);
    A[31] = sin(x[5]) * cos(x[4]);
    A[32] = cos(x[5]) * cos(x[4]);
    A[33] = 0;
    A[34] = 0;
    A[35] = 0;
    A[36] = 0;
    A[37] = 0;
    A[38] = 0;
    A[39] = 0;
    A[40] = x[10] * sin(x[5]) * sin(x[4]) / pow(cos(x[4]), 2) + x[11] * sin(x[4]) * cos(x[5]) / pow(cos(x[4]), 2);
    A[41] = x[10] * cos(x[5]) / cos(x[4]) - x[11] * sin(x[5]) / cos(x[4]);
    A[42] = 0;
    A[43] = 0;
    A[44] = 0;
    A[45] = 0;
    A[46] = sin(x[5]) / cos(x[4]);
    A[47] = cos(x[5]) / cos(x[4]);
    A[48] = 0;
    A[49] = 0;
    A[50] = 0;
    A[51] = 0;
    A[52] = 0;
    A[53] = -x[10] * sin(x[5]) - x[11] * cos(x[5]);
    A[54] = 0;
    A[55] = 0;
    A[56] = 0;
    A[57] = 0;
    A[58] = cos(x[5]);
    A[59] = -sin(x[5]);
    A[60] = 0;
    A[61] = 0;
    A[62] = 0;
    A[63] = 0;
    A[64] = x[10] * (pow(tan(x[4]), 2) + 1) * sin(x[5]) + x[11] * (pow(tan(x[4]), 2) + 1) * cos(x[5]);
    A[65] = x[10] * cos(x[5]) * tan(x[4]) - x[11] * sin(x[5]) * tan(x[4]);
    A[66] = 0;
    A[67] = 0;
    A[68] = 0;
    A[69] = 1;
    A[70] = sin(x[5]) * tan(x[4]);
    A[71] = cos(x[5]) * tan(x[4]);
    A[72] = 0;
    A[73] = 0;
    A[74] = 0;
    A[75] = 0;
    A[76] = g * cos(x[4]);
    A[77] = 0;
    A[78] = 0;
    A[79] = x[11];
    A[80] = -x[10];
    A[81] = 0;
    A[82] = -x[8];
    A[83] = x[7];
    A[84] = 0;
    A[85] = 0;
    A[86] = 0;
    A[87] = 0;
    A[88] = g * sin(x[5]) * sin(x[4]);
    A[89] = -g * cos(x[5]) * cos(x[4]);
    A[90] = -x[11];
    A[91] = 0;
    A[92] = x[9];
    A[93] = x[8];
    A[94] = 0;
    A[95] = -x[6];
    A[96] = 0;
    A[97] = 0;
    A[98] = 0;
    A[99] = 0;
    A[100] = g * sin(x[4]) * cos(x[5]);
    A[101] = g * sin(x[5]) * cos(x[4]);
    A[102] = x[10];
    A[103] = -x[9];
    A[104] = 0;
    A[105] = -x[7];
    A[106] = x[6];
    A[107] = 0;
    A[108] = 0;
    A[109] = 0;
    A[110] = 0;
    A[111] = 0;
    A[112] = 0;
    A[113] = 0;
    A[114] = 0;
    A[115] = 0;
    A[116] = 0;
    A[117] = 0;
    A[118] = -85899976080679.0 / 175721491136944.0 * x[11];
    A[119] = -85899976080679.0 / 175721491136944.0 * x[10];
    A[120] = 0;
    A[121] = 0;
    A[122] = 0;
    A[123] = 0;
    A[124] = 0;
    A[125] = 0;
    A[126] = 0;
    A[127] = 0;
    A[128] = 0;
    A[129] = (95876456000597.0 / 185697971056862.0) * x[11];
    A[130] = 0;
    A[131] = (95876456000597.0 / 185697971056862.0) * x[9];
    A[132] = 0;
    A[133] = 0;
    A[134] = 0;
    A[135] = 0;
    A[136] = 0;
    A[137] = 0;
    A[138] = 0;
    A[139] = 0;
    A[140] = 0;
    A[141] = -9976479919918.0 / 271597947137541.0 * x[10];
    A[142] = -9976479919918.0 / 271597947137541.0 * x[9];
    A[143] = 0;

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
    B[11] = 0;
    B[12] = 0;
    B[13] = 0;
    B[14] = 0;
    B[15] = 0;
    B[16] = 0;
    B[17] = 0;
    B[18] = 0;
    B[19] = 0;
    B[20] = 0;
    B[21] = 0;
    B[22] = 0;
    B[23] = 0;
    B[24] = 0;
    B[25] = 0;
    B[26] = 0;
    B[27] = 0;
    B[28] = 0;
    B[29] = 0;
    B[30] = 0;
    B[31] = 0;
    B[32] = 0;
    B[33] = 0;
    B[34] = 0;
    B[35] = 2000.0 / 63.0;
    B[36] = 625000000000000000.0 / 10982593196059.0;
    B[37] = 0;
    B[38] = 0;
    B[39] = 0;
    B[40] = 0;
    B[41] = 5000000000000000000.0 / 92848985528431.0;
    B[42] = 0;
    B[43] = 0;
    B[44] = 0;
    B[45] = 0;
    B[46] = 10000000000000000000.0 / 271597947137541.0;
    B[47] = 0;

    euler_method_discretization(dt, A, B, 12, 4);
}