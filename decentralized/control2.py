#!/usr/bin/env python

"""Functional programming implementation of the iLQR solver.

[1] Jackson. AL iLQR Tutorial. https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf

"""


import numpy as np

import torch


def _integrate_discrete(f, x, u, _dt):
    """Zero-order hold to discretize a state given continuous dynamics f"""
    return x + f(x, u) * _dt


def _linearize_dynamics(f, x_torch, u_torch, _dt):
    """
    Compute the Jacobian linearization of the dynamics for a particular
    state `x0` and controls `u0` for each player. Outputs `A` and `Bi`
    matrices of a linear system:
      ```\dot x - f(x0, u0) = A (x - x0) + sum_i Bi (ui - ui0) ```
    REF: [1]
    """

    nx = x_torch.numel()
    nu = u_torch.numel()

    A_cont, B_cont = torch.autograd.functional.jacobian(f, (x_torch, u_torch))

    # Compute the discretized A and B.
    A = _dt * A_cont.reshape(nx, nx) + np.eye(nx, dtype=np.float32)
    B = _dt * B_cont.reshape(nx, nu)
    return A, B


def quadraticize_cost(cost, x_torch, u_torch, **kwargs):
    """
    Compute a quadratic approximation to the overall cost for a
    particular choice of state `x`, and controls `u` for each player.
    Returns the gradient and Hessian of the overall cost such that:
    ```
       cost(x + dx, [ui + dui]) \approx
            cost(x, u1, u2) +
            grad_x^T dx +
            0.5 * (dx^T hess_x dx + sum_i dui^T hess_ui dui)
    ```
    REF: [1]
    """

    nx = x_torch.numel()
    nu = u_torch.numel()

    # WIP: Assume x and u are already torch.Tensors.
    # Convert to torch.Tensor format.
    # x_torch = torch.from_numpy(x).requires_grad_(True)
    # u_torch = [torch.from_numpy(ui).requires_grad_(True) for ui in u]

    def cost_fn(x, u):
        return cost(x, u, **kwargs)

    L_x, L_u = torch.autograd.functional.jacobian(cost_fn, (x_torch, u_torch))
    L_x = L_x.reshape(nx)
    L_u = L_u.reshape(nu)

    (L_xx, _), (L_ux, L_uu) = torch.autograd.functional.hessian(
        cost_fn, (x_torch, u_torch)
    )
    L_xx = L_xx.reshape(nx, nx)
    L_ux = L_ux.reshape(nu, nx)
    L_uu = L_uu.reshape(nu, nu)

    return L_x, L_u, L_xx, L_uu, L_ux


class iLQR:
    """
    iLQR solver with a functional approach
    """

    DELTA_0 = 2.0  # initial regularization scaling
    MU_MIN = 1e-6  # regularization floor, below which is 0.0
    MU_MAX = 1e3  # regularization ceiling
    N_LS_ITER = 10  # number of line search iterations

    def __init__(self, dynamics, cost, n_x, n_u, dt=0.1, N=10):

        self.dynamics = dynamics
        self.cost = cost

        self.N = N
        self.dt = dt
        self.n_x = n_x
        self.n_u = n_u

        self.μ = 1.0  # regularization
        self.Δ = self.DELTA_0  # regularization scaling

    def linearize_dynamics(self, *args, **kwargs):
        return _linearize_dynamics(self.dynamics, *args, **kwargs, _dt=self.dt)

    def integrate_dynamics(self, *args, **kwargs):
        return _integrate_discrete(self.dynamics, *args, **kwargs, _dt=self.dt)

    def quadraticize_cost(self, *args, **kwargs):
        return quadraticize_cost(self.cost, *args, **kwargs)

    def _rollout(self, x0, U):
        """Rollout the system from an initial state with a control sequence U."""

        N = U.shape[0]
        X = torch.zeros((N + 1, self.n_x))
        X[0] = x0.flatten()
        J = 0.0

        for t in range(N):
            X[t + 1] = self.integrate_dynamics(X[t], U[t])
            J += self.cost(X[t], U[t]).item()
        J += self.cost(X[-1], torch.zeros(self.n_u), terminal=True).item()

        return X, J

    def _forward_pass(self, X, U, K, d, α):
        """Forward pass to rollout the control gains K and d."""

        X_next = torch.zeros((self.N + 1, self.n_x))
        U_next = torch.zeros((self.N, self.n_u))

        X_next[0] = X[0].clone()
        J = 0.0

        for t in range(self.N):
            δx = X_next[t] - X[t]
            δu = K[t] @ δx + α * d[t]

            U_next[t] = U[t] + δu
            X_next[t + 1] = self.integrate_dynamics(X_next[t], U_next[t])

            J += self.cost(X_next[t], U_next[t]).item()
        J += self.cost(X_next[-1], torch.zeros((self.n_u)), terminal=True).item()

        return X_next, U_next, J

    def _backward_pass(self, X, U):
        """Backward pass to compute gain matrices K and d from the trajectory."""

        K = torch.zeros((self.N, self.n_u, self.n_x))  # linear feedback gain
        d = torch.zeros((self.N, self.n_u))  # feedforward gain

        # self.μ = 0.0 # DBG
        reg = self.μ * torch.eye(self.n_x)

        L_x, _, L_xx, _, _ = self.quadraticize_cost(
            X[-1], torch.zeros(self.n_u), terminal=True
        )
        p = L_x
        P = L_xx

        for t in range(self.N - 1, -1, -1):
            L_x, L_u, L_xx, L_uu, L_ux = self.quadraticize_cost(X[t], U[t])
            A, B = self.linearize_dynamics(X[t], U[t])

            Q_x = L_x + A.T @ p
            Q_u = L_u + B.T @ p
            Q_xx = L_xx + A.T @ P @ A
            Q_uu = L_uu + B.T @ (P + reg) @ B
            Q_ux = L_ux + B.T @ (P + reg) @ A

            K[t] = -torch.linalg.solve(Q_uu, Q_ux)
            d[t] = -torch.linalg.solve(Q_uu, Q_u)

            p = Q_x + K[t].T @ Q_uu @ d[t] + K[t].T @ Q_u + Q_ux.T @ d[t]
            P = Q_xx + K[t].T @ Q_uu @ K[t] + K[t].T @ Q_ux + Q_ux.T @ K[t]
            P = 0.5 * (P + P.T)

        return K, d

    def solve(self, x0, U=None, n_lqr_iter=50, tol=1e-3):

        if U is None:
            U = torch.zeros((self.N, self.n_u))
            # U = torch.full((self.N, self.n_u), 0.1)
            # U = 1e-3 * torch.rand((self.N, self.n_u))
        if U.shape != (self.N, self.n_u):
            raise ValueError

        # Reset regularization terms.
        self.μ = 1.0
        self.Δ = self.DELTA_0

        x0 = x0.reshape(-1, 1)
        is_converged = False
        alphas = 1.1 ** (-torch.arange(self.N_LS_ITER, dtype=torch.float32) ** 2)

        X, J_star = self._rollout(x0, U)

        print(f"0/{n_lqr_iter}\tJ: {J_star:g}")
        for i in range(n_lqr_iter):
            accept = False

            # Backward recurse to compute gain matrices.
            K, d = self._backward_pass(X, U)
            alphas_ls = alphas

            # Conduct a line search to find a satisfactory trajectory where we
            # continually decrease α. We're effectively getting closer to the
            # linear approximation in the LQR case.
            for α in alphas_ls:

                X_next, U_next, J = self._forward_pass(X, U, K, d, α)
                
                if J < J_star:
                    if np.abs((J_star - J) / J_star) < tol:
                        is_converged = True

                    X = X_next
                    U = U_next
                    J_star = J

                    # Decrease regularization to converge more slowly.
                    self.Δ = min(1.0, self.Δ) / self.DELTA_0
                    self.μ *= self.Δ
                    if self.μ <= self.MU_MIN:
                        self.μ = 0.0

                    accept = True
                    break

            if not accept:

                # DBG: Regularization is pointless for quadratic costs.
                # print('[solve] Failed line search.. giving up.')
                # break

                # Increase regularization if we're not converging.
                print("Failed line search.. increasing μ.")
                self.Δ = max(1.0, self.Δ) * self.DELTA_0
                self.μ = max(self.MU_MIN, self.μ * self.Δ)
                if self.μ >= self.MU_MAX:
                    print("Exceeded max regularization term...")
                    break

            if is_converged:
                break

            print(f"{i+1}/{n_lqr_iter}\tJ: {J_star:g}\tμ: {self.μ:g}\tΔ: {self.Δ:g}")

        return X.detach().numpy(), U.detach().numpy(), J

    def __repr__(self):
        return (
            f"iLQR(\n\tdynamics: {self.dynamics},\n\tcost: {self.cost},"
            f"\n\tN: {self.N},\n\tdt: {self.dt},\n\tμ: {self.μ},\n\tΔ: {self.Δ}"
            "\n)"
        )


class iLQR_v2:
    """
    iLQR solver based on the original C++
    """

    DELTA_0 = 2.0  # initial regularization scaling
    MU_MIN = 1e-6  # regularization floor, below which is 0.0
    MU_MAX = 1e3  # regularization ceiling
    N_LS_ITER = 10  # number of line search iterations

    def __init__(self, dynamics, cost, n_x, n_u, dt=0.1, N=10):

        self.dynamics = dynamics
        self.cost = cost

        self.N = N
        self.dt = dt
        self.n_x = n_x
        self.n_u = n_u

        self._reset_regularization()

    def linearize_dynamics(self, *args, **kwargs):
        return _linearize_dynamics(self.dynamics, *args, **kwargs, _dt=self.dt)

    def integrate_dynamics(self, *args, **kwargs):
        return _integrate_discrete(self.dynamics, *args, **kwargs, _dt=self.dt)

    def quadraticize_cost(self, *args, **kwargs):
        return quadraticize_cost(self.cost, *args, **kwargs)

    def solve(self, x0, U=None, n_lqr_iter=50, tol=1e-3):

        if U is None:
            U = torch.zeros((self.N, self.n_u))
            # U = torch.full((self.N, self.n_u), 0.1)
            # U = 1e-3 * torch.rand((self.N, self.n_u))
        if U.shape != (self.N, self.n_u):
            raise ValueError

        # breakpoint()
        self._reset_regularization()

        x0 = x0.reshape(-1, 1)
        converged = False
        changed = True
        alphas = 1.1 ** (-torch.arange(self.N_LS_ITER, dtype=torch.float32) ** 2)

        print(f"0/{n_lqr_iter}\tμ: {self.μ:g}\tΔ: {self.Δ:g}")
        for i in range(n_lqr_iter):
            accepted = False

            if changed:
                X, J_opt, F_derivs, L_derivs = self._forward_rollout(x0, U)
                changed = False

            # Backward recurse to compute gain matrices.
            K, d = self._backward_pass(F_derivs, L_derivs)

            # Conduct a line search to find a satisfactory trajectory where we
            # continually decrease α. We're effectively getting closer to the
            # linear approximation in the LQR case.
            for α in alphas:

                X_next, U_next, J = self._control(X, U, K, d, α)

                if J < J_opt:
                    if abs((J_opt - J) / J_opt) < tol:
                        converged = True

                    J_opt = J
                    X = X_next
                    U = U_next
                    changed = True
                    accepted = True
                    self._decrease_regularization()

                    break

            if not accepted:
                self._increase_regularization()
                if self.μ >= self.MU_MAX:
                    print("Exceeded max regularization term...")
                    break
                print("Failed line search.. increasing μ.")

            if converged:
                print(f"{i+1}/{n_lqr_iter}\tJ: {J_opt:g}\tμ: {self.μ:g}\tΔ: {self.Δ:g}")
                break

            print(f"{i+1}/{n_lqr_iter}\tJ: {J_opt:g}\tμ: {self.μ:g}\tΔ: {self.Δ:g}")

        return X.detach().numpy(), U.detach().numpy(), J

    def _forward_rollout(self, x0, U):
        """Rollout the system from an initial state with a control sequence U,
           gathering derivatives along the way.
        """

        N = U.shape[0]
        X = torch.zeros((N + 1, self.n_x))
        X[0] = x0.flatten()
        J = 0.0

        L_derivs = []
        F_derivs = []
        for t in range(N):
            F_derivs.append(self.linearize_dynamics(X[t], U[t]))
            L_derivs.append(self.quadraticize_cost(X[t], U[t]))
            X[t + 1] = self.integrate_dynamics(X[t], U[t])
            J += self.cost(X[t], U[t]).item()

        J += self.cost(X[-1], torch.zeros(self.n_u), terminal=True).item()
        L_derivs.append(
            self.quadraticize_cost(X[-1], torch.zeros(self.n_u), terminal=True)
        )

        return X, J, F_derivs, L_derivs

    def _backward_pass(self, F_derivs, L_derivs):
        """Backward pass to compute gain matrices K and d from the trajectory."""

        K = torch.zeros((self.N, self.n_u, self.n_x))  # linear feedback gain
        d = torch.zeros((self.N, self.n_u))  # feedforward gain

        reg = self.μ * torch.eye(self.n_x)

        L_x, _, L_xx, _, _ = L_derivs[-1]
        p = L_x
        P = L_xx

        for t in range(self.N - 1, -1, -1):
            L_x, L_u, L_xx, L_uu, L_ux = L_derivs[t]
            A, B = F_derivs[t]

            Q_x = L_x + A.T @ p
            Q_u = L_u + B.T @ p
            Q_xx = L_xx + A.T @ P @ A
            Q_uu = L_uu + B.T @ (P + reg) @ B
            Q_ux = L_ux + B.T @ (P + reg) @ A

            K[t] = -torch.linalg.solve(Q_uu, Q_ux)
            d[t] = -torch.linalg.solve(Q_uu, Q_u)

            p = Q_x + K[t].T @ Q_uu @ d[t] + K[t].T @ Q_u + Q_ux.T @ d[t]
            P = Q_xx + K[t].T @ Q_uu @ K[t] + K[t].T @ Q_ux + Q_ux.T @ K[t]
            P = 0.5 * (P + P.T)

        return K, d

    def _control(self, X, U, K, d, α):
        """Apply the control gains to obtain a new trajectory"""

        X_next = torch.zeros((self.N + 1, self.n_x))
        U_next = torch.zeros((self.N, self.n_u))

        J = 0.0
        X_next[0] = X[0].clone()

        for t in range(self.N):
            U_next[t] = U[t] + α * d[t] + K[t] @ (X_next[t] - X[t])
            X_next[t + 1] = self.integrate_dynamics(X_next[t], U_next[t])
            J += self.cost(X_next[t], U_next[t]).item()

        J += self.cost(X_next[-1], torch.zeros(self.n_u), terminal=True).item()

        return X_next, U_next, J

    def _reset_regularization(self):
        """Reset regularization terms to their factory defaults."""
        self.μ = 1.0  # regularization
        self.Δ = self.DELTA_0  # regularization scaling

    def _decrease_regularization(self):
        """Decrease regularization to converge more slowly."""
        self.Δ = min(1.0, self.Δ) / self.DELTA_0
        self.μ *= self.Δ
        if self.μ <= self.MU_MIN:
            self.μ = 0.0

    def _increase_regularization(self):
        """Increase regularization to go a different direction"""
        self.Δ = max(1.0, self.Δ) * self.DELTA_0
        self.μ = max(self.MU_MIN, self.μ * self.Δ)
