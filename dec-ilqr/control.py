# control.py
# Contains various implementations of LQR controllers including LQR and iLQR.

import abc

import numpy as np


class BaseController(abc.ABC):
    """Abstract base class for Optimal Control Problem solvers."""
    
    def __init__(self, dynamics, cost, N):
        assert N > 1
        
        self.dynamics = dynamics
        self.cost = cost
        self.N = N
    
    @property
    def n_x(self):
        return self.dynamics.n_x
    
    @property
    def n_u(self):
        return self.dynamics.n_u
    
    @property
    def dt(self):
        return self.dynamics.dt
    
    @property
    def xf(self):
        return self.cost.xf

    @abc.abstractmethod
    def run(self, x0):
        """Implements functionality to solve the OCP at the current state x0."""
        pass
    
    def rollout(self, x0, U):
        """Rollout the system from an initial state with a control sequence U."""
        
        X = np.zeros((self.N+1, self.n_x))
        X[0] = x0
        J = 0.0
        
        for t in range(self.N):
            X[t+1] = self.dynamics(X[t], U[t])
            J += self.cost(X[t], U[t])
        J += self.cost(X[-1], np.zeros(self.n_u), terminal=True)

        return X, J

    
class LQR(BaseController):
    """Linear Quadratic Regulator that assumes linear dynamics and quadratic costs."""
        
    @property
    def Q(self):
        return self.cost.Q
    
    @property
    def R(self):
        return self.cost.R
        
    def backward_pass(self, X, U):
        """Compute the optimal feedforward gain K."""
        
        K = np.zeros((self.N, self.n_u, self.n_x))
        P = np.zeros((self.N, self.n_x, self.n_x))
        P = self.Q.copy()
        
        for t in range(self.N-1, -1, -1):
            A, B = self.dynamics.linearize(X[t], U[t])
            # Feedforward gain [4] (30)
            K[t] = np.linalg.inv(self.R + B.T @ P @ B) @ B.T @ P @ A
            # Cost-to-go [4] (32)
            P = self.Q + (A.T @ P @ A) - A.T @ P.T @ B @ K[t]
        return K
    
    def forward_pass(self, X, U, K):
        """Apply the feedforward gain to compute the next trajectory."""
        
        X_next = np.zeros((self.N+1, self.n_x))
        U_next = np.zeros((self.N, self.n_u))
        
        X_next[0] = X[0]
        J = 0.0

        for t in range(self.N):
            U_next[t] = -K[t] @ (X_next[t] - self.xf)
            X_next[t+1] = self.dynamics(X_next[t], U_next[t])
            J += self.cost(X_next[t], U_next[t])
        J += self.cost(X_next[-1], np.zeros((self.n_u)), terminal=True)

        return X_next, U_next, J
    
    def run(self, x0, n_iter=10):
        """Solve the LQR OCP."""
        
        U = np.random.randn(self.N, self.n_u) * 1e-3
        X, J0 = self.rollout(x0, U)

        K = self.backward_pass(X, U)
        X, U, Jf = self.forward_pass(X, U, K)
        print(f'J0: {J0:.3g}\tJf: {Jf:.3g}')

        return X, U, Jf

    
class iLQR(BaseController):
    """iLQR solver."""
    
    DELTA_0 = 2.0 # initial regularization scaling
    MU_MIN = 1e-6 # regularization floor, below which is 0.0
    MU_MAX = 1e3 # regularization ceiling
    N_LS_ITER = 10 # number of line search iterations
    
    def __init__(self, dynamics, cost, N=10):
        super(iLQR, self).__init__(dynamics, cost, N)
        
        self.cost_hist = None
        self.μ = 1.0 # regularization
        self.Δ = self.DELTA_0 # regularization scaling
    
    def forward_pass(self, X, U, K, d, α):
        """Forward pass to rollout the control gains K and d."""
        
        X_next = np.zeros((self.N+1, self.n_x))
        U_next = np.zeros((self.N, self.n_u))

        X_next[0] = X[0].copy()
        J = 0.0

        for t in range(self.N):
            δx = X_next[t] - X[t]
            δu = K[t]@δx + α*d[t]

            U_next[t] = U[t] + δu
            X_next[t+1] = self.dynamics(X_next[t], U_next[t])
            
            J += self.cost(X_next[t], U_next[t])
        J += self.cost(X_next[-1], np.zeros((self.n_u)), terminal=True)

        return X_next, U_next, J
    
    def backward_pass(self, X, U):
        """Backward pass based on algorithm 1 [4]."""
        
        K = np.zeros((self.N, self.n_u, self.n_x)) # linear feedback gain
        d = np.zeros((self.N, self.n_u)) # feedforward gain

        # self.μ = 0.0 # DBG
        reg = self.μ * np.eye(self.n_x)
        
        # Initialize cost jacobian and hessian.
        L_x, _, L_xx, _, _ = self.cost.quadraticize(X[-1], np.zeros(self.n_u), terminal=True)
        p = L_x
        P = L_xx

        for t in range(self.N-1, -1, -1):
            L_x, L_u, L_xx, L_uu, L_ux = self.cost.quadraticize(X[t], U[t])
            A, B = self.dynamics.linearize(X[t], U[t])
            
            # if t in [self.N-1, 0]:
                # print(f'{t=}', P, p)
                # print(f'{t=}', L_x, L_u, L_xx, L_uu, L_ux)
                # print(f'{t=}', X[t], U[t], A, B)
                
            Q_x = L_x + A.T @ p
            Q_u = L_u + B.T @ p
            Q_xx = L_xx + A.T @ P @ A
            Q_uu = L_uu + B.T @ (P + reg) @ B
            Q_ux = L_ux + B.T @ (P + reg) @ A

            K[t] = -np.linalg.solve(Q_uu, Q_ux)
            d[t] = -np.linalg.solve(Q_uu, Q_u)
            
            p = Q_x + K[t].T @ Q_uu @ d[t] + K[t].T @ Q_u + Q_ux.T @ d[t]
            P = Q_xx + K[t].T @ Q_uu @ K[t] + K[t].T @ Q_ux + Q_ux.T @ K[t]
            P = 0.5 * (P + P.T) # to maintain symmetry
            
        return K, d
        
    def run(self, x0, n_lqr_iter=50, tol=1e-6):
        """Solve the OCP."""
        
        # Reset regularization terms.
        self.μ = 1.0
        self.Δ = self.DELTA_0
        
        is_converged = False
        alphas = 1.1**(-np.arange(self.N_LS_ITER)**2)
        
        U = np.zeros((self.N, self.n_u))
        # U = np.random.uniform(-1, 1, (self.N, self.n_u))
        X, J_star = self.rollout(x0, U)
        
        self.cost_hist = [J_star]
        print(f'[run] 0/{n_lqr_iter}\tJ: {J_star:g}')        
        
        for i in range(n_lqr_iter):
            accept = False
            
            # if i == 1:
            #     print(X[:5], X[-5:], U[:5], U[-5:])
            #     break
            
            # Backward recurse to compute gain matrices.
            K, d = self.backward_pass(X, U)
            
            # if i == 1:
            #     print(K[:5], K[-5:], d[:5], d[-5:])
            #     break
            
            # Conduct a line search to find a satisfactory trajectory where we continually
            # decrease α. We're effectively getting closer to the linear approximation in
            # the LQR case.
            for α in alphas:
                X_next, U_next, J = self.forward_pass(X, U, K, d, α)
                
                # if i == 1:
                #     print(X_next[:5], X_next[-5:], U_next[:5], U_next[-5:])
                #     converged = True
                #     break
                
                if J < J_star:
                    if np.abs((J_star - J) / J_star) < tol:
                        is_converged = True
                        
                    X = X_next
                    U = U_next
                    J_star = J
                    self.cost_hist.append(J_star)
                    
                    # Decrease regularization to converge more slowly.
                    self.Δ = min(1.0, self.Δ) / self.DELTA_0
                    self.μ *= self.Δ
                    if self.μ <= self.MU_MIN:
                        self.μ = 0.0
                    
                    # print(f'[run] {α=}')
                    accept = True
                    break

            if not accept:
                # TEMP: Regularization is pointless for quadratic costs.
                print('[run] Failed line search.. giving up.')
                break

                # Increase regularization if we're not converging.
                print('[run] Failed line search.. increasing μ.')
                self.Δ = max(1.0, self.Δ) * self.DELTA_0
                self.μ = max(self.MU_MIN, self.μ*self.Δ)
                if self.μ >= self.MU_MAX:
                    print("[run] Exceeded max regularization term...")
                    break

            if is_converged:
                break
            
            print(f'[run] {i+1}/{n_lqr_iter}\tJ: {J_star:g}\tμ: {self.μ:g}\tΔ: {self.Δ:g}')

        return X, U, J
    
