import numpy as np
from scipy.optimize import minimize
from kernels import KERNELS


class CustomSVM:
    def __init__(self, C=1.0, kernel='linear', kernel_params=None):
        self.C = C
        self.kernel_name = kernel
        self.kernel_params = kernel_params or {}
        self.kernel = KERNELS[kernel]
        self.lambdas = None
        self.sv_idx = None
        self.w = None
        self.w0 = None
        self.X_sv = None
        self.y_sv = None

    def _compute_kernel_matrix(self, X):
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(X[i], X[j], **self.kernel_params)
        return K

    def _objective(self, lambdas, K, y):
        return 0.5 * np.sum(np.outer(lambdas, lambdas) * np.outer(y, y) * K) - np.sum(lambdas)

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        n_samples = X.shape[0]

        K = self._compute_kernel_matrix(X)

        obj = lambda lam: self._objective(lam, K, y)

        constraints = {'type': 'eq', 'fun': lambda lam: np.dot(lam, y)}
        bounds = [(0, self.C) for _ in range(n_samples)]

        res = minimize(obj, np.zeros(n_samples), method='SLSQP', bounds=bounds, constraints=constraints)
        self.lambdas = res.x

        # Опорные векторы
        self.sv_idx = np.where(self.lambdas > 1e-6)[0]
        self.X_sv = X[self.sv_idx]
        self.y_sv = y[self.sv_idx]
        lambdas_sv = self.lambdas[self.sv_idx]

        margin_sv = (self.lambdas > 1e-6) & (self.lambdas < self.C - 1e-6)
        if np.any(margin_sv):
            idx = np.where(margin_sv)[0][0]
            K_row = np.array([self.kernel(X[idx], x_sv, **self.kernel_params) for x_sv in self.X_sv])
            self.w0 = self.y_sv[np.where(self.sv_idx == idx)[0][0]] - np.sum(lambdas_sv * self.y_sv * K_row)
        else:
            w0_vals = []
            for i, idx in enumerate(self.sv_idx):
                K_row = np.array([self.kernel(X[idx], x_sv, **self.kernel_params) for x_sv in self.X_sv])
                w0_vals.append(y[idx] - np.sum(lambdas_sv * self.y_sv * K_row))
            self.w0 = np.mean(w0_vals)

        if self.kernel_name == 'linear':
            self.w = np.sum((self.lambdas[self.sv_idx] * self.y_sv)[:, None] * self.X_sv, axis=0)

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            K_vals = np.array([self.kernel(x, x_sv, **self.kernel_params) for x_sv in self.X_sv])
            decision = np.sum(self.lambdas[self.sv_idx] * self.y_sv * K_vals) - self.w0
            predictions.append(np.sign(decision))
        return np.array(predictions)