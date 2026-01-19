import numpy as np
from .linear_regression import matrix_inverse, solve_linear_system


class LogisticRegression:
    def __init__(self, solver='nr', max_iter=200, tol=1e-3):
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.weights = np.zeros(X.shape[1])
        
        if self.solver == 'nr':
            self._fit_newton_raphson(X, y)
        elif self.solver == 'irls':
            self._fit_irls(X, y)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

    def _fit_newton_raphson(self, X, y):
        prev_weights = np.zeros_like(self.weights)
        for i in range(self.max_iter):
            z = X @ self.weights
            p = LogisticRegression._sigmoid(z)
            gradient = X.T @ (p - (y + 1) / 2)
            D = np.diag(p * (1 - p))
            hessian = X.T @ D @ X
            hessian += 1e-6 * np.eye(hessian.shape[0])
            self.weights -= np.linalg.inv(hessian) @ gradient
            upd_norm = np.linalg.norm(self.weights - prev_weights)
            if upd_norm < self.tol:
                break
            prev_weights = self.weights.copy()

    def _fit_irls(self, X, y):
        X_orig = X.copy()
        y_orig = y.copy()
        prev_sigm = np.zeros_like(y)
        for i in range(self.max_iter):
            sigm = LogisticRegression._sigmoid(X_orig @ self.weights * y_orig)
            gamma = np.sqrt(sigm * (1 - sigm))
            X_weighted = np.diag(gamma) @ X_orig
            y_weighted = y_orig * (gamma / (sigm + 1e-10))
            self.weights = solve_linear_system(X_weighted, y_weighted)
            upd_norm = np.linalg.norm(sigm - prev_sigm)
            if upd_norm < self.tol:
                break
            prev_sigm = sigm

    def predict_proba(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return self._sigmoid(X @ self.weights)

    def predict(self, X, threshold=0.5):
        return np.where(self.predict_proba(X) >= threshold, 1, -1)
