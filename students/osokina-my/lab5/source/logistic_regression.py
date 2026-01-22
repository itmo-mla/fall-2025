import numpy as np

from .linear_regression import solve_system

EPS = 1e-10
HESSIAN_REG = 1e-6


class LogisticRegression:
    def __init__(self, solver="nr", max_iter=200, tol=1e-3):
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None

    @staticmethod
    def sigmoid(z):
        z = np.asarray(z)
        out = np.empty_like(z, dtype=float)
        pos_mask = z >= 0
        neg_mask = ~pos_mask
        out[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
        exp_z = np.exp(z[neg_mask])
        out[neg_mask] = exp_z / (1 + exp_z)
        return out

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.weights = np.zeros(X.shape[1])

        solvers = {
            "nr": self.fit_newton_raphson,
            "irls": self.fit_irls,
        }
        try:
            solvers[self.solver](X, y)
        except KeyError as exc:
            raise ValueError(f"Unknown solver: {self.solver}") from exc

    def fit_newton_raphson(self, X, y):
        prev_weights = np.zeros_like(self.weights)
        for _ in range(self.max_iter):
            z = X @ self.weights
            prob = LogisticRegression.sigmoid(z)
            gradient = X.T @ (prob - (y + 1) / 2)
            weights_diag = np.diag(prob * (1 - prob))
            hessian = X.T @ weights_diag @ X
            hessian += HESSIAN_REG * np.eye(hessian.shape[0])
            self.weights -= np.linalg.inv(hessian) @ gradient
            upd_norm = np.linalg.norm(self.weights - prev_weights)
            if upd_norm < self.tol:
                break
            prev_weights = self.weights.copy()

    def fit_irls(self, X, y):
        X_orig = X.copy()
        y_orig = y.copy()
        prev_sigm = np.zeros_like(y)
        for _ in range(self.max_iter):
            sigm = LogisticRegression.sigmoid(X_orig @ self.weights * y_orig)
            gamma = np.sqrt(sigm * (1 - sigm))
            X_weighted = np.diag(gamma) @ X_orig
            y_weighted = y_orig * (gamma / (sigm + EPS))
            self.weights = solve_system(X_weighted, y_weighted)
            upd_norm = np.linalg.norm(sigm - prev_sigm)
            if upd_norm < self.tol:
                break
            prev_sigm = sigm

    def predict_proba(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return self.sigmoid(X @ self.weights)

    def predict(self, X, threshold=0.5):
        return np.where(self.predict_proba(X) >= threshold, 1, -1)
