import numpy as np


class LogisticRegressionNewtonRaphson:
    def __init__(self, max_iter=100, tol=1e-6, C=1.0):
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
        self.beta = None
        self.history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X = np.column_stack([np.ones(X.shape[0]), X])
        _, p = X.shape
        self.beta = np.zeros(p)

        lam = 1.0 / self.C

        for _ in range(self.max_iter):
            eta = X @ self.beta
            mu = self.sigmoid(eta)

            w = np.maximum(mu * (1 - mu), 1e-10)
            W = np.diag(w)

            reg_vec = lam * self.beta
            reg_vec[0] = 0

            gradient = X.T @ (y - mu) - reg_vec

            hessian = X.T @ W @ X
            hessian_reg = hessian + lam * np.eye(p)
            hessian_reg[0, 0] -= lam

            try:
                delta = np.linalg.solve(hessian_reg, gradient)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(hessian_reg, gradient, rcond=None)[0]

            self.beta += delta

            diff = np.linalg.norm(delta)
            self.history.append(diff)

            if diff < self.tol:
                break

        return self

    def predict_proba(self, X):
        X = np.column_stack([np.ones(X.shape[0]), X])
        return self.sigmoid(X @ self.beta)  # type: ignore

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class LogisticRegressionIRLS:
    def __init__(self, max_iter=100, tol=1e-6, C=1.0):
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
        self.beta = None
        self.history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X = np.column_stack([np.ones(X.shape[0]), X])
        _, p = X.shape
        self.beta = np.zeros(p)
        lam = 1.0 / self.C

        for _ in range(self.max_iter):
            eta = X @ self.beta
            mu = self.sigmoid(eta)

            w = np.maximum(mu * (1 - mu), 1e-10)
            W = np.diag(w)

            reg_vec = lam * self.beta
            reg_vec[0] = 0

            gradient = X.T @ (y - mu) - reg_vec

            hessian = X.T @ W @ X
            hessian_reg = hessian + lam * np.eye(p)
            hessian_reg[0, 0] -= lam

            try:
                delta = np.linalg.solve(hessian_reg, gradient)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(hessian_reg, gradient, rcond=None)[0]

            self.beta += delta

            diff = np.linalg.norm(delta)
            self.history.append(diff)

            if diff < self.tol:
                break

        return self

    def predict_proba(self, X):
        X = np.column_stack([np.ones(X.shape[0]), X])
        return self.sigmoid(X @ self.beta)  # type: ignore

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
