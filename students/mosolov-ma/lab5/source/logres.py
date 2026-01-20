import numpy as np


class LogisticRegression:
    def __init__(self, lr=1, n_iter=100, tol=0.0001):
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.weights = None
        self.coef_ = None
        self.intercept_ = None

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.weights = np.linalg.lstsq(X, y, rcond=None)[0]

        mu_prev = None

        for _ in range(self.n_iter):
            mu = self._sigmoid(y * (X @ self.weights))

            if mu_prev is not None and (np.linalg.norm(mu - mu_prev) < self.tol):
                break
            
            mu_prev = mu.copy()

            W = np.diag(np.sqrt(mu * (1. - mu)))
            y_head = y * np.sqrt((1. / mu) - 1.)

            X_head = W @ X

            A = X_head.T @ X_head
            b = X_head.T @ y_head

            self.weights += self.lr * np.linalg.solve(A, b)

        self.coef_ = self.weights[1:]
        self.intercept_ = self.weights[0]

    def predict_proba(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return self._sigmoid(X @ self.weights)
    
    def predict(self, X):
        return (self.predict_proba(X) > 0.5) * 2 - 1
