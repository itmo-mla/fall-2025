from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class SVM:
    def __init__(self, kernel, C: float = 1.0):
        self.kernel = kernel
        self.C = C
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        unique = np.unique(y)
        if set(unique.tolist()) == {0.0, 1.0}:
            y = 2.0 * y - 1.0

        self.X_train = X
        self.y_train = y

        K = self.kernel.transform(X)
        yyK = np.outer(y, y) * K

        def function_to_minimize(lambd):
            return 0.5 * lambd @ (yyK @ lambd) - np.sum(lambd)

        n_samples = len(y)
        bounds = [(0.0, self.C) for _ in range(n_samples)]
        constraints = {
            "type": "eq",
            "fun": lambda lambd, y=y: np.dot(lambd, y),
        }

        res = minimize(
            function_to_minimize,
            np.zeros(n_samples),
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            tol=1e-3,
            options={"maxiter": 1000},
        )

        alpha_full = res.x

        support_indices = alpha_full > 1e-6
        self.alpha = alpha_full[support_indices]
        self.support_vectors = X[support_indices]
        self.support_vector_labels = y[support_indices]

        K_sv = self.kernel.transform(self.support_vectors)
        Ay = self.alpha * self.support_vector_labels
        decision_on_sv = K_sv.T @ Ay
        self.b = np.mean(self.support_vector_labels - decision_on_sv)

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        K = self.kernel.transform(X, self.support_vectors)
        Ay = self.alpha * self.support_vector_labels
        return K @ Ay + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores > 0.0).astype(int)



