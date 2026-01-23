import numpy as np
from scipy.optimize import minimize


class Kernel:
    def compute(self, X1, X2):
        pass


class LinearKernel(Kernel):
    def compute(self, X1, X2):
        return np.dot(X1, X2.T)


class QuadraticKernel(Kernel):
    def compute(self, X1, X2):
        return np.dot(X1, X2.T) ** 2


class RBFKernel(Kernel):
    def __init__(self, gamma=0.1):
        self.gamma = gamma

    def compute(self, X1, X2):
        diff = X1[:, np.newaxis] - X2
        sq_dist = np.sum(diff**2, axis=2)
        return np.exp(-self.gamma * sq_dist)


class SVMClassifier:
    def __init__(
        self, kernel=LinearKernel(), C=1.0, max_iter=1000, tol=1e-3, threshold=1e-4
    ):
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.threshold = threshold
        self.support_vectors = None
        self.support_vector_labels = None
        self.lambdas = None
        self.w0 = 0.0
        self.is_fitted = False

    def _objective(self, alpha, X, y):
        K = self.kernel.compute(X, X)
        return -np.sum(alpha) + 0.5 * np.sum(np.outer(alpha * y, alpha * y) * K)

    def _constraints(self, alpha):
        return np.sum(alpha * self._y_constraint)

    def fit(self, X, y):
        n_samples = X.shape[0]
        self._y_constraint = y

        initial_alpha = np.zeros(n_samples)
        constraints = {"type": "eq", "fun": self._constraints}
        bounds = [(0, self.C) for _ in range(n_samples)]

        result = minimize(
            self._objective,
            initial_alpha,
            args=(X, y),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        sv_mask = result.x > self.threshold

        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]
        self.lambdas = result.x[sv_mask]

        K_sv = self.kernel.compute(self.support_vectors, self.support_vectors)
        w_dot_x = np.dot(K_sv, self.lambdas * self.support_vector_labels)

        boundary_idx = self.lambdas < (self.C - 1e-5)
        self.w0 = np.mean(
            self.support_vector_labels[boundary_idx] - w_dot_x[boundary_idx]
        )

    def predict(self, X):
        K = self.kernel.compute(self.support_vectors, X)
        decision_values = (
            np.sum(
                self.lambdas[:, np.newaxis]
                * self.support_vector_labels[:, np.newaxis]
                * K,
                axis=0,
            )
            + self.w0
        )

        return np.sign(decision_values)
