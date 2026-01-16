import numpy as np
from scipy.optimize import minimize
from .utils import linear_kernel, polynomial_kernel, rbf_kernel

class OwnSVM:
    def __init__(
            self, 
            kernel="linear", 
            C=1.0, 
            degree=3, 
            gamma=None
    ):
        self.kernel_name = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma

        if kernel == "linear":
            self.kernel = lambda x, y: linear_kernel(x, y)
        elif kernel == "poly":
            self.kernel = lambda x, y: polynomial_kernel(x, y, degree)
        elif kernel == "rbf":
            self.kernel = lambda x, y: rbf_kernel(x, y, gamma)

    def compute_kernel_matrix(self, X):
        n = X.shape[0]
        return np.array([
            [self.kernel(X[i], X[j]) for j in range(n)]
            for i in range(n)
        ])

    def objective(self, lambdas, y):
        return 0.5 * np.sum((lambdas[:, None] * lambdas[None, :]) * 
                            (y[:, None] * y[None, :]) * self.K) - np.sum(lambdas)

    def equality_constraint(self, lambdas, y):
        return np.dot(lambdas, y)

    def fit(self, X, y):
        # Получим размер выборки
        n_samples, _ = X.shape
        # Применим ядро к нашему датасету
        self.K = self.compute_kernel_matrix(X)

        # Решим двойственную задачу
        initial = np.zeros(n_samples)
        bounds = [(0, self.C) for _ in range(n_samples)]
        constraints = {"type": "eq", "fun": self.equality_constraint, "args": (y,)}
        solution = minimize(
            self.objective, 
            initial, 
            args=(y,),
            bounds=bounds, 
            constraints=constraints
        )

        # Получим решение
        self.lambdas = solution.x
        sv_mask = self.lambdas > 1e-5
        self.support_vectors = X[sv_mask]
        self.support_lambdas = self.lambdas[sv_mask]
        self.support_y = y[sv_mask]

        # Посчитаем условие для сдвига
        K_sv = np.array([
            [self.kernel(sv_i, sv_j) for sv_j in self.support_vectors]
            for sv_i in self.support_vectors
        ])
        self.b = np.mean(
            self.support_y
            - K_sv @ (self.support_lambdas * self.support_y)
        )

        return self

    def project(self, X):
        K = np.array([[self.kernel(x, sv) for sv in self.support_vectors] for x in X])
        return (K @ (self.support_lambdas * self.support_y)) + self.b

    def predict(self, X):
        return np.sign(self.project(X))