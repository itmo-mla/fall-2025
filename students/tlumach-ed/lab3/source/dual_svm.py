import numpy as np
from scipy.optimize import minimize
from kernels import linear_kernel

class DualSVM:
    def __init__(self, C=1.0, kernel=linear_kernel):
        self.C = C
        self.kernel = kernel

    def fit(self, X, y):
        n_samples = X.shape[0]
        K = np.array([[self.kernel(X[i], X[j]) for j in range(n_samples)] for i in range(n_samples)])

        # Целевая функция (минимизация двойственной задачи)
        def objective(lambdas):
            return 0.5 * np.sum(np.outer(lambdas, lambdas) * np.outer(y, y) * K) - np.sum(lambdas)

        # Производная для ускорения сходимости
        def objective_grad(lambdas):
            return np.dot((np.outer(y, y) * K), lambdas) - np.ones_like(lambdas)

        constraints = {'type': 'eq', 'fun': lambda l: np.dot(l, y)}
        bounds = [(0, self.C) for _ in range(n_samples)]

        # Старт с малых положительных чисел, а не нуля
        initial_lambdas = np.full(n_samples, 1e-4)

        res = minimize(objective, initial_lambdas, jac=objective_grad,
                       bounds=bounds, constraints=constraints, tol=1e-6)

        self.lambdas = res.x

        # Опорные векторы (0 < λ_i < C)
        eps = 1e-5
        mask_sv = (self.lambdas > eps) & (self.lambdas < self.C - eps)
        self.X_sv = X[mask_sv]
        self.y_sv = y[mask_sv]
        self.lambdas_sv = self.lambdas[mask_sv]

        # Вычисление w и w0
        if self.kernel == linear_kernel:
            self.w = np.sum(self.lambdas[:, None] * y[:, None] * X, axis=0)
            self.w0 = np.mean(self.y_sv - np.dot(self.X_sv, self.w))
        else:
            self.w = None
            self.w0 = np.mean([
                self.y_sv[i] - np.sum(
                    self.lambdas_sv * self.y_sv *
                    np.array([self.kernel(self.X_sv[j], self.X_sv[i]) for j in range(len(self.lambdas_sv))])
                )
                for i in range(len(self.lambdas_sv))
            ])

    def predict(self, X):
        if self.kernel == linear_kernel:
            return np.sign(np.dot(X, self.w) + self.w0)
        else:
            return np.sign(np.array([
                np.sum(self.lambdas_sv * self.y_sv * np.array(
                    [self.kernel(self.X_sv[j], x) for j in range(len(self.lambdas_sv))]
                )) + self.w0
                for x in X
            ]))
