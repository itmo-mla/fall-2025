import numpy as np
from scipy.optimize import minimize

from utils import linear_kernel, poly_kernel, rbf_kernel


class CustomSVM:
    def __init__(
            self, 
            kernel_name="linear", 
            C=1.0, 
            degree=2, 
            gamma=0.5
        ):
        self.kernel = linear_kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
            
        if kernel_name == "poly":
            self.kernel = lambda x, y: poly_kernel(x, y, degree)
        elif kernel_name == "rbf":
            self.kernel = lambda x, y: rbf_kernel(x, y, gamma)

    def objective_function(self, l, y):
        """
        Целевая функция двойственной задачи
        """
        return 0.5*np.sum((l[:, None]*l[None, :])*(y[:, None]*y[None, :])*self.K) - np.sum(l)

    def fit(self, X, y):
        # Количество объектов
        n = X.shape[0]
        # Прогон объектов через ядро
        self.K = np.array([
            [self.kernel(X[i], X[j]) for j in range(n)]
            for i in range(n)
        ])

        # Решение двойственной задачи
        bounds = [(0, self.C) for _ in range(n)]
        constraints = {"type": "eq", "fun": np.dot, "args": (y,)}
        res = minimize(
            fun=self.objective_function,
            x0=np.zeros(n),
            args=(y,),
            bounds=bounds,
            constraints=constraints
        )

        # Сохраняем найденные параметры
        self.l = res.x
        mask = self.l > 1e-5
        self.support_vectors = X[mask]
        self.support_lambdas = self.l[mask]
        self.support_y = y[mask]

        # Считаем смещение
        K = np.array([
            [self.kernel(sv_i, sv_j)
             for sv_j in self.support_vectors]
             for sv_i in self.support_vectors
        ])
        self.b = np.mean(self.support_y - K@(self.support_lambdas*self.support_y))

        return self

    def predict(self, X):
        K = np.array([[self.kernel(x, sv) for sv in self.support_vectors] for x in X])
        return np.sign(K@(self.support_lambdas*self.support_y) + self.b)
