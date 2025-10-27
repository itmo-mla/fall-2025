import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import minimize


class Kernel(ABC):
    @abstractmethod
    def __call__(self, obj1: np.ndarray, obj2: np.ndarray) -> float:
        pass
    

class LinearKernel(Kernel):
    def __call__(self, obj1: np.ndarray, obj2: np.ndarray) -> float:
        return np.dot(obj1, obj2.T)
    

class SquaredKernel(Kernel):
    def __call__(self, obj1: np.ndarray, obj2: np.ndarray) -> float:
        return np.dot(obj1, obj2.T) ** 2


class RBFKernel(Kernel):
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma


    def __call__(self, obj1: np.ndarray, obj2: np.ndarray) -> float:
        diff = obj1[:, np.newaxis] - obj2
        sq_dist = np.sum(diff ** 2, axis=-1)
        return np.exp(-self.gamma * sq_dist)


class SVM:
    def __init__(self, C: float, kernel: Kernel):
        self.C = C
        self.kernel = kernel

    @staticmethod
    def _get_lagrangian(lambdas: np.ndarray, X: np.ndarray, y: np.ndarray, kernel: Kernel) -> float:
        return  - (lambdas.sum() - 0.5 * np.sum(np.outer(lambdas * y, lambdas * y) * kernel(X, X)))
    
    @staticmethod
    def _get_jac(lambdas: np.ndarray, X: np.ndarray, y: np.ndarray, kernel: Kernel) -> float:
        G = y[:, np.newaxis] * kernel(X, X) * y[:, np.newaxis].T
        return - (np.ones_like(lambdas) - np.dot(lambdas, G))

    def fit(self, X: np.ndarray, y: np.ndarray, maxiter: int = 1000, ftol: float = 1e-3, support_vector_threshold = 1e-4) -> tuple[np.ndarray, np.ndarray]:
        optmized_result = minimize(
            fun=self._get_lagrangian,
            x0=np.zeros(len(X)),
            args=(X, y, self.kernel),
            method="SLSQP",
            jac=self._get_jac,
            constraints={"type": "eq", "fun": lambda lambdas: np.sum(lambdas  * y)},
            bounds = [(0, self.C) for _ in range(len(X))],
            options={"maxiter": maxiter, "ftol": ftol}
        )

        if not optmized_result.success:
            raise ArithmeticError("Failed to optimize")

        lambdas = optmized_result.x

        self.support_vectors_mask = lambdas > support_vector_threshold
        self.support_vectors = X[self.support_vectors_mask]
        self.support_vectors_labels = y[self.support_vectors_mask]
        self.lambdas = lambdas[lambdas > support_vector_threshold]

        self.w = np.sum(lambdas[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)

        self.w0 = np.mean(
            np.sum(
                lambdas[self.support_vectors_mask, np.newaxis]
                * y[self.support_vectors_mask, np.newaxis]
                * self.kernel(X[self.support_vectors_mask], X[self.support_vectors_mask]),
                axis=0,
            )
            - y[self.support_vectors_mask, np.newaxis]
        )
        return lambdas, self.support_vectors_mask
    
    def predict(self, x: np.ndarray):
        return np.sign(np.sum(self.lambdas * self.support_vectors_labels * self.kernel(x, self.support_vectors)) - self.w0)

