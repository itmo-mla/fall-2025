import numpy as np


class Kernel:
    def compute(self, X1, X2):
        pass


class LinearKernel(Kernel):
    def compute(self, X1, X2):
        return np.dot(X1, X2.T)


class PolynomialKernel(Kernel):
    """
    Полиномиальное ядро: K(x, y) = (γ * x · y + coef0)^d
    """

    def __init__(self, degree=3, gamma=1.0, coef0=1.0):
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def compute(self, X1, X2):
        return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree


class RBFKernel(Kernel):
    """
    Радиально-базисное ядро (Гауссово): K(x, y) = exp(-γ||x - y||²)
    """

    def __init__(self, gamma=0.1):
        self.gamma = gamma

    def compute(self, X1, X2):
        # ||x - y||² = ||x||² + ||y||² - 2*x·y
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        sq_dist = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * sq_dist)
