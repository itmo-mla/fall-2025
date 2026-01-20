import numpy as np


class Regularizer:
    def penalty(self, w: np.ndarray) -> float: raise NotImplementedError

    def gradient(self, w: np.ndarray) -> np.ndarray: raise NotImplementedError


class L2Regularizer(Regularizer):
    def __init__(self, lambda_reg: float = 0.001):
        self.lambda_reg = lambda_reg

    def penalty(self, w): return self.lambda_reg * np.sum(w ** 2)

    def gradient(self, w): return 2 * self.lambda_reg * w

