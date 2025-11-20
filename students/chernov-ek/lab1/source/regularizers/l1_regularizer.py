import numpy as np

from .abc_regularizer import ABCRegularizer


class L1Regularizer(ABCRegularizer):
    def __init__(self, lambda_q: float = 0.1):
        super().__init__(lambda_q)

    def __call__(self, weights: np.ndarray, losses: np.ndarray) -> np.ndarray:
        return losses + self.lambda_q*np.sum(np.abs(weights))
    
    def pd_wrt_w(self, lr: float, weights: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        return lr*self.lambda_q + lr*gradients
