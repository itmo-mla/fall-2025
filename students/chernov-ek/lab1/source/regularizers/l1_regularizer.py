import numpy as np

from source.regularizers import ABCRegularizer


class L1Regularizer(ABCRegularizer):
    def __init__(self, lambda_q: float = 0.1):
        super().__init__(lambda_q)

    def __call__(self, weights: tuple[np.ndarray, np.ndarray | None], losses: np.ndarray) -> np.ndarray:
        W, b = weights
        sum_weights = np.sum(np.abs(W)) if b is None else np.sum(np.abs(W)) + np.sum(np.abs(b))
        return losses + self.lambda_q*sum_weights
    
    def pd_wrt_w(self, lr: float, weights: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        return lr*self.lambda_q + lr*gradients
