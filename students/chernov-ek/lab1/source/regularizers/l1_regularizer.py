import numpy as np

from source.regularizers import ABCRegularizer


class L1Regularizer(ABCRegularizer):
    def __init__(self, lambda_q: float = 0.1, bias_regularizer: bool = False):
        super().__init__(lambda_q, bias_regularizer)

    def __call__(self, weights: tuple[np.ndarray, np.ndarray | None]) -> np.ndarray:
        W, b = weights
        sum_weights = np.sum(np.abs(W)) + np.sum(np.abs(b)) if self.bias_regularizer else np.sum(np.abs(W))
        return self.lambda_q*sum_weights
    
    def pd_wrt_w(
        self,
        weights: tuple[np.ndarray, np.ndarray | None]
        ) -> tuple[np.ndarray, np.ndarray | None]:
        W, b = weights
        reg_W = self.lambda_q * np.sign(W)

        reg_b = None
        if self.bias_regularizer:
            reg_b = self.lambda_q * np.sign(b)

        return reg_W, reg_b
