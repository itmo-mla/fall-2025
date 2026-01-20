import numpy as np

from source.losses import ABCLoss


class BCELoss(ABCLoss):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        
        self.eps = eps

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.learning: self.y_true = y_true.copy()
        return np.mean(-y_true*np.log(y_pred + self.eps))
    
    def backward_pass(self, inputs: np.ndarray) -> np.ndarray:
        return -(self.y_true/(inputs + self.eps)) + ((1 - self.y_true)/(1 - (inputs + self.eps)))
