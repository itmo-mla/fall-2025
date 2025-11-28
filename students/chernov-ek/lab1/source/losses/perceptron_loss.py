import numpy as np

from source.losses import ABCLoss


class PerceptronLoss(ABCLoss):
    """
    Функция ошибки персептрона или кусочно-линейная функция потерь (Hebb’s rule).
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.learning: self.y_true = y_true.copy()
        return np.mean(np.maximum(0, -y_true*y_pred))
    
    def backward_pass(self, inputs: np.ndarray) -> np.ndarray:
        return np.where(self.y_true*inputs < 0, -self.y_true, 0.)
