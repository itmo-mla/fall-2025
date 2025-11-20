import numpy as np

from .abc_loss import ABCLoss
from source.layers import ABCLayer
from source.activations import ABCActivation


class PerceptronLoss(ABCLoss):
    """
    Функция ошибки персептрона или кусочно-линейная функция потерь (Hebb’s rule).
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.learning: self.y_true = y_true.copy()
        return np.mean(np.maximum(0, -y_true*y_pred))
    
    def pd_wrt_inputs(self, inputs: np.ndarray) -> np.ndarray:
        self.dL_dI = np.where(self.y_true*inputs < 0, -self.y_true, 0.)
        return self.dL_dI
