import numpy as np

from .abc_loss import ABCLoss


class BCELoss(ABCLoss):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        
        self.eps = eps

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.learning: self.y_true = y_true.copy()
        return np.mean(-y_true*np.log(y_pred + self.eps))
    
    def pd_wrt_inputs(self, inputs: np.ndarray) -> np.ndarray:
        # Если бинарная классификация и выходной слой отдаёт 1 значение, то подгоняем формат
        self.dL_dI = -(self.y_true/inputs) + ((1 - self.y_true)/(1 - inputs))
        return self.dL_dI
