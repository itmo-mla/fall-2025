import numpy as np

from .abc_loss import ABCLoss


class HingeLoss(ABCLoss):
    """
    Шарнирная функция ошибки.
    """
    def __call__(self, y_true: np.ndarray, Z: np.ndarray) -> np.float64:
        if self.learning: self.y_true = y_true.copy()
        return np.mean(np.maximum(0, 1 - y_true*Z))
    
    def pd_wrt_w(self, X: np.ndarray) -> tuple[np.ndarray, np.float64]:
        """
        Partial derivative with recpect to weights.

        :param X: Признаки объекта.
        :type X: np.ndarray

        :return: Приращение функции ошибки по весам и по смещению с отрицательным знаком.
        :rtype: tuple[np.ndarray, np.float64]
        """
        return -np.mean(X*self.y_true, axis=0), -np.mean(self.y_true)
