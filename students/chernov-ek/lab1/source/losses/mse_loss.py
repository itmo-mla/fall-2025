import numpy as np

from .abc_loss import ABCLoss


class MSELoss(ABCLoss):
    """
    Квадратичная функция потерь.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # self.y_true = y_true.copy()
        # return np.mean(np.maximum(0, -self.y_true*y_pred))
        pass
    
    def pd_wrt_a(self, A: np.ndarray) -> np.ndarray:
        """
        Partial derivative with recpect to activation function.

        :param A: Признаки объекта.
        :type A: np.ndarray

        :return: Приращение функции ошибки по весам и по смещению с отрицательным знаком.
        :rtype: tuple[np.ndarray, np.float64]
        """
        # dL_dA = np.where(self.y_true*A < 0, -self.y_true, 0.)
        # return dL_dA
        pass
