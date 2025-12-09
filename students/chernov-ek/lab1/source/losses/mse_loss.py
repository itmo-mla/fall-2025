import numpy as np

from source.losses import ABCLoss


class MSELoss(ABCLoss):
    """
    Квадратичная функция потерь.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # if self.learning: self.y_true = y_true.copy()
        # return np.mean(np.maximum(0, -y_true*y_pred))
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


# OLD
class MSE(ABCLoss):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        self.losses = np.array([np.mean((y_true - y_pred)**2)])
        return self.losses
    
    def partial_derivative_wrt_a(self, i_neuron: int, input: float) -> float:
        n = len(self.losses)
        return 2/n*(input - self.losses[i_neuron])
