import numpy as np
from abc import ABC, abstractmethod


class ABCLoss(ABC):
    def __init__(self):
        self.y_true: np.ndarray | None = None

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Подсчёт ошибки на истинных значениях и предсказанных.

        :param y_true: Истинные метки класса.
        :type y_true: np.ndarray
        :param y_pred: Предсказанные метки класса.
        :type y_pred: np.ndarray

        :return: Значение ошибки на объектах.
        :rtype: float
        """
        raise NotImplementedError()
    
    def to_str(self):
        return str(self).split('.')[-1].split()[0]
    
    @abstractmethod
    def pd_wrt_a(self, A: np.ndarray) -> np.ndarray:
        """
        Частная производная функции ошибки по входам.

        :param A: Входы функции потерь.
        :type A: np.ndarray

        :return: Приращение функции ошибки по входам.
        :rtype: np.ndarray
        """
        raise NotImplementedError()
