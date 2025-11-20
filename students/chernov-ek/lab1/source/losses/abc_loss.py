import numpy as np
from abc import ABC, abstractmethod


from source.layers import ABCLayer
from source.activations import ABCActivation


class ABCLoss(ABC):
    def __init__(self):
        self.learning: bool = True

        self.y_true: np.ndarray | None = None
        self.dL_dI: np.ndarray | None = None

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
    def pd_wrt_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """
        Частная производная функции ошибки по входам.

        :param inputs: Входы функции потерь.
        :type inputs: np.ndarray

        :return: Приращение функции ошибки по входам.
        :rtype: np.ndarray
        """
        raise NotImplementedError()
    
    def train(self):
        self.learning = True

    def eval(self):
        self.learning = False
