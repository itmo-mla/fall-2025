import numpy as np
from abc import ABC, abstractmethod


class ABCRegularizer(ABC):
    def __init__(self, lambda_q: float = 0.1, bias_regularizer: bool = False):
        self.lambda_q = lambda_q
        self.bias_regularizer = bias_regularizer

    @abstractmethod
    def __call__(self, weights: tuple[np.ndarray, np.ndarray | None]) -> np.ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def pd_wrt_w(
        self,
        weights: tuple[np.ndarray, np.ndarray | None]
        ) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError()
