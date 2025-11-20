import numpy as np
from abc import ABC, abstractmethod


class ABCRegularizer(ABC):
    def __init__(self, lambda_q: float = 0.1):
        self.lambda_q = lambda_q

    @abstractmethod
    def __call__(self, weights: np.ndarray, losses: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def pd_wrt_w(self, lr: float, weights: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
