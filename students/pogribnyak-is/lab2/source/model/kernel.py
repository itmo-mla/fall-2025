import numpy as np
from abc import ABC, abstractmethod

class Kernel(ABC):
    @abstractmethod
    def __call__(self, distances: np.ndarray, h: float = 1.0) -> np.ndarray:
        pass

class GaussianKernel(Kernel):
    def __init__(self, c: float = 0.5):
        self.c = c

    def __call__(self, distances: np.ndarray, h: float = 1.0) -> np.ndarray:
        return np.exp(-self.c * (distances / h) ** 2)
