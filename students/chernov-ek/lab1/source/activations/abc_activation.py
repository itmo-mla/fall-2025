import numpy as np
from abc import ABC, abstractmethod


class ABCActivation(ABC):
    def __init__(self):
        self.A: np.ndarray | None = None
        self.dA_dZ: np.ndarray | None = None

    @abstractmethod
    def __call__(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def pd_wrt_z(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
