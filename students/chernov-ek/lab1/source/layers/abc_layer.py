import numpy as np
from abc import ABC, abstractmethod


class ABCLayer(ABC):
    def __init__(self):
        self.W: np.ndarray | None = None
        self.gradients: np.ndarray | None = None

        self.learning = True
        self.inputs: np.ndarray | None = None
        self.outputs: np.ndarray | None = None

    @abstractmethod
    def get_weights(self) -> np.ndarray:
         raise NotImplementedError()

    @abstractmethod
    def get_size(self) -> tuple[int]:
        raise NotImplementedError()

    @abstractmethod
    def get_gradients(self) -> np.ndarray | None:
        raise NotImplementedError()
    
    @abstractmethod
    def update_weights(self, weights: np.ndarray):
        raise NotImplementedError()
    
    def train(self):
        self.learning = True

    def eval(self):
        self.learning = False
