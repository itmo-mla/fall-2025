import numpy as np
from abc import ABC, abstractmethod


class ABCActivation(ABC):
    def __init__(self):
        self.learning = True

        self.inputs: np.ndarray | None = None 
        self.outputs: np.ndarray | None = None

    @abstractmethod
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def train(self):
        self.learning = True

    def eval(self):
        self.learning = False
