import numpy as np
from abc import ABC, abstractmethod


class ABCLayer(ABC):
    def __init__(self):
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.gradW: np.ndarray | None = None
        self.gradb: np.ndarray | None = None

        self.learning = True
        self.inputs: np.ndarray | None = None
        self.outputs: np.ndarray | None = None

    @abstractmethod
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_weights(self) -> tuple[np.ndarray, np.ndarray | None]:
         return self.W, self.b

    def get_gradients(self) -> tuple[np.ndarray, np.ndarray | None]:
        return self.gradW, self.gradb
    
    def update_weights(self, W: np.ndarray, b: np.ndarray | None):
        self.W, self.b = W, b
    
    @abstractmethod
    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def train(self):
        self.learning = True

    def eval(self):
        self.learning = False
