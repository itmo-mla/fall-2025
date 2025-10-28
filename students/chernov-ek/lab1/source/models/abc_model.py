import numpy as np
from abc import ABC, abstractmethod
from typing import Callable

from source.layers import ABCLayer
from source.activations import ABCActivation
from source.losses import ABCLoss
from source.optimizers import ABCOptimizer


class ABCModel(ABC):
    def __init__(self, arch_model: list[ABCLayer| ABCActivation]):
        self.arch_model = arch_model

    @abstractmethod
    def __call__(self, X: np.ndarray, postprocess: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
        raise NotImplementedError()
    
    def get_layers(self) -> list[ABCLayer]:
        return [struc_element for struc_element in self.arch_model if isinstance(struc_element, ABCLayer)]
    
    @abstractmethod
    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def backward_pass(self, loss: ABCLoss, prev_layer: ABCLayer):
        raise NotImplementedError()

    @abstractmethod
    def train(
            self,
            n_epochs: int,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            loss: ABCLoss,
            optimizer: ABCOptimizer,
            postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
            count_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
            verbose_n_batch_multiple: int = 1
        ):
        raise NotImplementedError()
