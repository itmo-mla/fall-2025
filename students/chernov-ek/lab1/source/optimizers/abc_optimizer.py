from abc import ABC, abstractmethod

from source.data_loaders import ABCLoader, BaseLoader
from source.layers import ABCLayer


class ABCOptimizer(ABC):
    def __init__(self, model_layers: list[ABCLayer], data_loader: ABCLoader | None = None):
        self.model_layers = model_layers
        self.data_loader = data_loader if data_loader is not None else BaseLoader()

    @abstractmethod
    def step(self):
        raise NotImplementedError()
