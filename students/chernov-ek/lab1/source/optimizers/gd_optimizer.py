import numpy as np

from .abc_optimizer import ABCOptimizer
from source.regularizers import ABCRegulatizer
from source.data_loaders import ABCLoader
from source.layers import ABCLayer
from source.activations import ABCActivation


class GDOptimizer(ABCOptimizer):
    def __init__(
            self,
            model_layers: list[ABCLayer],
            data_loader: ABCLoader | None = None,
            regularizer: ABCRegulatizer | None = None,
            lr: float = 0.001
        ):
        super().__init__(model_layers, data_loader)

        self.regularizer = regularizer
        self.lr = lr

    def step(self):
        for layer in reversed(self.model_layers):
            weights: np.ndarray = layer.get_weights()
            gradients: np.ndarray = layer.get_gradients()
            weights -= self.lr*gradients  # TODO: add regularizer
            layer.update_weights(weights)
