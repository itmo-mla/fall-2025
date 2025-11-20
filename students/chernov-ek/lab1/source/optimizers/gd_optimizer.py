import numpy as np

from .abc_optimizer import ABCOptimizer
from source.regularizers import ABCRegularizer
from source.data_loaders import ABCLoader
from source.layers import ABCLayer


class GDOptimizer(ABCOptimizer):
    def __init__(
            self,
            model_layers: list[ABCLayer],
            data_loader: ABCLoader | None = None,
            lr: float = 0.001
        ):
        super().__init__(model_layers, data_loader, lr)

    def step(self, regularizer: ABCRegularizer | None = None):
        for layer in reversed(self.model_layers):
            weights: np.ndarray = layer.get_weights()
            gradients: np.ndarray = layer.get_gradients()
            weights -= regularizer.pd_wrt_w(self.lr, weights, gradients) if regularizer else self.lr*gradients
            layer.update_weights(weights)
