from source.optimizers import ABCOptimizer
from source.regularizers import ABCRegularizer
from source.data_loaders import ABCLoader
from source.layers import ABCLayer


class GDOptimizer(ABCOptimizer):
    def __init__(
            self,
            model_weights_layers: list[ABCLayer],
            data_loader: ABCLoader | None = None,
            lr: float = 0.001
        ):
        super().__init__(model_weights_layers, data_loader, lr)

    def step(self, regularizer: ABCRegularizer | None = None):
        for layer in self.model_weights_layers:
            W, b = layer.get_weights()
            gradW, gradb = layer.get_gradients()

            if b is not None:
                if regularizer is None:
                    b -= self.lr*b
                else:
                    b -= regularizer.pd_wrt_w(self.lr, b, gradb)

            if regularizer is None:
                W -= self.lr*gradW
            else:
                W -= regularizer.pd_wrt_w(self.lr, W, gradW)

            layer.update_weights(W, b)
