import numpy as np

from source.optimizers import ABCOptimizer
from source.regularizers import ABCRegularizer
from source.data_loaders import ABCLoader
from source.layers import ABCLayer


class GDOptimizer(ABCOptimizer):
    def __init__(
            self,
            model_weights_layers: list[ABCLayer],
            data_loader: ABCLoader | None = None,
            lr: float = 0.001,
            momentum: float = 0.,
            dampening: float = 0.
        ):
        super().__init__(model_weights_layers, data_loader, lr)

        self.momentum = momentum
        self.dampening = dampening

        # Для каждого слоя создаём буферы импульса
        self.momentum_buffers = [None] * len(self.model_weights_layers)

    def step(self, regularizer: ABCRegularizer | None = None):
        for i, layer in enumerate(self.model_weights_layers):
            W, b = layer.get_weights()
            gradW, gradb = layer.get_gradients()

            # Применение регуляризатора
            if regularizer is not None:
                reg_W, reg_b = regularizer.pd_wrt_w((W, b))
                gradW = gradW + reg_W
                if b is not None:
                    gradb = gradb + reg_b
            
            # Применение метода импульсов
            if self.momentum:
                # Инициализация буфера
                if self.momentum_buffers[i] is None:
                    # Буфер хранит momentum с прошлой эпохи
                    vW = np.zeros_like(W)
                    vb = np.zeros_like(b) if b is not None else None
                    self.momentum_buffers[i] = (vW, vb)

                vW, vb = self.momentum_buffers[i]
                # Экспоненциальное скользящее среднее по градиентам
                # v = momentum*v + (1 - dampening)*grad
                vW = self.momentum*vW + (1 - self.dampening)*gradW
                if b is not None:
                    vb = self.momentum*vb + (1 - self.dampening)*gradb

                self.momentum_buffers[i] = (vW, vb)

                gradW = vW
                if b is not None:
                    gradb = vb

            # Шаг оптимизации
            W -= self.lr*gradW
            if b is not None:
                b -= self.lr*gradb

            layer.update_weights(W, b)
