import numpy as np
from typing import Callable

from .abc_model import ABCModel
from source.layers import LinearLayer
from source.activations import SignActivation
from source.losses import ABCLoss
from source.optimizers import ABCOptimizer


class LinearClassificator(ABCModel):
    def __init__(self, in_features: int):
        super().__init__([
            LinearLayer(in_features, 1, True),
            SignActivation()
        ])

        self.X = None
    
    def __call__(self, X: np.ndarray, postprocess: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
        outputs = X
        for layer in self.arch_model:
            outputs = layer(outputs)

        return outputs
    
    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        self.X = X.copy()
        return self.__call__(self.X)

    def backward_pass(self, loss: ABCLoss):
        # Считаем частные производные Loss функции по весам
        dL_dA = loss.pd_wrt_a(self.arch_model[1].A)
        # Частная производная функции активации по выходу ф-и линейной трансформации
        dA_dZ = self.arch_model[1].pd_wrt_z(self.arch_model[0].Z)
        # Считаем частные производные ф-и линейной трансформации
        dZ_dW = self.arch_model[0].pd_wrt_w()  # Если считаем по bias, то dZ_dW будет равна 1
        # Частная производная Loss по весу нейрона: dL_dW = dL_dA*dA_dZ*dZ_dW
        gradients = (dL_dA*dA_dZ).T@dZ_dW / len(self.X)
        # Сохраняем градиент для весов
        self.arch_model[0].gradients = gradients

    def train(
            self,
            n_epochs: int,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray | None,
            y_val: np.ndarray | None,
            loss: ABCLoss,
            optimizer: ABCOptimizer,
            postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
            count_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
            verbose_n_batch_multiple: int = 1
        ) -> tuple[np.ndarray, np.ndarray]:
        losses_epochs = []
        metrics_epochs = []
        for n_epoch in range(n_epochs):
            losses = []
            metrics = []
            for n_batch, X_train_batch, y_train_batch in optimizer.data_loader.get_data(X_train, y_train):
                # forward pass
                y_pred_batch = self.forward_pass(X_train_batch)
                loss_batch = loss(y_train_batch, y_pred_batch)

                # backward pass
                self.backward_pass(loss)
                optimizer.step()

                if X_val is not None and verbose_n_batch_multiple and n_batch%verbose_n_batch_multiple == 0:
                    losses.append(loss_batch)
                    y_pred = self.__call__(X_val)
                    metrics.append(count_metric(y_val, y_pred))
                    print(f"Epoch {n_epoch + 1} ({n_batch*optimizer.data_loader.batch_size}/{len(y_train)}): \
                        {loss.to_str()} = {round(np.mean(losses), 3)} \
                        {count_metric.__name__} = {round(np.mean(metrics), 3)}")
                    
            losses_epochs.append(np.mean(losses))
            metrics_epochs.append(np.mean(metrics))

        return np.array(losses_epochs), np.array(metrics_epochs)
