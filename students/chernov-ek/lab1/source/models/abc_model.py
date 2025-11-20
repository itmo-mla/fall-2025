import numpy as np
from abc import ABC, abstractmethod
from typing import Callable

from source.layers import ABCLayer
from source.activations import ABCActivation
from source.losses import ABCLoss
from source.optimizers import ABCOptimizer
from source.regularizers import ABCRegularizer


class ABCModel(ABC):
    def __init__(self, arch_model: list[ABCLayer| ABCActivation]):
        self.arch_model = arch_model

        self.learning = True
        self.losses_train = []
        self.losses_val = []
        self.metrics_val = []

    def __call__(self, X: np.ndarray, postprocess: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
        outputs = X
        for layer in self.arch_model:
            outputs = layer(outputs)

        return outputs if postprocess is None else postprocess(outputs)
    
    def _count_verbose(
            self,
            X_train, y_train, X_val, y_val,
            n_epoch, n_batch,
            losses_train_batch, losses_val_batch, metrics_val_batch, loss_batch,
            optimizer, loss, regularizer, postprocess, count_metric,
            verbose_n_batch_multiple, verbose_statistic
        ):
        if X_val is not None and verbose_n_batch_multiple and n_batch%verbose_n_batch_multiple == 0:
            # Отключаем сохранение параметров
            self.eval()
            loss.eval()

            # Val loss
            y_val_probs = self.__call__(X_val)
            loss_val = loss(y_val, y_val_probs)
            loss_val = regularizer(self.get_weights(), loss_val) if regularizer else loss_val
            # Metric
            y_val_preds = y_val_probs if postprocess is None else postprocess(y_val_probs)
            metrics_val_batch.append(count_metric(y_val, y_val_preds))

            if verbose_statistic == 'all':  # Полное сохранение
                losses_train_batch.append(loss_batch)
                losses_val_batch.append(loss_val)
            else:
                if len(losses_train_batch) == 0:
                    _, X_rand, y_rand = next(optimizer.data_loader.get_data(X_train, y_train))
                    y_rand_pred = self.__call__(X_rand)
                    loss_rand = loss(y_rand, y_rand_pred)
                    loss_rand = regularizer(self.get_weights(), loss_rand) if regularizer else loss_rand
                    losses_train_batch = [loss_rand]
                    losses_val_batch = [loss_rand]
                
                lambda_q = 0.1
                if verbose_statistic == 'SMA':
                    m = 10
                    lambda_q = 1/m
                elif verbose_statistic == 'EMA':
                    m = 10
                    lambda_q = 2/(m + 1)

                # Train loss
                losses_train_batch = [lambda_q*loss_batch + (1 - lambda_q)*losses_train_batch[0]]
                # Val loss
                losses_val_batch = [lambda_q*loss_val + (1 - lambda_q)*losses_val_batch[0]]

            # Visualize
            print(f"Epoch {n_epoch + 1} ({n_batch*optimizer.data_loader.batch_size}/{len(y_train)}):\t"
                    f"Train {loss.to_str()} = {round(np.mean(losses_train_batch), 3)}\t"
                    f"Val {loss.to_str()} = {round(np.mean(losses_train_batch), 3)}\t"
                    f"{count_metric.__name__} = {round(np.mean(metrics_val_batch), 3)}")
            
            # Включаем сохранение параметров
            self.train()
            loss.train()
        
        return losses_train_batch, losses_val_batch, metrics_val_batch
    
    def get_layers(self) -> list[ABCLayer]:
        return [struc_element for struc_element in self.arch_model if isinstance(struc_element, ABCLayer)]
    
    def get_weights(self) -> np.ndarray:
        return np.array([layer.get_weights() for layer in self.get_layers()])
    
    def backward_pass(self, loss: ABCLoss):
        # Инициализируем переменную хранящую слой, который стоит правее в прямом проходе
        next_layer = None
        for layer in reversed(self.arch_model):
            # Выходной слой
            if next_layer is None:
                # Считаем частные производные Loss функции по выходам последнего слоя
                delta = loss.pd_wrt_inputs(layer.outputs)
            else:  # Скрытый слой
                # Структурный слой
                if not isinstance(layer, ABCLayer):
                    # Считаем дельта правило
                    delta = delta*layer.pd_wrt_inputs()
                else:  # Слой весов
                    # Частная производная струтурного слоя по выходу слоя весов [n_samples, n_inputs]
                    dF_dI = next_layer.pd_wrt_inputs()
                    # Считаем частные производные слоя весов по весам [n_samples, n_inputs]
                    dZ_dW = layer.pd_wrt_w()
                    # Локальная ошибка для этого слоя [n_samples, n_outputs]
                    local_error = delta*dF_dI
                    # Частная производная Loss по весу нейрона: dL_dW = dL_dA*dF_dI*dZ_dW

                    gradients = local_error.T@dZ_dW / len(layer.inputs)

                    # Сохраняем градиент для весов
                    layer.gradients = gradients
                    # Передаём ошибку дальше влево через веса
                    delta = local_error@layer.pd_wrt_inputs()

            next_layer = layer

    def train_model(
            self,
            n_epochs: int,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray | None,
            y_val: np.ndarray | None,
            loss: ABCLoss,
            optimizer: ABCOptimizer,
            regularizer: ABCRegularizer | None = None,
            postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
            count_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
            verbose_n_batch_multiple: int = 1,
            verbose_statistic: str = 'all'
        ):
        for n_epoch in range(n_epochs):
            losses_train_batch = []
            losses_val_batch = []
            metrics_val_batch = []
            for n_batch, X_train_batch, y_train_batch in optimizer.data_loader.get_data(X_train, y_train):
                # forward pass
                y_pred_batch = self.__call__(X_train_batch)
                loss_batch = loss(y_train_batch, y_pred_batch)
                loss_batch = regularizer(self.get_weights(), loss_batch) if regularizer else loss_batch

                # backward pass
                self.backward_pass(loss)
                optimizer.step(regularizer)

                # verbose
                losses_train_batch, losses_val_batch, metrics_val_batch = self._count_verbose(
                    X_train, y_train, X_val, y_val,
                    n_epoch, n_batch,
                    losses_train_batch, losses_val_batch, metrics_val_batch, loss_batch,
                    optimizer, loss, regularizer, postprocess, count_metric,
                    verbose_n_batch_multiple, verbose_statistic
                )
                    
            self.losses_train.append(np.mean(losses_train_batch))
            self.losses_val.append(np.mean(losses_val_batch))
            self.metrics_val.append(np.mean(metrics_val_batch))

    def train(self):
        self.learning = True
        for layer in self.arch_model:
            layer.train()

    def eval(self):
        self.learning = False
        for layer in self.arch_model:
            layer.eval()
