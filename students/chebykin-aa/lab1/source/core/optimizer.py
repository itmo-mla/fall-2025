from typing import Union

import numpy as np

from .init_weights import init_weights_using_correlation, init_weights_using_multistart
from .loss import HingeLoss

class SGD():
    def __init__(
        self,
        init_type: str = "none",
        loss_type: str = "hinge",
        subsampling_type: str = "random",
        lr: float = 1e-3,
        reg_coef: float = 0.0,
        m: int = 3,
        momentum: float = 0.0,
        nesterov: bool = False,
        h_optimization = True,
        batch_size: int = 4,
        n_iters: int = 100
    ):
        self.init_type = init_type
        self.subsampling_type = subsampling_type
        self.lr = lr
        self.reg_coef = reg_coef
        self.lmbda = 1. / m
        self.momentum = momentum
        self.nesterov = nesterov
        self.h_optimization = h_optimization
        self.batch_size = batch_size
        self.n_iters = n_iters
        self.eps = 1e-9
        if loss_type == "hinge":
            self.loss = HingeLoss()
        
    def init_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
    )-> np.ndarray:
        """Метод первоначальной инициализации весов"""
        if self.init_type == "corr":
            model_w = init_weights_using_correlation(X, y)
        elif self.init_type == "multi_start":
            model_w = init_weights_using_multistart(X, self.loss)
        else:
            model_w = np.zeros(X.shape[1])

        return model_w

    def get_batch_idxs(
        self,
        n_samples: int,
        subsampling_type: Union[str, None] = None,
        eps: float = 1e-8
    )-> np.ndarray:
        """Метод выбора батча объектов"""
        # случайное предъявление объектов
        if subsampling_type == "random":
            batch_idxs = np.random.choice(n_samples, size=self.batch_size, replace = False)
        # предъявление объектов по модулю отступа
        elif subsampling_type == "margin_abs":
            idxs = np.array(list(self.margin_values.keys()), dtype = int)
            margins = np.array(list(self.margin_values.values()), dtype = float)
            scores = 1. / (np.abs(margins) + eps) 
            probabilities = scores / scores.sum()
            batch_idxs = np.random.choice(idxs, size = self.batch_size, replace = False, p = probabilities)

        return batch_idxs
    
    def get_train_info(
        self
    )-> tuple[dict, dict, dict]:
        """Метод, позволяющий получить данные об обучении модели"""
        if self.margin_values is None:
            raise ValueError(
                "Перед тем как получить данные необходимо обучить модель!"
            )
        return self.loss_values, self.q_values, self.lr_values

    def __call__(
        self,
        X: np.ndarray,
        y: np.ndarray
    ):
        # Инициализируем веса
        model_w = self.init_weights(X, y)
        # Инициализируем оценку функционала
        n_samples = len(X)
        random_idxs = self.get_batch_idxs(n_samples, subsampling_type = "random")
        q = self.loss(np.dot(X[random_idxs], model_w))
        # Инициализируем ускорение
        velocity = 0.0
        # Инициализируем переменные для хранения данных обучения
        self.margin_values = {str(idx): 0 for idx in range(n_samples)}
        self.loss_values = {"0": 0}
        self.q_values = {"0": q}
        self.lr_values = {"0": self.lr}

        for iter_idx in range(self.n_iters):
            # Получим батч объектов
            batch_idxs = self.get_batch_idxs(n_samples, subsampling_type = self.subsampling_type)
            X_sub, y_sub = X[batch_idxs], y[batch_idxs]

            # Сдвинемся вперед, если необходимо
            if self.nesterov:
                model_w -= self.lr * self.momentum * velocity
            # Посчитаем функцию потерь и ее градиент
            batch_margins = np.dot(X_sub, model_w) * y_sub
            batch_loss = self.loss(batch_margins) + self.reg_coef * 0.5 * np.sum(model_w**2)
            batch_grad = self.loss.get_grad(X_sub, y_sub, batch_margins < 1) + self.reg_coef * model_w
            # Пересчитаем градиент, если необходимо
            if self.nesterov or self.momentum > 0:
                velocity = self.momentum * velocity + (1 - self.momentum) * batch_grad
                batch_grad = velocity

            # Обновим веса и оценку функционала
            model_w -= self.lr * batch_grad
            q = self.lmbda * batch_loss + (1 - self.lmbda) * q
            # Найдем оптимальный learning rate
            if self.h_optimization:
                lr_candidates = []
                for x_i, y_i in zip(X_sub, y_sub):
                    lr_star = self.loss.get_optimal_h(x_i, model_w, y_i)
                    if lr_star > 0:
                        lr_candidates.append(lr_star)
                self.lr = min(lr_candidates) if lr_candidates else 0.0

            # Сохраним значения отступов, лосса, learning rate, функционала при обучении
            self.loss_values[str(iter_idx+1)] = batch_loss
            self.q_values[str(iter_idx+1)] = q
            self.lr_values[str(iter_idx+1)] = self.lr
            for sub_idx, sub_margin in zip(batch_idxs, batch_margins):
                self.margin_values[str(sub_idx+1)] = sub_margin

        return model_w