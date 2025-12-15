import numpy as np
from typing import Optional, Callable

from core.loss import LossFunction, HingeLoss
from core.regularizer import Regularizer, L2Regularizer
from core.optimizer import Optimizer, SGDWithMomentum
from core.initializer import WeightInitializer, RandomInitializer

class LinearClassifier:
    def __init__(self,
                 loss: Optional[LossFunction] = None,
                 regularizer: Optional[Regularizer] = None,
                 optimizer: Optional[Optimizer] = None,
                 initializer: Optional[WeightInitializer] = None):
        self.loss = loss or HingeLoss()
        self.regularizer = regularizer or L2Regularizer()
        self.optimizer = optimizer or SGDWithMomentum()
        self.initializer = initializer or RandomInitializer()

        self.w: Optional[np.ndarray] = None
        self.b: Optional[float] = None
        self.n_features: Optional[int] = None

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'margins': [],
            'recurrent_loss': []
        }
        self.recurrent_loss_alpha = 0.05
        self.recurrent_loss_value: Optional[float] = None

    def _initialize_weights(self, X: np.ndarray, y: Optional[np.ndarray] = None, seed: Optional[int] = None):
        self.w, self.b = self.initializer.initialize(
            X=X, y=y, n_features=X.shape[1], seed=seed
        )
        self.n_features = X.shape[1]
        if hasattr(self.optimizer, "reset"): self.optimizer.reset()

    def margin(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        if self.w is None or self.b is None: raise ValueError("Модель не обучена. Сначала вызовите fit()")
        pred = X @ self.w + self.b
        return y * pred if y is not None else pred

    def predict(self, X: np.ndarray) -> np.ndarray: return np.sign(self.margin(X))

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.w is None or self.b is None: raise ValueError("Модель не обучена. Сначала вызовите fit()")
        y_pred = X @ self.w + self.b
        loss_value = self.loss.compute(y, y_pred)
        if self.regularizer is not None: loss_value += self.regularizer.penalty(self.w)
        return loss_value

    def _compute_gradient(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        if self.w is None or self.b is None:
            raise ValueError("Модель не обучена. Сначала вызовите fit()")
        grad_w, grad_b = self.loss.gradient(X, y, self.w, self.b)
        if self.regularizer is not None:
            grad_w += self.regularizer.gradient(self.w)
        return grad_w, grad_b

    def fit(self,
            X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 50, batch_size: int = 1, shuffle: bool = True,
            seed: Optional[int] = None, verbose: bool = True,
            callback: Optional[Callable[['LinearClassifier', int], None]] = None,
            order_by_margin: bool = False):

        self._initialize_weights(X_train, y_train, seed)
        if verbose: print(f"Начальная норма весов: {np.linalg.norm(self.w):.6f}, смещение: {self.b:.6f}")
        self.recurrent_loss_value = None

        n_samples = X_train.shape[0]
        indices = np.arange(n_samples)

        for epoch in range(epochs):
            if order_by_margin:
                margins = self.margin(X_train, y_train)
                indices = np.argsort(np.abs(margins))
            elif shuffle:
                if seed is not None: np.random.seed(seed + epoch)
                np.random.shuffle(indices)

            epoch_losses = []

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                if batch_size == 1:
                    x, y = X_batch[0], y_batch[0]
                    grad_w, grad_b = self.loss.gradient_single(x, y, self.w, self.b)
                    if self.regularizer: grad_w += self.regularizer.gradient(self.w)
                else: grad_w, grad_b = self._compute_gradient(X_batch, y_batch)

                current_loss = self._compute_loss(X_batch, y_batch)
                if self.recurrent_loss_value is None: self.recurrent_loss_value = current_loss
                else: self.recurrent_loss_value = (
                        (1 - self.recurrent_loss_alpha) * self.recurrent_loss_value +
                        self.recurrent_loss_alpha * current_loss
                    )
                self.history['recurrent_loss'].append(self.recurrent_loss_value)

                ctx = type("Ctx", (), {})()
                ctx.X_batch, ctx.y_batch, ctx.loss_fn, ctx.regularizer, ctx.current_loss, ctx.iteration = \
                    X_batch, y_batch, self.loss, self.regularizer, current_loss, epoch

                self.w, self.b = self.optimizer.update(self.w, self.b, grad_w, grad_b, ctx)

                epoch_losses.append(current_loss)

            train_loss = float(np.mean(epoch_losses))
            train_acc = float(self.score(X_train, y_train))

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            val_loss = val_acc = None
            if X_val is not None and y_val is not None:
                val_loss = float(self._compute_loss(X_val, y_val))
                val_acc = float(self.score(X_val, y_val))
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

            self.history['margins'].append(self.margin(X_train, y_train).copy())

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                if val_loss is not None and val_acc is not None: msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                print(msg)

            if callback: callback(self, epoch)

    def score(self, X: np.ndarray, y: np.ndarray) -> float: return float(np.mean(self.predict(X) == y))

    def get_weights(self) -> tuple[np.ndarray, float]:
        if self.w is None or self.b is None: raise ValueError("Модель не обучена. Сначала вызовите fit()")
        return self.w.copy(), self.b

