import numpy as np
from typing import Optional
from loss import RecurrentLogisticLoss
from regularization import L2Regularizer
from optimizer import OptimizerContext, FastOptimizer, MomentumOptimizer
from sampler import RandomSampler, MarginSampler


class LinearClassifier:
    def __init__(
            self,
            init_method: str = "random",
            sampling_strategy: str = 'margin',
            optimizer_type: str = 'momentum',
            lr: float = 1e-4,
            momentum: float = 0.5,
            alpha: Optional[float] = 1e-3,
            loss_smoothing: float = 1e-3,
            random_seed: Optional[int] = None
    ):
        self.init_method = init_method
        self.sampling_strategy = sampling_strategy
        self.random_seed = random_seed

        self.regularizer = L2Regularizer(alpha) if alpha is not None else None
        self.loss = RecurrentLogisticLoss(smoothing=loss_smoothing, regularizer=self.regularizer)

        if optimizer_type == 'fast':
            self.optimizer = FastOptimizer(lr=lr)
        elif optimizer_type == 'momentum':
            self.optimizer = MomentumOptimizer(lr=lr, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        self.w = None
        self.b = np.zeros(1)

    def initialize_weights(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        if self.init_method == 'correlation':
            self.w = np.zeros((n_features, 1))
            for i in range(n_features):
                feature = X[:, i]
                denom = np.dot(feature, feature)
                self.w[i, 0] = (np.dot(y, feature) / denom) if denom != 0 else 0.0
        else:
            rng = np.random.default_rng(self.random_seed)
            scale = 2.0 / n_features
            self.w = rng.normal(scale=scale, size=(n_features, 1))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return (X @ self.w + self.b).ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(self.predict_proba(X)).astype(int)

    def backward(self, X_batch: np.ndarray, y_batch: np.ndarray):
        scores = self.predict_proba(X_batch)
        loss_value = self.loss.compute_loss(y_batch, scores, self.w)

        d_out = self.loss.gradient(y_batch, scores)

        d_w = X_batch.T @ d_out
        d_b = np.sum(d_out, axis=0)

        if self.regularizer is not None:
            d_w += self.regularizer.gradient(self.w)

        ctx = OptimizerContext(
            X_batch=X_batch,
            y_batch=y_batch,
            loss_fn=lambda yt, scores: self.loss(yt, scores),
            reg_fn=self.regularizer,
            bias=self.b
        )

        # Обновление весов
        self.w = self.optimizer.update("weights", self.w, d_w, ctx)
        self.b = self.optimizer.update("bias", self.b, d_b, ctx)

        return loss_value

    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 50,
            batch_size: int = 32,
            verbose: bool = False
    ) -> tuple:
        if self.w is None:
            self.initialize_weights(X_train, y_train)

        sampler_cls = MarginSampler if self.sampling_strategy == "margin" else RandomSampler
        sampler = sampler_cls(X_train, y_train, batch_size=batch_size, random_seed=self.random_seed)

        epoch_train_losses = []
        epoch_val_losses = []

        for epoch in range(epochs):
            predictions = self.predict_proba(X_train)
            batch_losses = []

            for _, X_batch, y_batch in sampler.get_batches(pred=predictions):
                loss = self.backward(X_batch, y_batch)
                batch_losses.append(loss)

            avg_epoch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            epoch_train_losses.append(avg_epoch_loss)

            if X_val is not None and y_val is not None:
                val_loss = float(self.loss(y_val, self.predict_proba(X_val)))
                epoch_val_losses.append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                val_info = f", Val Loss: {epoch_val_losses[-1]:.4f}" if epoch_val_losses else ""
                print(f"[Epoch {epoch + 1}/{epochs}] Train Loss: {avg_epoch_loss:.4f}{val_info}")

        return epoch_train_losses, epoch_val_losses
