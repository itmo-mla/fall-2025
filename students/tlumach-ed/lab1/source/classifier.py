import os
import numpy as np
from typing import Optional
from margin_analyzer import Margin
from losses import RecurrentLogisticLoss, L2Regularizer
from momentum_fast import Optimizer

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

class DataSampler:
    """ sampler: 'random', 'margin'."""
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32,
                 strategy: str = 'margin', random_seed: Optional[int] = None):
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
        self.batch_size = batch_size
        self.strategy = strategy
        self.rng = np.random.default_rng(random_seed)

    def _create_batches(self, indices: np.ndarray):
        n_batches = int(np.ceil(self.n_samples / self.batch_size))
        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            batch_idx_slice = indices[start:start + self.batch_size]
            yield batch_idx, self.X[batch_idx_slice], self.y[batch_idx_slice]

    #чаще брать объекты, на которых уверенность меньше
    def _margin_sampling(self, predictions: np.ndarray):
        # predictions: scores for all training samples
        margins = predictions * self.y
        # p ~ 1/|margin|
        abs_margin = np.abs(margins)
        abs_margin[abs_margin < 1e-6] = 1e-6
        probs = 1.0 / abs_margin
        probs = probs / probs.sum()
        indices = self.rng.choice(self.n_samples, size=self.n_samples, replace=False, p=probs)
        return self._create_batches(indices)

    def _random_sampling(self):
        indices = self.rng.choice(self.n_samples, size=self.n_samples, replace=False)
        return self._create_batches(indices)

    def sample_batches(self, predictions: Optional[np.ndarray] = None, batch_size: Optional[int] = None):
        if batch_size is not None:
            self.batch_size = batch_size

        if self.strategy == 'margin':
            if predictions is None:
                raise ValueError("Margin sampling requires predictions.")
            return self._margin_sampling(predictions)
        else:
            return self._random_sampling()



class LinearClassifier:
    def __init__(self, init_method: str = "random", batch_strategy: str = 'margin',
                 optimizer_type: str = 'momentum', lr: float = 1e-4,
                 momentum: float = 0.5, l2_strength: Optional[float] = 1e-3,
                 loss_smoothing: float = 1e-3, rng_seed: Optional[int] = None):
        self.init_method = init_method
        self.batch_strategy = batch_strategy
        self.optimizer_type = optimizer_type
        self.rng = np.random.default_rng(rng_seed)

        # regularizer and loss
        self.regularizer = L2Regularizer(l2_strength) if l2_strength is not None else None
        self.loss = RecurrentLogisticLoss(smoothing=loss_smoothing, regularizer=self.regularizer)

        # optimizer
        self.optimizer = Optimizer(lr=lr, momentum=momentum, mode=optimizer_type)

        # model params (weights as column vector, bias)
        self.w = None
        self.b = np.zeros(1)

        # training history (filled by fit)
        self.epoch_train_loss = []
        self.epoch_val_loss = []

    def initialize_weights(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        if self.init_method == 'correlation':
            self.w = np.zeros((n_features, 1))
            for i in range(n_features):
                feature = X[:, i]
                denom = np.dot(feature, feature)
                self.w[i, 0] = (np.dot(y, feature) / denom) if denom != 0 else 0.0
        else:
            scale = 2.0 / n_features
            self.w = self.rng.normal(scale=scale, size=(n_features, 1))

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        # returns 1d scores array
        return (X @ self.w + self.b).ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.predict_scores(X)
        return np.sign(scores).astype(int)

    def compute_margins(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.predict_scores(X) * y

    def plot_margins(self, X: np.ndarray, y: np.ndarray, title: str = "") -> np.ndarray:
        # delegate to margin_analyzer for consistent plotting style
        margin_tool = Margin()
        return margin_tool.plot_margins(self.predict_scores(X), y, title)

    def plot_training_history(self, title: str = ""):
        # Uses self.loss.history (batch-level) and self.epoch_train_loss / self.epoch_val_loss
        if not hasattr(self, 'epoch_train_loss') or not self.epoch_train_loss:
            return

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))

        # Batch-level plot
        plt.subplot(1, 2, 1)
        if self.loss.history:
            plt.plot(self.loss.history, 'b-', label="Train Loss", linewidth=1, alpha=0.7)

        if self.epoch_val_loss and self.loss.history:
            batches_per_epoch = max(1, len(self.loss.history) // len(self.epoch_train_loss))
            val_interpolated = []
            for epoch_loss in self.epoch_val_loss:
                val_interpolated.extend([epoch_loss] * batches_per_epoch)
            val_interpolated = val_interpolated[:len(self.loss.history)]
            plt.plot(val_interpolated, 'r-', label="Validation Loss", linewidth=1, alpha=0.7)

        plt.xlabel('Батч')
        plt.ylabel('Loss')
        plt.title(f'{title}\nTrain/Validation Loss по батчам')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Epoch-level plot
        plt.subplot(1, 2, 2)
        epochs = range(len(self.epoch_train_loss))
        plt.plot(epochs, self.epoch_train_loss, 'b-', label="Train Loss", linewidth=2)
        if self.epoch_val_loss:
            plt.plot(epochs, self.epoch_val_loss, 'r-', label="Validation Loss", linewidth=2)

        plt.xlabel('Эпоха')
        plt.ylabel('Loss')
        plt.title(f'{title}\nTrain/Validation Loss по эпохам')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        img_name = f"training_{title.replace(' ', '_').replace('+', '_')}.png"
        plt.savefig(os.path.join(IMAGES_DIR, img_name), dpi=300, bbox_inches='tight')
        plt.show()

    def sgd_step(self, X_batch: np.ndarray, y_batch: np.ndarray):
        # forward
        y_pred = self.predict_scores(X_batch)
        loss_value = self.loss.compute_loss(y_batch, y_pred, self.w)

        # backward: dL/dscore per sample (n,1)
        d_out = self.loss.gradient(y_batch, y_pred)

        # reg gradient
        d_w_reg = self.regularizer.gradient(self.w) if self.regularizer is not None else 0.0

        # weight gradient (n_features, 1)
        d_w = X_batch.T @ d_out + d_w_reg
        d_b = np.sum(d_out, axis=0)

        # update params
        if self.optimizer_type == "momentum":
            self.w = self.optimizer.update("weights", self.w, d_w)
            self.b = self.optimizer.update("bias", self.b, d_b)
        else:
            # fast update uses the batch to pick learning rate for weights
            self.w = self.optimizer.update("weights", self.w, d_w,
                                           X_batch=X_batch, y_batch=y_batch,
                                           loss_fn=lambda yt, scores: self.loss(yt, scores),
                                           reg_fn=(self.regularizer if self.regularizer is not None else None),
                                           bias=self.b)
            # bias simple step with base lr
            self.b = self.b - self.optimizer.lr * d_b

        return loss_value

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            n_epochs: int = 50, batch_size: int = 32, verbose: int = 0):
        if self.w is None:
            self.initialize_weights(X_train, y_train)

        sampler = DataSampler(X_train, y_train, batch_size=batch_size, strategy=self.batch_strategy)
        self.loss.history = []
        self.epoch_train_loss = []
        self.epoch_val_loss = []

        for epoch in range(n_epochs):
            predictions = self.predict_scores(X_train) if self.batch_strategy == "margin" else None

            epoch_losses = []
            for _, X_batch, y_batch in sampler.sample_batches(predictions=predictions):
                loss = self.sgd_step(X_batch, y_batch)
                epoch_losses.append(loss)

            avg_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            self.epoch_train_loss.append(avg_epoch_loss)

            if X_val is not None and y_val is not None:
                val_loss = float(self.loss(y_val, self.predict_scores(X_val)))
                self.epoch_val_loss.append(val_loss)
                self.loss.val_history.append(val_loss)

            if verbose > 0:
                val_info = f", Val Loss: {self.epoch_val_loss[-1]:.4f}" if self.epoch_val_loss else ""
                print(f"\r[Epoch {epoch + 1}/{n_epochs}] Train Loss: {avg_epoch_loss:.4f}{val_info}", end="")

        if verbose > 0:
            print()
        return {
            "train_history_batches": self.loss.history,
            "epoch_train_loss": self.epoch_train_loss,
            "epoch_val_loss": self.epoch_val_loss
        }
