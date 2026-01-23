import numpy as np
from sklearn.metrics import accuracy_score


class LinearClassifier:
    def __init__(
        self,
        learning_rate=0.01,
        momentum=0.9,
        lambda_reg=0.1,
        n_epochs=100,
        batch_size=32,
    ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.lambda_reg = lambda_reg
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.w = None
        self.loss_history = []
        self.val_loss_history = []
        self.quality_history = []

    def loss(self, w, X, y):
        """
        L(w) = 1/N * sum((y_i - <w, x_i>)^2) + lambda/2 * ||w||^2
        """
        predictions = np.dot(X, w)
        error = y - predictions
        mse = np.mean(error**2)
        reg = (self.lambda_reg / 2) * np.dot(w, w)
        return mse + reg

    def gradient(self, w, X_batch, y_batch):
        """
        grad L = -2/N * X^T (y - Xw) + lambda * w
        """
        N = X_batch.shape[0]
        predictions = np.dot(X_batch, w)
        error = y_batch - predictions
        grad_data = -2 * np.dot(X_batch.T, error) / N
        grad_reg = self.lambda_reg * w
        return grad_data + grad_reg

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        method="sgd",
        init="random",
        presentation="random",
    ):
        n_samples, n_features = X.shape
        self.loss_history = []
        self.val_loss_history = []
        self.quality_history = []

        if init == "random":
            self.w = np.random.randn(n_features) * 0.01
        elif init == "correlation":
            corrs = []
            for i in range(n_features):
                if np.std(X[:, i]) == 0:
                    corrs.append(0)
                else:
                    corrs.append(np.corrcoef(X[:, i], y)[0, 1])
            self.w = np.array(corrs) * 0.01
            self.w = np.nan_to_num(self.w)

        if method == "sgd":
            self._fit_sgd(X, y, X_val, y_val, presentation)
        elif method == "steepest":
            self._fit_steepest(X, y, X_val, y_val)

    def _fit_sgd(self, X, y, X_val, y_val, presentation):
        n_samples = X.shape[0]
        v = np.zeros_like(self.w)

        for epoch in range(self.n_epochs):
            indices = np.arange(n_samples)

            if presentation == "random":
                np.random.shuffle(indices)
            elif presentation == "margin":
                margins = np.abs(y * np.dot(X, self.w))
                indices = np.argsort(margins)

            for i in range(0, n_samples, self.batch_size):
                batch_idx = indices[i : i + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                grad = self.gradient(self.w, X_batch, y_batch)

                # Gradient clipping для стабильности
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 10.0:
                    grad = grad * (10.0 / grad_norm)

                v = self.momentum * v + grad
                self.w -= self.learning_rate * v

            self.loss_history.append(self.loss(self.w, X, y))
            if X_val is not None and y_val is not None:
                self.val_loss_history.append(self.loss(self.w, X_val, y_val))

    def _fit_steepest(self, X, y, X_val, y_val):
        n_samples = X.shape[0]

        for epoch in range(self.n_epochs):
            grad = self.gradient(self.w, X, y)

            # Оптимальный размер шага для квадратичной функции
            # alpha = (g^T g) / (g^T H g)
            # где H = 2/N * X^T X + lambda * I
            Xg = np.dot(X, grad)
            grad_sq = np.dot(grad, grad)

            denominator = (2.0 / n_samples) * np.dot(Xg, Xg) + self.lambda_reg * grad_sq

            if denominator < 1e-12:
                step = self.learning_rate
            else:
                step = grad_sq / denominator

            self.w -= step * grad
            self.loss_history.append(self.loss(self.w, X, y))
            if X_val is not None and y_val is not None:
                self.val_loss_history.append(self.loss(self.w, X_val, y_val))

    def predict(self, X):
        return np.sign(np.dot(X, self.w))

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)
