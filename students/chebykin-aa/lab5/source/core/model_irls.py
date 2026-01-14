import numpy as np
import logging

from .utils import sigmoid

class OwnLogisticRegressionIRLS:
    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.eps = 1e-12
        self.weights = None
        self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Получаем размерность данных
        n_samples, _ = X.shape

        # Начальная инициализация
        F = np.hstack([X, np.ones((n_samples, 1))])
        w = np.linalg.pinv(F.T @ F) @ F.T @ y

        for iteration in range(self.max_iter):
            # Считаем сигма i
            sigma = sigmoid(y * (F @ w))
            sigma = np.clip(sigma, self.eps, 1 - self.eps)

            # Считаем гамма i
            gamma = np.sqrt(sigma * (1 - sigma))

            # Считаем F тильда
            Gamma = np.diag(gamma)
            F_tilde = Gamma @ F

            # Считаем y тильда
            y_tilde = y * np.sqrt((1 - sigma) / sigma)

            # Выбираем градиентный шаг
            try:
                delta = np.linalg.solve(
                    F_tilde.T @ F_tilde,
                    F_tilde.T @ y_tilde
                )
            except np.linalg.LinAlgError:
                logging.info("ошибка МНК")
                break

            w_new = w + delta

            if np.linalg.norm(w_new - w) < self.tol:
                logging.info(f"сходимость на итерации {iteration}")
                break

            w = w_new

        self.weights = w[:-1]
        self.bias = w[-1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
