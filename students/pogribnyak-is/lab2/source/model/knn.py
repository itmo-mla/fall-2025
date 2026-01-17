import numpy as np
from typing import Optional

from model.kernel import GaussianKernel, Kernel
from utils.distances import compute_euclidean_distances


class KNN:
    def __init__(self, k: int = 5, kernel: Kernel = GaussianKernel(1.5)) -> None:
        if k <= 0:
            raise ValueError("k должно быть положительным числом")
        self.k = k
        self.kernel = kernel
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._unique_classes: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNN':
        if X.shape[0] != y.shape[0]:
            raise ValueError("Количество образцов в X и y должно совпадать")

        self._X_train = np.asarray(X, dtype=np.float64)
        self._y_train = np.asarray(y)
        self._unique_classes = np.unique(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        self._check_fitted()

        distances = compute_euclidean_distances(X, self._X_train)
        k_idx, k_distances, k_labels = self._get_k_neighbors(distances)
        weights = self._compute_weights(k_distances)
        class_weights = self._compute_class_weights(k_labels, weights)

        predicted_indices = np.argmax(class_weights, axis=1)
        return self._unique_classes[predicted_indices]

    def _check_fitted(self):
        if self._X_train is None or self._y_train is None: raise ValueError("Модель не обучена. Вызовите fit() перед predict()")

    def _get_k_neighbors(self, distances: np.ndarray):
        k_idx = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        k_distances = np.take_along_axis(distances, k_idx, axis=1)
        k_labels = self._y_train[k_idx]
        return k_idx, k_distances, k_labels

    def _compute_weights(self, k_distances: np.ndarray):
        return self.kernel(k_distances / np.maximum(k_distances.max(axis=1), 1e-8)[:, np.newaxis])

    def _compute_class_weights(self, k_labels: np.ndarray, weights: np.ndarray):
        n_test = k_labels.shape[0]
        n_classes = len(self._unique_classes)
        class_weights = np.zeros((n_test, n_classes), dtype=np.float64)

        for class_idx, cls in enumerate(self._unique_classes):
            mask = (k_labels == cls)
            class_weights[:, class_idx] = np.sum(weights * mask, axis=1)

        return class_weights
