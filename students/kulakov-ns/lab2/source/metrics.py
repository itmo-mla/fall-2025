import numpy as np
from abc import ABC, abstractmethod


class Metric(ABC):
    """Абстрактный базовый класс для метрик расстояния."""

    @abstractmethod
    def _get_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Реализация конкретной метрики между двумя векторами."""
        raise NotImplementedError

    def get_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Расстояние между двумя объектами."""
        return self._get_distance(x, y)

    def get_distances(self, x: np.ndarray, xs: np.ndarray) -> np.ndarray:
        """Вектор расстояний от x до каждой строки xs (shape: (n_samples, n_features))."""
        return np.asarray([self.get_distance(x, row) for row in xs])


class MinkowskyMetric(Metric):
    """Метрика Минковского с параметром p."""

    def __init__(self, p: float) -> None:
        if p <= 0:
            raise ValueError("p must be positive")
        self.p = float(p)

    def _get_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        diff = np.abs(x - y) ** self.p
        return float(np.sum(diff) ** (1.0 / self.p))


class EuclideanMetric(MinkowskyMetric):
    """Евклидова метрика (частный случай Минковского при p=2)."""

    def __init__(self) -> None:
        super().__init__(p=2.0)
