from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


Array = np.ndarray


class DistanceMetric(ABC):
    """Абстрактный класс для метрических расстояний между объектами."""

    @abstractmethod
    def pairwise(self, x: Array, xs: Array) -> Array:
        """Вектор расстояний от объекта `x` до каждой строки матрицы `xs`."""
        raise NotImplementedError

    def __call__(self, x: Array, xs: Array) -> Array:
        return self.pairwise(x, xs)


class EuclideanDistance(DistanceMetric):
    """Евклидова метрика."""

    def pairwise(self, x: Array, xs: Array) -> Array:
        xs = np.asarray(xs, dtype=float)
        x = np.asarray(x, dtype=float)
        diff = xs - x
        return np.sqrt(np.sum(diff * diff, axis=1))


def classification_error(y_true: Array, y_pred: Array) -> float:
    """Доля неверно классифицированных объектов."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true и y_pred должны иметь одинаковую форму")
    return float(np.mean(y_true != y_pred))


def accuracy_score(y_true: Array, y_pred: Array) -> float:
    """Доля верно классифицированных объектов."""
    return 1.0 - classification_error(y_true, y_pred)


def ensure_1d(a: Any) -> Array:
    """Вспомогательная функция: приводит вход к одномерному numpy‑вектору."""
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError("Ожидается одномерный вектор")
    return arr

