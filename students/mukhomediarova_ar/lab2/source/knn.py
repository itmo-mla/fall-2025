from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

try:
    # Запуск как пакет: python -m students.mukhomediarova_ar.lab2.source.main
    from .metrics import Array, DistanceMetric, EuclideanDistance
except ImportError:  # pragma: no cover - для запуска как простого скрипта
    # Запуск напрямую: python students/mukhomediarova_ar/lab2/source/main.py
    from metrics import Array, DistanceMetric, EuclideanDistance  # type: ignore


KernelFn = Callable[[Array], Array]


def gaussian_kernel(r: Array) -> Array:
    """Гауссово ядро K(r) = exp(-r^2 / 2) для r >= 0."""
    r = np.asarray(r, dtype=float)
    return np.exp(-0.5 * r * r)


def weighted_majority_vote(labels: Sequence, weights: Array) -> object:
    """Взвешенное голосование по меткам.

    Параметры
    ---------
    labels : последовательность меток (любые hashable‑объекты)
    weights : массив весов той же длины
    """
    labels_arr = np.asarray(labels)
    weights = np.asarray(weights, dtype=float)
    if labels_arr.shape[0] != weights.shape[0]:
        raise ValueError("labels и weights должны иметь одинаковую длину")

    unique_labels, inverse = np.unique(labels_arr, return_inverse=True)
    class_weights = np.bincount(inverse, weights=weights)
    best_idx = int(np.argmax(class_weights))
    return unique_labels[best_idx]


@dataclass
class ParzenWindowKNN:
    """KNN с методом окна Парзена переменной ширины.

    Ширина окна определяется расстоянием до k‑го ближайшего соседа.
    Веса соседей вычисляются с помощью ядра (по умолчанию — гауссово).
    """

    k: int
    metric: DistanceMetric | None = None
    kernel: KernelFn = gaussian_kernel

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("Параметр k должен быть положительным целым числом")
        if self.metric is None:
            self.metric = EuclideanDistance()

        if not callable(self.kernel):
            raise TypeError("kernel должен быть вызываемым объектом")

        self._x_train: Array | None = None
        self._y_train: Array | None = None

    @property
    def is_fitted(self) -> bool:
        return self._x_train is not None and self._y_train is not None

    def fit(self, x: Array, y: Array) -> "ParzenWindowKNN":
        """Сохраняет обучающую выборку."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.shape[0] != y.shape[0]:
            raise ValueError("Количество объектов и меток должно совпадать")
        if x.shape[0] == 0:
            raise ValueError("Обучающая выборка не может быть пустой")

        self._x_train = x
        self._y_train = y
        return self

    def _check_is_fitted(self) -> tuple[Array, Array]:
        if not self.is_fitted:
            raise RuntimeError("Модель ещё не обучена. Сначала вызовите fit().")
        assert self._x_train is not None
        assert self._y_train is not None
        return self._x_train, self._y_train

    def _compute_weights(self, x: Array) -> Array:
        """Веса всех объектов обучающей выборки для одного объекта x."""
        x_train, _ = self._check_is_fitted()

        distances = self.metric.pairwise(x, x_train)  # type: ignore[union-attr]
        if distances.size == 0:
            raise RuntimeError("Обучающая выборка пуста")

        # Эффективное k не может превышать количество объектов
        k_eff = min(self.k, distances.shape[0])
        order = np.argsort(distances)
        # Расстояние до k‑го соседа (с учётом 0‑индексации)
        h = float(distances[order[k_eff - 1]])

        if h <= 0.0:
            # Если все расстояния почти нулевые, считаем веса равными
            return np.ones_like(distances, dtype=float)

        args = distances / h
        return self.kernel(args)

    def predict_one(self, x: Array) -> object:
        """Предсказание класса для одного объекта."""
        x = np.asarray(x, dtype=float)
        x_train, y_train = self._check_is_fitted()
        weights = self._compute_weights(x)
        return weighted_majority_vote(y_train, weights)

    def predict(self, x: Array) -> Array:
        """Предсказание классов для набора объектов."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return np.asarray([self.predict_one(x)])
        preds = [self.predict_one(row) for row in x]
        return np.asarray(preds)

