from dataclasses import dataclass
from typing import Callable

import numpy as np

from metrics import Metric


KernelFn = Callable[[float], float]


@dataclass
class ParzenWindow:
    """Окно Парзена переменной ширины.

    Ширина окна задаётся расстоянием до k-го ближайшего соседа.
    """

    k: int
    metric_estimator: Metric
    kernel_function: KernelFn

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be positive")
        if not isinstance(self.metric_estimator, Metric):
            raise TypeError("metric_estimator must implement Metric interface")
        if not callable(self.kernel_function):
            raise TypeError("kernel_function must be callable")

        # Векторизуем ядро один раз
        self._kernel_vectorized: Callable[[np.ndarray], np.ndarray]
        self._kernel_vectorized = np.vectorize(self.kernel_function)

    def _get_arguments(self, x: np.ndarray, xs: np.ndarray) -> np.ndarray:
        """Аргументы ядра: расстояния / расстояние до k-го соседа."""
        xs = np.asarray(xs)
        if xs.size == 0:
            return np.array([])

        distances = self.metric_estimator.get_distances(x, xs)

        if self.k > len(distances):
            raise ValueError(
                f"k={self.k} is greater than number of anchor objects ({len(distances)})"
            )

        # Индекс k-го соседа с учётом 0-индексации (исправление исходной off-by-one ошибки)
        sorted_idx = np.argsort(distances)
        k_index = sorted_idx[self.k - 1]
        max_distance = distances[k_index]

        if max_distance == 0:
            # Все объекты совпали с x → все аргументы 0
            return np.zeros_like(distances)

        return distances / max_distance

    def get_weights(self, x: np.ndarray, xs: np.ndarray) -> np.ndarray:
        """Вес каждого якоря для объекта x."""
        kernel_args = self._get_arguments(x, xs)
        if kernel_args.size == 0:
            return kernel_args
        return self._kernel_vectorized(kernel_args)
