from __future__ import annotations

from typing import Callable

import numpy as np
from tqdm import tqdm

from metrics import Metric, MinkowskyMetric, EuclideanMetric
from window import ParzenWindow


KernelFn = Callable[[float], float]


class ParzenKNN:
    """KNN с окном Парзена переменной ширины."""

    def __init__(self, k: int, metric_estimator: Metric, kernel_function: KernelFn):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.metric_estimator = metric_estimator
        self.kernel_function = kernel_function
        self.window_function = ParzenWindow(k, metric_estimator, kernel_function)

    def predict(
        self,
        x: np.ndarray,
        anchors: np.ndarray,
        classes: np.ndarray,
        n_anchor_objects: int | None = None,
    ):
        """Предсказать класс одного объекта x по якорям anchors и меткам classes."""
        anchors = np.asarray(anchors)
        classes = np.asarray(classes)

        if anchors.shape[0] != classes.shape[0]:
            raise ValueError("anchors and classes must have the same length")

        n_objects = anchors.shape[0]

        # Подсэмплирование якорей (для ускорения)
        if n_anchor_objects is not None:
            if n_anchor_objects <= 0:
                raise ValueError("n_anchor_objects must be positive")
            if n_anchor_objects > n_objects:
                raise ValueError(
                    "n_anchor_objects cannot be greater than number of anchors"
                )
            idx = np.random.choice(n_objects, n_anchor_objects, replace=False)
            anchors = anchors[idx]
            classes = classes[idx]

        weights = self.window_function.get_weights(x, anchors)
        if weights.size == 0:
            raise ValueError("cannot predict without anchor objects")

        # Суммируем веса по классам через numpy (без defaultdict + str)
        labels, inverse = np.unique(classes, return_inverse=True)
        class_weights = np.bincount(inverse, weights=weights.astype(float))
        best_label = labels[int(np.argmax(class_weights))]

        return best_label

    def predict_batched(
        self,
        test_objects: np.ndarray,
        anchor_objects: np.ndarray,
        classes: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Предсказать классы для набора объектов."""
        test_objects = np.asarray(test_objects)
        return np.array(
            [
                self.predict(obj, anchor_objects, classes, **kwargs)
                for obj in tqdm(test_objects, total=len(test_objects))
            ]
        )

    # Совместимость с исходной опечаткой
    def predict_bathced(
        self,
        test_objects: np.ndarray,
        anchor_objects: np.ndarray,
        classes: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        return self.predict_batched(test_objects, anchor_objects, classes, **kwargs)
