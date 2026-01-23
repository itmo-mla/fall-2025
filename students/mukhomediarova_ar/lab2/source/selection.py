from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    # Запуск как пакет
    from .knn import ParzenWindowKNN, gaussian_kernel
    from .metrics import Array, EuclideanDistance
except ImportError:  # pragma: no cover
    # Запуск как скрипт
    from knn import ParzenWindowKNN, gaussian_kernel  # type: ignore
    from metrics import Array, EuclideanDistance  # type: ignore


@dataclass
class CondensedPrototypeSelector:
    """Простой жадный алгоритм отбора эталонов (condensed nearest neighbours).

    Стартуем с по одному объекту каждого класса и по очереди добавляем
    в множество эталонов те объекты, которые текущий классификатор
    (KNN с окном Парзена) предсказывает неверно.
    """

    k: int = 1
    max_iter: int = 10
    random_state: int = 0

    def select(self, x: Array, y: Array) -> tuple[Array, Array, np.ndarray]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        n_objects = x.shape[0]
        if n_objects == 0:
            raise ValueError("Пустая выборка")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x и y должны иметь одинаковую длину")

        rng = np.random.default_rng(self.random_state)
        indices = rng.permutation(n_objects)

        # Инициализируем по одному объекту на класс
        classes = np.unique(y)
        is_prototype = np.zeros(n_objects, dtype=bool)
        for cls in classes:
            cls_indices = indices[y[indices] == cls]
            if cls_indices.size > 0:
                is_prototype[cls_indices[0]] = True

        metric = EuclideanDistance()
        model = ParzenWindowKNN(k=max(1, self.k), metric=metric, kernel=gaussian_kernel)

        for _ in range(self.max_iter):
            changed = False
            model.fit(x[is_prototype], y[is_prototype])

            for idx in indices:
                if is_prototype[idx]:
                    continue

                y_pred = model.predict(x[idx : idx + 1])[0]
                if y_pred != y[idx]:
                    is_prototype[idx] = True
                    changed = True
                    model.fit(x[is_prototype], y[is_prototype])

            if not changed:
                break

        return x[is_prototype], y[is_prototype], is_prototype

