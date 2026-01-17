from __future__ import annotations

from itertools import product
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from knn import ParzenKNN


def LOO_grid_search(
    estimator_cls: type[ParzenKNN],
    param_grid: Mapping[str, Sequence[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    scoring: Callable[[np.ndarray, np.ndarray], float] = accuracy_score,
    n_samples_to_score: int | None = None,
    n_anchor_objects: int | None = None,
) -> dict[str, Any]:
    """Подбор гиперпараметров методом leave-one-out (LOO)."""
    best_score = -np.inf
    best_params: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    for combination in product(*values):
        params = dict(zip(keys, combination))
        model = estimator_cls(**params)

        indices = np.arange(len(X_train))
        if n_samples_to_score is not None:
            if n_samples_to_score <= 0:
                raise ValueError("n_samples_to_score must be positive")
            if n_samples_to_score > len(indices):
                raise ValueError(
                    "n_samples_to_score cannot be greater than len(X_train)"
                )
            # берём случайное подмножество индексов из полного диапазона
            indices = np.random.choice(indices, n_samples_to_score, replace=False)

        predictions: list[Any] = []

        for i in tqdm(indices, total=len(indices)):
            anchor_idx = np.arange(len(X_train))

            if n_anchor_objects is not None:
                if n_anchor_objects <= 0:
                    raise ValueError("n_anchor_objects must be positive")
                if n_anchor_objects > len(anchor_idx):
                    raise ValueError(
                        "n_anchor_objects cannot be greater than len(X_train)"
                    )
                anchor_idx = np.random.choice(anchor_idx, n_anchor_objects, replace=False)

            # исключаем i-й объект из якорей (LOO)
            anchor_idx = anchor_idx[anchor_idx != i]

            y_pred = model.predict(
                X_train[i],
                X_train[anchor_idx],
                y_train[anchor_idx],
            )
            predictions.append(y_pred)

        y_true_subset = y_train[indices]
        y_pred_arr = np.asarray(predictions)
        score = float(scoring(y_true_subset, y_pred_arr))

        results.append({"params": params, "score": score})

        if score > best_score:
            best_score = score
            best_params = params

    return {
        "best_params": best_params,
        "best_score": best_score,
        "results": results,
    }
