from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm


def get_distances(
    candidates: np.ndarray,
    objects: np.ndarray,
    metric_estimator: Callable | None = None,
) -> np.ndarray:
    """Матрица расстояний между кандидатами и объектами."""
    metric = metric_estimator or "euclidean"
    return cdist(candidates, objects, metric=metric)


def get_compact_profile(
    m: int,
    candidates_classes: np.ndarray,
    objects_classes: np.ndarray,
    sorted_distances_idx: np.ndarray,
) -> float:
    """Компакт-характеристика для m-го соседа."""
    return float(
        np.mean(
            candidates_classes != objects_classes[sorted_distances_idx[:, m - 1]]
        )
    )


def select_anchors(
    m_max: int,  # оставлен для совместимости с исходным API
    objects: np.ndarray,
    objects_classes: np.ndarray,
    n_samples_to_estimate: int | None = None,
    val_share: float = 0.2,
    metric_estimator: Callable | None = None,
    plot: bool = False,
    stop_threshold: float = 0.01,
) -> np.ndarray:
    """Отбор эталонов по компакт-характеристике (CCV).

    Возвращает булев массив длины len(objects): True — объект оставлен якорем.
    """
    n_objects = len(objects)
    l_val = int(n_objects * val_share)

    candidates = objects
    candidates_classes = objects_classes

    if n_samples_to_estimate is not None:
        if n_samples_to_estimate <= 0:
            raise ValueError("n_samples_to_estimate must be positive")
        if n_samples_to_estimate > n_objects:
            raise ValueError(
                "n_samples_to_estimate cannot be greater than number of objects"
            )
        idx = np.random.choice(n_objects, n_samples_to_estimate, replace=False)
        candidates = candidates[idx]
        candidates_classes = candidates_classes[idx]

    distances = get_distances(candidates, objects, metric_estimator)

    current_ccv = np.inf
    anchor_mask = np.ones(n_objects, dtype=bool)

    if plot:
        ccv_history: list[float] = []
        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(221)
        fig.suptitle("CCV value")

    for _ in tqdm(range(n_objects), total=n_objects):
        best_new_ccv: float | None = None
        idx_to_delete: int | None = None

        for i in range(n_objects):
            if not anchor_mask[i]:
                continue

            distances_excluded = distances.copy()
            distances_excluded[:, i] = np.inf
            sorted_idx_excluded = distances_excluded.argsort(axis=1)[:, 1:]

            ccv = (
                get_compact_profile(
                    1,
                    candidates_classes,
                    objects_classes,
                    sorted_idx_excluded,
                )
                * l_val
                / (l_val + 2)
                + get_compact_profile(
                    2,
                    candidates_classes,
                    objects_classes,
                    sorted_idx_excluded,
                )
                * 2
                * l_val
                / ((l_val + 1) * (l_val + 2))
                + get_compact_profile(
                    3,
                    candidates_classes,
                    objects_classes,
                    sorted_idx_excluded,
                )
                * 2
                / ((l_val + 1) * (l_val + 2))
            )

            if best_new_ccv is None or ccv < best_new_ccv:
                best_new_ccv = ccv
                idx_to_delete = i

        if current_ccv != np.inf:
            delta = (best_new_ccv - current_ccv) / current_ccv
            if delta >= stop_threshold:
                break

        if plot:
            ccv_history.append(best_new_ccv)
            ax1.plot(range(1, len(ccv_history) + 1), ccv_history, "g-")
            ax1.set_title("CCV value")
            ax1.set_xlabel("iterations")
            ax1.grid(True)
            fig.canvas.draw()

        anchor_mask[idx_to_delete] = False
        distances[:, idx_to_delete] = np.inf
        current_ccv = best_new_ccv

    return anchor_mask
