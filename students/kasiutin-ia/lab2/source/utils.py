import numpy as np
from knn import ParzenKNN
from sklearn.metrics import accuracy_score
from itertools import product
from tqdm import tqdm
from scipy.spatial.distance import cdist
from typing import Callable
import matplotlib.pyplot as plt


class MetricsEstimator:
    def __init__(self):
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1_score = None

    def get_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self.accuracy = self.get_accuracy(y_true, y_pred)
        self.precision = self.get_precision(y_true, y_pred)
        self.recall = self.get_recall(y_true, y_pred)
        self.f1_score = self.get_f1_score(y_true, y_pred)

    def get_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sum(y_true == y_pred) / len(y_true)

    def get_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = np.sum((y_true == 1) * (y_pred == 1))
        fp = np.sum((y_true == -1) * (y_pred == 1))
        return tp / (tp + fp)

    def get_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = np.sum((y_true == 1) * (y_pred == 1))
        fn = np.sum((y_true == 1) * (y_pred == -1))
        return tp / (tp + fn)

    def get_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        precision = self.get_precision(y_true, y_pred)
        recall = self.get_recall(y_true, y_pred)
        return 2 * precision * recall / (precision + recall)

    def __str__(self):
        return f"accuracy = {self.accuracy}\nprecision = {self.precision}\nrecall = {self.recall}\nf1_score = {self.f1_score}"


def kernel_function(kernel_arg: float) -> float:
    """Не возрастает и положительно на [0, 1]"""

    if kernel_arg >= 1:
        return 0
    return 1 / kernel_arg


def LOO_grid_search(
    estimator: ParzenKNN,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    scoring=accuracy_score,
    n_samples_to_score: int | None = None,
    n_anchor_objects: int | None = None,
):
    best_score = -np.inf
    best_params = None
    results = []

    keys = param_grid.keys()
    values = param_grid.values()

    for combination in product(*values):
        params = dict(zip(keys, combination))

        model = estimator(**params)

        predictions = []

        samples_to_score = np.arange(len(X_train))
        if n_samples_to_score:
            samples_to_score = np.random.choice(
                n_samples_to_score, n_samples_to_score, replace=False
            )

        for i in tqdm(samples_to_score, total=len(samples_to_score)):
            anchor_samples_idx = np.arange(len(X_train))

            if n_anchor_objects is not None:
                anchor_samples_idx = np.random.choice(
                    anchor_samples_idx, n_anchor_objects, replace=False
                )

            mask = np.isin(anchor_samples_idx, i)
            anchor_samples_idx = anchor_samples_idx[~mask]

            y_pred = model.predict(
                X_train[i], X_train[anchor_samples_idx], y_train[anchor_samples_idx]
            )
            predictions.append(y_pred)

        score = scoring(y_train[samples_to_score].astype(str), predictions)

        results.append({"params": params, "score": score})

        if score > best_score:
            best_score = score
            best_params = params

    return {"best_params": best_params, "best_score": best_score, "results": results}


def get_distances(
    candidates: np.ndarray,
    objects: np.ndarray,
    metric_estimator: Callable | None = None,
) -> np.ndarray:
    if not metric_estimator:
        metric_estimator = "euclidean"
    distances = cdist(candidates, objects, metric=metric_estimator)

    return distances


def get_compact_profile(
    m: int,
    candidates_classes: np.ndarray,
    objects_classes: np.ndarray,
    sorted_distances_idx: np.ndarray,
) -> float:
    return np.sum(
        candidates_classes != objects_classes[sorted_distances_idx[:, m - 1]]
    ) / len(candidates_classes)


def select_anchors(
    m_max: int,
    objects: np.ndarray,
    objects_classes: np.ndarray,
    n_samples_to_estimate: int | None = None,
    val_share: float = 0.2,
    metric_estimator: Callable | None = None,
    plot: bool = False,
    stop_threshold: float = 0.01,
) -> np.ndarray:
    L = len(objects)
    l_val = int(L * val_share)
    ccv = 0

    candidates = objects
    candidates_classes = objects_classes

    if n_samples_to_estimate:
        random_samples_idx = np.random.choice(
            len(objects), n_samples_to_estimate, replace=False
        )
        candidates = candidates[random_samples_idx]
        candidates_classes = candidates_classes[random_samples_idx]

    distances = get_distances(candidates, objects, metric_estimator)

    current_ccv = np.inf
    best_new_ccv = current_ccv
    idx_to_delete = None
    anchor_objects_indices = np.ones(len(objects))

    if plot:
        ccv_history = []
        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(221)
        fig.suptitle("CCV value")
        step = 1

    for _ in tqdm(range(len(objects)), total=len(objects)):
        for i in range(len(objects)):
            if not anchor_objects_indices[i]:
                continue

            distances_excluded = distances.copy()
            distances_excluded[:, i] = np.inf
            sorted_distances_idx_excluded = distances_excluded.argsort(axis=1)[:, 1:]
            ccv = (
                get_compact_profile(
                    1,
                    candidates_classes,
                    objects_classes,
                    sorted_distances_idx_excluded,
                )
                * l_val
                / (l_val + 2)
                + get_compact_profile(
                    2,
                    candidates_classes,
                    objects_classes,
                    sorted_distances_idx_excluded,
                )
                * 2
                * l_val
                / ((l_val + 1) * (l_val + 2))
                + get_compact_profile(
                    3,
                    candidates_classes,
                    objects_classes,
                    sorted_distances_idx_excluded,
                )
                * 2
                / ((l_val + 1) * (l_val + 2))
            )

            if best_new_ccv is None or ccv < best_new_ccv:
                best_new_ccv = ccv
                idx_to_delete = i

        if (best_new_ccv - current_ccv) / current_ccv >= stop_threshold:
            break

        if plot:
            ccv_history.append(best_new_ccv)
            plt.grid()
            ax1.plot(range(step), ccv_history, "g-")
            ax1.set_title("CCV value")
            ax1.set_xlabel("iterations")
            step += 1
            fig.canvas.draw()

        anchor_objects_indices[idx_to_delete] = 0
        distances[:, idx_to_delete] = np.inf
        idx_to_delete = None
        current_ccv = best_new_ccv
        best_new_ccv = np.inf

    return anchor_objects_indices.astype(bool)
