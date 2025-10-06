import numpy as np
from knn import ParzenKNN, Metric
from sklearn.metrics import accuracy_score
from itertools import product
from tqdm import tqdm
from scipy.sparse import spmatrix
from scipy.spatial.distance import cdist
from typing import Callable
from sklearn.neighbors import NearestNeighbors
from math import comb



def kernel_function(kernel_arg: float) -> float:
    """Не возрастает и положительно на [0, 1]"""

    if kernel_arg >= 1:
        return 0
    return 1 / kernel_arg


def LOO_grid_search(estimator: ParzenKNN, param_grid: dict, X_train: np.ndarray, y_train: np.ndarray, scoring=accuracy_score, n_samples_to_score: int = 1000, n_anchor_objects: int = 5000):
    best_score = -np.inf
    best_params = None
    results = []
    
    keys = param_grid.keys()
    values = param_grid.values()
    
    for combination in product(*values):
        params = dict(zip(keys, combination))
        
        model = estimator(**params)
        
        predictions = []
        samples_to_score = np.random.choice(len(X_train), n_samples_to_score, replace=False)
        for i in tqdm(samples_to_score, total=len(samples_to_score)):
            random_samples_idx = np.random.choice(len(X_train), n_anchor_objects, replace=False)
            mask = np.isin(random_samples_idx, i)
            random_samples_idx = random_samples_idx[~mask]
            
            y_pred = model.predict(X_train[i], X_train[random_samples_idx], y_train[random_samples_idx])
            predictions.append(y_pred)

        score = scoring(y_train[samples_to_score].astype(str), predictions)
        
        results.append({
            'params': params,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'results': results
    }


def get_compact_profile(m: int, candidates: np.ndarray, candidates_classes: np.ndarray, objects: np.ndarray, objects_classes: np.ndarray, metric_estimator: Callable | None = None) -> float:
    if len(objects) != len(objects_classes) or len(candidates) != len(candidates_classes):
        raise ValueError("len(objects) != len(classes)")

    
    if not metric_estimator:
        metric_estimator = "euclidean"
    distances = cdist(candidates, objects, metric=metric_estimator)
    sorted_distances_idx = distances.argsort(axis=-1)

    return np.sum(candidates_classes != objects_classes[sorted_distances_idx[:, m]]) / len(candidates)


def select_anchors(m_max: int, objects: np.ndarray, objects_classes: np.ndarray, n_samples_to_estimate: int | None = None, val_share: float = 0.2, metric_estimator: Callable | None = None) -> np.ndarray:
    L = len(objects)
    l = int(L * val_share)
    ccv = 0

    for m in range(1, m_max + 1):
        candidates = objects
        candidates_classes = objects_classes

        if n_samples_to_estimate:
            random_samples_idx = np.random.choice(len(objects), n_samples_to_estimate, replace=False)
            candidates = candidates[random_samples_idx]
            candidates_classes = candidates_classes[random_samples_idx]
        print(comb(L - 1, l - 1 - m))
        print(comb(L, l))

        ccv += get_compact_profile(m, candidates, candidates_classes, objects, objects_classes) * comb(L - 1, l - 1 - m) / comb(L, l)

    print(f"{ccv}")
