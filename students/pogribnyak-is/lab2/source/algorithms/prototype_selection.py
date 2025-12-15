import numpy as np
from typing import Optional

from utils.distances import compute_euclidean_distances


class PrototypeSelection:
    def __init__(self):
        self.prototypes: Optional[np.ndarray] = None
        self.prototype_labels: Optional[np.ndarray] = None
        self.selected_indices: Optional[np.ndarray] = None
        self.original_size: int = 0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PrototypeSelection':
        n_samples = len(X)
        self.original_size = n_samples
        selected = np.zeros(n_samples, dtype=bool)

        selected[np.array([np.where(y == cls)[0][0] for cls in np.unique(y)])] = True

        changed = True
        while changed:
            changed = False
            selected_indices = np.where(selected)[0]
            if len(selected_indices) == 0: break
            distances = compute_euclidean_distances(X, X[selected_indices])
            for i in range(n_samples):
                if not selected[i] and y[selected_indices][np.argsort(distances[i])[0]] != y[i]:
                    selected[i] = True
                    changed = True

        self.selected_indices = np.where(selected)[0]
        self.prototypes = X[self.selected_indices]
        self.prototype_labels = y[self.selected_indices]
        
        return self
    
    def transform(self) -> tuple[np.ndarray, np.ndarray]:
        if self.prototypes is None or self.prototype_labels is None: raise ValueError("Модель не обучена. Вызовите fit() перед transform()")
        return self.prototypes, self.prototype_labels
    
    def get_reduction_ratio(self) -> float:
        if self.selected_indices is None or self.original_size == 0: return 0.0
        return len(self.selected_indices) / self.original_size

