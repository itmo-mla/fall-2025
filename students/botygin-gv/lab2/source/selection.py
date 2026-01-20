import numpy as np
from itertools import combinations
from models import ParzenWindowKNN
from evaluation import loo_validate


class PrototypeSelection:

    def __init__(self, max_remove=2):
        self.max_remove = max_remove

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n = len(X)
        keep = np.ones(n, dtype=bool)
        best_error = loo_validate(X, y, 1, ParzenWindowKNN)

        while True:
            current_indices = np.where(keep)[0]
            if len(current_indices) <= self.max_remove:
                break

            best_improvement = False
            best_combo = None
            best_temp_error = best_error

            for r in range(1, min(self.max_remove + 1, len(current_indices))):
                improved = False
                for combo in combinations(current_indices, r):
                    temp_keep = keep.copy()
                    temp_keep[list(combo)] = False
                    X_temp = X[temp_keep]
                    y_temp = y[temp_keep]

                    if len(X_temp) == 0:
                        continue

                    error = loo_validate(X_temp, y_temp, 1, ParzenWindowKNN)

                    if error < best_temp_error:
                        best_temp_error = error
                        best_combo = combo
                        improved = True

                if improved:
                    keep[list(best_combo)] = False
                    best_error = best_temp_error
                    best_improvement = True
                    break

            if not best_improvement:
                break

        return X[keep], y[keep]