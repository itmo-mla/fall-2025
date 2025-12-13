import numpy as np
from models import ParzenWindowKNN
from evaluation import loo_validate


class PrototypeSelection:

    def __init__(self):
        pass

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n = len(X)
        keep = np.ones(n, dtype=bool)
        best_error = loo_validate(X, y, 1, ParzenWindowKNN)

        changed = True
        while changed:
            changed = False
            current_indices = np.where(keep)[0]
            if len(current_indices) <= 1:
                break

            for idx in reversed(current_indices):
                temp_keep = keep.copy()
                temp_keep[idx] = False
                X_temp = X[temp_keep]
                y_temp = y[temp_keep]

                if len(X_temp) == 0:
                    continue

                error = loo_validate(X_temp, y_temp, 1, ParzenWindowKNN)
                if error < best_error:
                    best_error = error
                    keep[idx] = False
                    changed = True
                    break

        return X[keep], y[keep]
