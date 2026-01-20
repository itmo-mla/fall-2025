import numpy as np
from collections import defaultdict


class KNNClassifier:
    def __init__(self, k: int = 3, h: float = 1.0, p: float = 2.0, window: str = "dynamic"):
        assert window in ("fixed", "dynamic")
        self.k = k
        self.h = h
        self.p = p
        self.window = window
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.classes_ = np.unique(y)
        return self
    
    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.X - x, ord=self.p, axis=1)
    
    def predict(self, x: np.ndarray) -> int:
        x = x.ravel()
        distances = self._compute_distances(x)

        # Определяем радиус окна
        if self.window == "dynamic":
            sorted_distances = np.sort(distances)
            k_eff = min(self.k, len(sorted_distances) - 1)
            h_x = sorted_distances[k_eff]
            if h_x < 1e-10:
                h_x = 1.0
        else:
            h_x = self.h

        # Выбираем все точки, попавшие в окно
        inside_window = distances <= h_x

        if not np.any(inside_window):
            return self.y[np.argmin(distances)]

        window_distances = distances[inside_window]
        window_classes = self.y[inside_window]

        weights = self.gaussian_kernel(window_distances / h_x)

        scores = defaultdict(float)
        for w, cls in zip(weights, window_classes):
            scores[cls] += w

        return max(scores.items(), key=lambda x: x[1])[0]
    
    @staticmethod
    def gaussian_kernel(u: np.ndarray) -> np.ndarray:
        return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)
    
    def loo_estimate_k(self, k_values: list) -> tuple:
        X, y = self.X, self.y
        n_samples = len(X)
        
        best_k = k_values[0]
        best_error = float('inf')

        for k in k_values:
            errors = 0
            for i in range(n_samples):
                x_test = X[i]
                y_true = y[i]

                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(y, i)

                temp_clf = KNNClassifier(
                    k=k,
                    h=self.h,
                    p=self.p,
                    window=self.window 
                )
                temp_clf.fit(X_train, y_train)

                y_pred = temp_clf.predict(x_test)
                if y_pred != y_true:
                    errors += 1

            error_rate = errors / n_samples
            if error_rate < best_error:
                best_error = error_rate
                best_k = k
        
        return best_k, best_error
    
    def predict_batch(self, X_test: np.ndarray) -> np.ndarray:
        return np.array([self.predict(x) for x in X_test])
    
    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        y_pred = self.predict_batch(X_test)
        return np.mean(y_pred == y_test)
