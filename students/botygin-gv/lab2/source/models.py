import numpy as np
from kernels import gaussian_kernel


class ParzenWindowKNN:
    """
    KNN с методом окна Парзена переменной ширины.
    """

    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.classes = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes = np.unique(y)

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            sorted_indices = np.argsort(distances)
            h = distances[sorted_indices[self.k+1]]

            if h == 0:
                nearest_idx = np.argmin(distances)
                pred_class = self.y_train[nearest_idx]
            else:
                weights = gaussian_kernel(distances / h)
                class_weights = {}
                for cls in self.classes:
                    class_weights[cls] = np.sum(weights[self.y_train == cls])
                pred_class = max(class_weights, key=class_weights.get)
            predictions.append(pred_class)
        return np.array(predictions)
