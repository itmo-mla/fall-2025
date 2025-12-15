import numpy as np


def compute_euclidean_distances(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
    cross = X1 @ X2.T
    distances_squared = X1_sq + X2_sq - 2 * cross
    distances_squared = np.maximum(distances_squared, 0)  # <-- защита от отрицательных чисел
    distances = np.sqrt(distances_squared)
    return distances