import numpy as np
from typing import Tuple, Protocol


class Decomposition(Protocol):
    @staticmethod
    def compute(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...


class SVDDecomposition:
    @staticmethod
    def compute(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        return U, S, Vt
