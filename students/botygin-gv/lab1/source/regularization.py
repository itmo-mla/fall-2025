import numpy as np


class L2Regularizer:
    def __init__(self, reg_strength: float = 1e-3):
        self.reg_strength = float(reg_strength)

    def __call__(self, w: np.ndarray) -> float:
        return self.reg_strength * np.sum(w ** 2)

    def gradient(self, w: np.ndarray) -> np.ndarray:
        return 2.0 * self.reg_strength * w