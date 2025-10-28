import numpy as np

from source.activations import ABCActivation


class SignActivation(ABCActivation):
    def __call__(self, Z: np.ndarray) -> np.ndarray:
        # Расчёт выходов активации
        self.A = np.sign(Z)
        return self.A
    
    def pd_wrt_z(self, Z: np.ndarray) -> np.ndarray:
        self.dA_dZ = np.ones(Z.shape)
        return self.dA_dZ
