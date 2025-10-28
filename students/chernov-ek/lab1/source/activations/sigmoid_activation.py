import numpy as np

from source.activations import ABCActivation


class SigmoidActivation(ABCActivation):
    def __call__(self, Z: np.ndarray) -> np.ndarray:
        # Расчёт выходов активации для нейронов на слое
        self.A = 1 / (1 + np.exp(-Z))
        # Инициализируем хранение производной
        self.dA_dZ = np.empty((0, self.A.shape[0]))
        return self.A
    
    def pd_wrt_z(self, i_neuron: int) -> np.ndarray:
        if self.A.ndim == 1:
            dA_dz = self.A*(1 - self.A)
        else:
            dA_dz = self.A[:, i_neuron]*(1 - self.A[:, i_neuron])
        
        self.dA_dZ = np.vstack((self.dA_dZ, dA_dz))
        return dA_dz
