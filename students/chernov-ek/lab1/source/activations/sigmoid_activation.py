import numpy as np

from source.activations import ABCActivation


class SigmoidActivation(ABCActivation):
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        if self.learning: self.inputs = inputs.copy()
        # Расчёт выходов активации для нейронов на слое
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs
    
    def pd_wrt_inputs(self) -> np.ndarray:
        self.dF_dI = self.outputs*(1 - self.outputs)
        return self.dF_dI
