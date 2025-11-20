import numpy as np

from source.activations import ABCActivation


class SignActivation(ABCActivation):
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        # Расчёт выходов активации
        if self.learning: self.inputs = inputs.copy()
        self.outputs = np.sign(inputs)
        return self.outputs
    
    def pd_wrt_inputs(self) -> np.ndarray:
        self.dF_dI = np.ones(self.inputs.shape)
        return self.dF_dI
