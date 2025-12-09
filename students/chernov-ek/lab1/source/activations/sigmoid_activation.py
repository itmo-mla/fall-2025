import numpy as np

from source.activations import ABCActivation


class SigmoidActivation(ABCActivation):
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        # Расчёт выходов активации для нейронов на слое
        outputs = 1 / (1 + np.exp(-inputs))
        if self.learning:
            self.inputs = inputs.copy()
            self.outputs = outputs.copy()
        
        return outputs
    
    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        dI = self.outputs*(1 - self.outputs)
        return delta*dI
