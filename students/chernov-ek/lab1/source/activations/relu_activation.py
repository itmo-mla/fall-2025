import numpy as np

from source.activations import ABCActivation


class ReLUActivation(ABCActivation):
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        outputs = np.maximum(0, inputs)
        if self.learning:
            self.inputs = inputs.copy()
            self.outputs = outputs.copy()
        
        return outputs
    
    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        dI = np.where(self.outputs > 0., 1., 0.)
        return delta*dI
