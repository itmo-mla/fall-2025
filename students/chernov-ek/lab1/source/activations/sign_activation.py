import numpy as np

from source.activations import ABCActivation


class SignActivation(ABCActivation):
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        outputs = np.sign(inputs)
        if self.learning:
            self.inputs = inputs.copy()
            self.outputs = outputs.copy()
        
        return outputs
    
    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        dI = np.ones(self.inputs.shape)
        return delta*dI
