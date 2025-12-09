import numpy as np

from source.layers import ABCLayer
from source.weights_initializers import random_numbers_init


class LinearLayer(ABCLayer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        weights = np.array([random_numbers_init(in_features + 1 if bias else in_features) for _ in range(out_features)])
        self.W = weights[:, :-1] if bias else weights
        self.b = weights[:, -1] if bias else None

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        outputs = inputs@self.W.T if self.b is None else inputs@self.W.T + self.b
        if self.learning:
            self.inputs = inputs.copy()
            self.outputs = outputs.copy()
        
        return outputs
        
    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        # Считаем частные производные слоя весов по весам [n_samples, n_inputs]
        dW = np.column_stack((self.inputs, np.ones(len(self.inputs)))) if any(self.b) else self.inputs
        # Частная производная по входам
        dI = self.W
        # Частная производная Loss по весу нейрона: dL_dW = dL_dA*dF_dI*dZ_dW
        gradients = delta.T@dW / len(self.inputs)
        self.gradW = gradients[:, :-1] if any(self.b) else gradients
        self.gradb = gradients[:, -1] if any(self.b) else None
        # Передаём ошибку дальше влево через веса
        return delta@dI
