import numpy as np

from .abc_layer import ABCLayer
from source.weights_initializers import random_numbers_init


class LinearLayer(ABCLayer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        weights = np.array([random_numbers_init(in_features + 1 if bias else in_features) for _ in range(out_features)])
        self.W = weights[:, :-1] if bias else weights
        self.b = weights[:, -1] if bias else np.zeros(out_features)
        
        self.X = None
        self.Z = None

    def __call__(self, X: np.ndarray) -> np.ndarray:
        self.X = X.copy()
        self.Z = self.X@self.W.T + self.b
        return self.Z

    def get_weights(self) -> np.ndarray:
         return np.column_stack((self.W, self.b)) if any(self.b) else self.W
    
    def get_size(self) -> tuple[int]:
        return self.get_weights().shape
    
    def get_gradients(self) -> np.ndarray | None:
        return self.gradients
    
    def update_weights(self, weights: np.ndarray):
        # Валидируем веса
        self._validate_weights_size(weights)

        # Обновляем веса
        if any(self.b):
            self.W, self.b = weights[:, :-1], weights[:, -1]
        else:
            self.W = weights
    
    def pd_wrt_w(self) -> np.ndarray | None:
        return np.column_stack((self.X, np.ones(len(self.X)))) if any(self.b) else self.X

    def _validate_weights_size(self, weights: np.ndarray):
        error = ''
        if any(self.b):
            if (len(weights[0]) - 1) != len(self.W[0]):
                error = 'Используется смещение, но значение отстуствует.'
        else:
            if len(weights[0]) != len(self.W[0]):
                error = f'Не совпадают размеры весов {len(weights)}x{len(weights[0])} и {len(self.W)}x{len(self.W[0])}.'

        assert len(error) == 0, error
