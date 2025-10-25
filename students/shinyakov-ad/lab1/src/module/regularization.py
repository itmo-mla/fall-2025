from abc import ABC, abstractmethod
import numpy as np

class BaseRegularization(ABC):
    NAME = None

    def __init__(self, reg_koef = 0.01):
        self.reg_koef = reg_koef

    @abstractmethod
    def calculate(self, weights):
        raise NotImplementedError("Regularization calculate method is not implemented")

    @abstractmethod
    def derivative(self, weights):
        raise NotImplementedError("Regularization derivative method is not implemented")

    def __call__(self, weights):
        return self.calculate(weights)

class L2Regularization(BaseRegularization):
    NAME = "l2_reg"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def calculate(self, weights):
        return self.reg_koef * np.sum(weights ** 2)
    
    def derivative(self, weights):
        return 2 * self.reg_koef * weights
    
REGULARIZATIONS = {
    v.NAME: v
    for v in globals().values()
    if isinstance(v, type) and issubclass(v, BaseRegularization) and v is not BaseRegularization
}
