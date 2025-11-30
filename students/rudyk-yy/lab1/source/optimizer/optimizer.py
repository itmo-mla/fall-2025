import abc 

class Optimizer(abc.ABC):
    @abc.abstractmethod
    def update(self, weights, gradient, learning_rate):
        pass