import abc 

class Loss(abc.ABC):
    @abc.abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abc.abstractmethod
    def derivative(self, y_true, y_pred):
        pass