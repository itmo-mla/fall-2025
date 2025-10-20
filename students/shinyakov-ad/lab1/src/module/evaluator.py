from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    NAME = None

    def __init__(self):
        self.history = []
        self.previous_evaluation = 0
    
    @abstractmethod
    def eval(self, loss):
        raise NotImplementedError("Evaluate function is not implemented")
    
    def save_evaluation(self):
        self.history.append(self.previous_evaluation)

    def refresh_current_evaluation(self):
        self.previous_evaluation = 0

    def get_evaluation_history(self):
        return self.history

    def __call__(self, loss):
        return self.eval(loss)

class ReccurentEvaluator(BaseEvaluator):
    NAME = "reccurent"
    
    def __init__(self, coef = 0.001):
        super().__init__()
        self.coef = coef

    def eval(self, loss):
        self.previous_evaluation = (1 - self.coef) * self.previous_evaluation + (self.coef) * loss

EVALUATORS = {
    v.NAME: v
    for v in globals().values()
    if isinstance(v, type) and issubclass(v, BaseEvaluator) and v is not BaseEvaluator
}