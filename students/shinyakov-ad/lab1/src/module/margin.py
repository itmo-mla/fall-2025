from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

class BaseMargin(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def calculate(self, pred_probs, expected_classes):
        raise NotImplementedError("Margin count function should be implemented")

class BinaryClassificationMargin(BaseMargin):
    def __init__(self, bin_width: float = 0.25):
        self.bin_width = bin_width
        self.lower_bound = -1
        self.upper_bound = 0.5

    def calculate(self, pred_probs, expected_classes):
        return pred_probs * expected_classes
    
    def visualize(self, margins: np.array):
        sorted_margins = np.sort(margins)
        
        indices = np.arange(len(sorted_margins))

        plt.plot(indices, sorted_margins, marker='.', linestyle='-', color='blue', alpha=0.7)
        
        plt.axhline(0, color='red', linestyle='--', label='Граница (M=0)')
        
        plt.fill_between(indices, sorted_margins, where=(sorted_margins <= self.lower_bound), interpolate=True, color='red', alpha=1, label='Выбросы')
        plt.fill_between(indices, sorted_margins, where=(np.logical_and(sorted_margins > self.lower_bound, sorted_margins <= self.upper_bound)), interpolate=True, color='yellow', alpha=0.5, label='Пограничные объекты')
        plt.fill_between(indices, sorted_margins, where=(sorted_margins >= self.upper_bound), interpolate=True, color='green', alpha=0.5, label='Достоверные объекты')
        
        plt.xlabel('Объекты')
        plt.ylabel('Отступ')
        plt.title('Margin по возрастанию')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()