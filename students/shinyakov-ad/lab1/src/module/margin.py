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

    def calculate(self, pred_probs, expected_classes):
        return pred_probs * expected_classes
    
    def visualize(self, margins: np.array):
        sorted_margins = np.sort(margins)

        plt.plot(sorted_margins, marker=".", linestyle="-", color="blue", alpha=0.7)
        plt.axhline(0, color="red", linestyle="--", label="Граница (M=0)")
        plt.xlabel("Объекты (отсортированные)")
        plt.ylabel("Отступ (margin)")
        plt.title("Margin по возрастанию")
        plt.legend()
        plt.grid()
        plt.show()