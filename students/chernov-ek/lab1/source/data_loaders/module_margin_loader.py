from typing import Iterator
import numpy as np

from .abc_loader import ABCLoader


class ModuleMarginLoader(ABCLoader):
    def __init__(self):
        super().__init__()

    def get_data(self, X: np.ndarray, y: np.ndarray) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        pass