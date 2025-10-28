import numpy as np
from abc import ABC, abstractmethod
from typing import Iterator


class ABCLoader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_data(self, X: np.ndarray, y: np.ndarray) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        raise NotImplementedError()
