import numpy as np
from typing import Iterator

from source.data_loaders import ABCLoader


class BaseLoader(ABCLoader):
    def get_data(self, X: np.ndarray, y: np.ndarray) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        yield 0, X, y
