from typing import Iterator
import numpy as np

from source.data_loaders import ABCLoader


class ShuffleLoader(ABCLoader):
    def __init__(self, batch_size: int = 32):
        super().__init__()

        self.batch_size = batch_size

    def get_data(self, X: np.ndarray, y: np.ndarray) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        n_samples = len(y)
        for n_batch in range(n_samples//self.batch_size):
            inds_batch = np.random.choice(n_samples, size=self.batch_size, replace=False)
            yield n_batch, X[inds_batch], y[inds_batch]
