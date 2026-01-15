from typing import Optional
import numpy as np
from abc import ABC, abstractmethod


class Sampler(ABC):
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, random_seed: Optional[int] = None):
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
        self.batch_size = batch_size
        self.rng = np.random.default_rng(random_seed)

    def _form_data_batches(self, indices: np.ndarray):
        n_batches = int(np.ceil(self.n_samples / self.batch_size))
        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            batch_idx_slice = indices[start:start + self.batch_size]
            yield batch_idx, self.X[batch_idx_slice], self.y[batch_idx_slice]

    @abstractmethod
    def get_batches(self, pred: Optional[np.ndarray] = None, batch_size: Optional[int] = None):
        pass


class RandomSampler(Sampler):
    def get_batches(self, pred: Optional[np.ndarray] = None, batch_size: Optional[int] = None):
        if batch_size is not None:
            self.batch_size = batch_size
        indices = self.rng.choice(self.n_samples, size=self.n_samples, replace=False)
        yield from self._form_data_batches(indices)


class MarginSampler(Sampler):
    def get_batches(self, pred: Optional[np.ndarray] = None, batch_size: Optional[int] = None):
        if batch_size is not None:
            self.batch_size = batch_size

        margins = pred * self.y
        abs_margin = np.clip(np.abs(margins), 1e-6, None)
        probs = 1.0 / abs_margin
        probs /= probs.sum()

        indices = self.rng.choice(self.n_samples, size=self.n_samples, replace=False, p=probs)
        yield from self._form_data_batches(indices)