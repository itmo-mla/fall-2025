from typing import Protocol

import numpy as np


class ComponentSelector(Protocol):
    def select(self, explained_variance_ratio: np.ndarray) -> int: ...


class VarianceThresholdSelector(ComponentSelector):
    def __init__(self, threshold: float = 0.95): self.threshold = threshold

    def select(self, explained_variance_ratio: np.ndarray) -> int: return int(np.searchsorted(np.cumsum(explained_variance_ratio), self.threshold) + 1)