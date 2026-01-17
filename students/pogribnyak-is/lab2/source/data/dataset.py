from abc import ABC, abstractmethod

import numpy as np
from pandas import DataFrame

from data.scaler import DefaultScaler


class Dataset(ABC):
    def __init__(self, target_col: str = 'target', seed: int = 42):
        self.seed = seed
        self.target_col = target_col
        self.scaler: DefaultScaler = DefaultScaler()

        self.df: DataFrame = DataFrame()
        self.X: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.y: np.ndarray = np.empty((0,), dtype=np.float32)

        self.X_train: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.X_test: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.y_train: np.ndarray = np.empty((0,), dtype=np.float32)
        self.y_test: np.ndarray = np.empty((0,), dtype=np.float32)

        self.load_data()

    @abstractmethod
    def load(self) -> DataFrame:
        pass

    @abstractmethod
    def preprocess(self) -> DataFrame:
        pass

    def load_data(self):
        self.df = self.load()
        self.df = self.preprocess()

        self.X = self.df.drop(columns=[self.target_col]).to_numpy(dtype=np.float32)
        self.y = self.df[self.target_col].to_numpy()

    def split_indices(self, n_samples: int, test_size: float = 0.2):
        np.random.seed(self.seed)
        indices = np.random.permutation(n_samples)

        test_count = int(n_samples * test_size)

        test_idx = indices[:test_count]
        train_idx = indices[test_count:]

        return train_idx, test_idx

    def scale_data(self):
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def split_and_scale(self, test_size: float = 0.2):
        train_idx, test_idx = self.split_indices(len(self.X), test_size)

        self.X_train, self.y_train = self.X[train_idx], self.y[train_idx]
        self.X_test, self.y_test = self.X[test_idx], self.y[test_idx]

        self.scale_data()

        return self.X_train, self.y_train, self.X_test, self.y_test


