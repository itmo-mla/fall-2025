import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from data.dataset import Dataset


class BreastCancerDataset(Dataset):
    def __init__(self, seed: int = 42):
        super().__init__(target_col='target', seed=seed)

    def load(self) -> pd.DataFrame:
        cancer = load_breast_cancer()
        df = pd.DataFrame(
            data=np.c_[cancer['data'], cancer['target']],
            columns=list(cancer['feature_names']) + ['target']
        )
        return df

    def preprocess(self) -> pd.DataFrame:
        return self.df

