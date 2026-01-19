import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from data.dataset import Dataset


class DiabetesDataset(Dataset):
    def __init__(self, seed: int = 42):
        super().__init__(target_col='target', seed=seed)

    def load(self) -> pd.DataFrame:
        diabetes = load_diabetes()
        df = pd.DataFrame(
            data=np.c_[diabetes['data'], diabetes['target']],
            columns=list(diabetes['feature_names']) + ['target']
        )
        return df

    def preprocess(self) -> pd.DataFrame:
        df = self.df.copy()
        df = df.dropna()
        return df