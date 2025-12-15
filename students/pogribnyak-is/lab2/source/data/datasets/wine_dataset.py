import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from data.dataset import Dataset


class WineDataset(Dataset):
    def __init__(self, seed: int = 42):
        super().__init__(target_col='target', seed=seed)
    
    def load(self) -> pd.DataFrame:
        wine = load_wine()
        df = pd.DataFrame(
            data=np.c_[wine['data'], wine['target']],
            columns=wine['feature_names'] + ['target']
        )
        return df
    
    def preprocess(self) -> pd.DataFrame:
        return self.df

