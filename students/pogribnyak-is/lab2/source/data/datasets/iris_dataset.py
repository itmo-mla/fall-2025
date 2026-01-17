import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from data.dataset import Dataset


class IrisDataset(Dataset):
    def __init__(self, seed: int = 42):
        super().__init__(target_col='target', seed=seed)
    
    def load(self) -> pd.DataFrame:
        iris = load_iris()
        df = pd.DataFrame(
            data=np.c_[iris['data'], iris['target']],
            columns=iris['feature_names'] + ['target']
        )
        return df
    
    def preprocess(self) -> pd.DataFrame:
        return self.df

