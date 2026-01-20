from pandas import DataFrame, concat
from data.dataset import Dataset
from sklearn.preprocessing import OneHotEncoder
import kagglehub
from kagglehub import KaggleDatasetAdapter


class DrugDataset(Dataset):
    def __init__(self, seed: int = 42):
        self.num_features = ['Age', 'Na_to_K']
        self.cat_features = ['Sex', 'BP', 'Cholesterol']
        super().__init__(target_col='target', seed=seed)

    def load(self) -> DataFrame:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "prathamtripathi/drug-classification",
            "drug200.csv"
        )
        return df

    def preprocess(self) -> DataFrame:
        df = self.df.copy()
        df[self.cat_features] = df[self.cat_features].astype(str)
        ohe = OneHotEncoder(sparse_output=False)
        cat_encoded = ohe.fit_transform(df[self.cat_features])
        cat_cols = ohe.get_feature_names_out(self.cat_features)
        df_cat = DataFrame(cat_encoded, columns=cat_cols, index=df.index)
        df_num = df[self.num_features].astype(float)
        df_preprocessed = concat([df_num, df_cat], axis=1)
        df_preprocessed['target'] = df['Drug']
        return df_preprocessed