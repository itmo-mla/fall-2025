import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def load_and_prepare_data(file_path="heart.csv"):
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "nareshbhat/health-care-data-set-on-heart-attack-possibility",
        file_path
    )

    df['target'] = df['target'] * 2 - 1

    df.dropna(inplace=True)

    return df

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    return X_scaled

def train_test_split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

