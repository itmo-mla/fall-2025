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

def add_bias_column(X):
    return np.hstack((X, np.ones((X.shape[0], 1))))

def generate_multicollinear_data(n_samples=100, noise_std=0.01, random_state=42):
    np.random.seed(random_state)
    
    z1 = np.random.randn(n_samples)
    z2 = np.random.randn(n_samples)
    
    X = np.empty((n_samples, 10))
    
    for i in range(5):
        X[:, i] = z1 + noise_std * np.random.randn(n_samples)
    
    for i in range(5, 10):
        X[:, i] = z2 + noise_std * np.random.randn(n_samples)
    
    return X