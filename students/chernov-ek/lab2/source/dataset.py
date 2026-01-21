from pathlib import Path
import shutil
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split


def load_iris_dataset(random_state: int = 42):
    # Download latest version    
    path_data = Path(kagglehub.dataset_download(handle="uciml/iris"))
    if path_data.exists():
        data = pd.read_csv(path_data / "Iris.csv")
        shutil.rmtree(path_data.parents[2])

        X = data.drop(['Id', 'Species'], axis=1).to_numpy()
        y = data['Species'].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, shuffle=True, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state, shuffle=True, stratify=y_train)

        return X_train, X_val, X_test, y_train, y_val, y_test
