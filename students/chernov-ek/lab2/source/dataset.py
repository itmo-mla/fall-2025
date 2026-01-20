import os
import shutil
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split


def load_iris_dataset(random_state: int = 42):
    # Download latest version
    path_save = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    file_path = os.path.join(path_save, '2/Iris.csv')
    if not os.path.exists(file_path):
        path = kagglehub.dataset_download(handle="uciml/iris")

        shutil.move(path, path_save)
        print("Path to dataset files:", path_save)

    data = pd.read_csv(file_path)
    X = data.drop(['Id', 'Species'], axis=1)
    y = data['Species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state, shuffle=True, stratify=y_train)

    return X_train, X_val, X_test, y_train, y_val, y_test
