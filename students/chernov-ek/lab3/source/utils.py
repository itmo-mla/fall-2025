import kagglehub
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def linear_kernel(x, y):
    return np.dot(x, y)


def poly_kernel(x, y, degree=3):
    return (np.dot(x, y) + 0.0)**degree


def rbf_kernel(x, y, gamma):
    diff = x - y
    return np.exp(-gamma*np.dot(diff, diff))


def get_samples():
    # 1. Загрузка датасета
    src = Path(kagglehub.dataset_download("romaneyvazov/32-dsdds"))
    
    # 2. Извлечение данных
    df = pd.read_csv(src / "data_banknote_authentication.csv")

    # 3. Предобработка
    # Отделяем признаки от таргета
    y = df.pop("Class").replace(0, -1).to_numpy()
    X = df.to_numpy()
    # Разделение на выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        shuffle=True,
        stratify=y
    )
    # Нормализация данных
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Удаление кэша
    shutil.rmtree(src.parents[2])

    return X_train_scaled, X_test_scaled, y_train, y_test
