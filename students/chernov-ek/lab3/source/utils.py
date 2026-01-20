import kagglehub
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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


def vis_solutions(models, X_train, X_test, y_train, y_test, save_path):
    # Отрисуем Confusion matrix
    axes = plt.subplots(1, 2, figsize=(10, 4))[1]
    for ax, (name, model, y) in zip(axes, models):
        ConfusionMatrixDisplay(confusion_matrix(y_test, y)).plot(ax=ax)
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(save_path / "confusion_matrices.png", dpi=300)
    plt.close()

    # Отрисуем нашу разграничительную полосу
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Переиницализиурем модели и обучим их
    for _, model, _ in models:
        model.fit(X_train, y_train)

    # Создадим сетку
    xx, yy = np.meshgrid(
        np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 200),
        np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Рисуем разграничительную полосу и опорные вектора
    axes = plt.subplots(1, len(models), figsize=(12, 5))[1]
    for ax, (title, model, _) in zip(axes, models):
        Z = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
        
        sv = model.support_vectors if hasattr(model, "support_vectors") else model.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=100, facecolors="none")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path / "decision_boundaries.png", dpi=300)
    plt.close()
