import logging
import random
import shutil
import os

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
import numpy as np

from core.utils import calculate_metrics
from core.model import OwnSVM

def run_pipeline(
        X_train,
        X_test,
        y_train,
        y_test,
        kernel,
        C = 1.0,
        degree = 2,
        gamma = 0.5,
        save_path = None
):
    logging.info(f"Запустили пайплайн с ядром {kernel}!")

    # Проинициализируем нашу модель
    svm_custom = OwnSVM(kernel = kernel, C = C, degree = degree, gamma = gamma)
    svm_custom.fit(X_train, y_train)
    logging.info("Обучили OwnSVM!")
    y_pred_custom = svm_custom.predict(X_test)
    calculate_metrics(y_pred_custom, y_test, "test", f"{save_path}/OwnSVM_metrics.txt")

    # Проинициализируем модель-эталон
    svm_sklearn = SVC(kernel = kernel, C = C, degree = degree, gamma = gamma)
    svm_sklearn.fit(X_train, y_train)
    logging.info("Обучили Sklearn SVM!")
    y_pred_sklearn = svm_sklearn.predict(X_test)
    calculate_metrics(y_pred_sklearn, y_test, "test", f"{save_path}/Sklearn_SVM_metrics.txt")

    # Отрисуем Confusion matrix
    axes = plt.subplots(1, 2, figsize=(10, 4))[1]
    model_preds = [("OwnSVM", y_pred_custom), ("Sklearn SVM", y_pred_sklearn)]
    for ax, (title, y_pred) in zip(axes, model_preds):
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confusion_matrices.png"), dpi=300)
    plt.close()
    logging.info("Отрисовали Confusion matrix!")

    # Отрисуем нашу разграничительную полосу
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Переиницализиурем модели и обучим их
    models = [
        ("OwnSVM", OwnSVM(kernel = kernel, C = C, degree = degree, gamma = gamma)),
        ("Sklearn SVM", SVC(kernel = kernel, C = C, degree = degree, gamma = gamma))
    ]
    for _, model in models:
        model.fit(X_train, y_train)

    # Создадим сетку
    xx, yy = np.meshgrid(
        np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 200),
        np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # отрисуем разграничительную полосу и опорные вектора
    axes = plt.subplots(1, len(models), figsize=(12, 5))[1]
    for ax, (title, model) in zip(axes, models):
        Z = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
        
        sv = model.support_vectors if hasattr(model, "support_vectors") else model.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=100, facecolors="none")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "decision_boundaries.png"), dpi=300)
    plt.close()
    logging.info("Отрисовали разграничительную полосу и опорные вектора!")

def main():
    # Зафиксируем seed и другие гиперпараметры
    SEED = 42
    kernels = ["linear", "poly", "rbf"]
    np.random.seed(SEED)
    random.seed(SEED)

    # Создадим директорию для хранения результатов
    results_path = f"{os.getcwd()}/students/chebykin-aa/lab3/source/results"
    if os.path.exists(results_path):
        shutil.rmtree(results_path)

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(f"{results_path}/linear", exist_ok=True)
    os.makedirs(f"{results_path}/poly", exist_ok=True)
    os.makedirs(f"{results_path}/rbf", exist_ok=True)

    # Настроим логи
    logging.basicConfig(
        filename=f"{results_path}/main.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Получим датасет 
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns = data.feature_names)
    logging.info(f"\n{X.head()}")
    logging.info(f"\n{X.describe()}")
    # Получим метки
    y = pd.Series(data.target)
    y = y.map({0: -1, 1: 1})
    logging.info("Статистика меток:")
    logging.info(f'Уникальные значения: {y.unique()}')
    logging.info(f'Распределение по классам:\n{y.value_counts()}')
    # Отнормируем признаки в [0; 1]
    scaler = MinMaxScaler(feature_range = (0, 1))
    X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns, index = X.index)
    logging.info(f"\n{X.describe()}")

    # Сконвертируем данные в необходимый формат
    X = X.to_numpy()
    y = y.to_numpy()

    # Разобьем на трейн/тест выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )   

    for kernel in kernels:
        run_pipeline(X_train, X_test, y_train, y_test, kernel, save_path = f"{results_path}/{kernel}")

if __name__ == "__main__":
    main()