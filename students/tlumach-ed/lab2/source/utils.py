# загрузка датасета, масштабирование, LOO и метрики
"""Утилиты: загрузка датасета (скейлится), LOO-подбор k для KNNParzen и графики.
"""
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def load_default_dataset(name='wine'):
    # по умолчанию используем Wine (UCI) — компактный и сбалансированный
    if name == 'wine':
        data = datasets.load_wine()
    elif name == 'iris':
        data = datasets.load_iris()
    elif name == 'breast_cancer':
        data = datasets.load_breast_cancer()
    else:
        raise ValueError('unknown dataset')
    X = data.data
    y = data.target
    # стандартизируем признаки
    X = StandardScaler().fit_transform(X)
    return X, y


def loo_search_k(parzen, X, y, k_values):
    # возвращаем список LOO ошибок для каждого k из k_values
    errs = []
    for k in k_values:
        n = X.shape[0]
        bad = 0
        for i in range(n):
            # классифицируем xi по X\{i}
            parzen.fit(np.delete(X, i, axis=0), np.delete(y, i, axis=0))
            pred = parzen.predict(X[i:i+1], k=k)[0]
            if pred != y[i]:
                bad += 1
        errs.append(bad / n)
    return np.array(errs)


def compare_with_sklearn(X, y, best_k):
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X, y)
    acc = knn.score(X, y)
    return acc


def plot_risk_vs_k(k_values, errors, outpath='risk_vs_k.png'):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, errors, marker='o', label='LOO risk (KNN-Parzen)')
    plt.xlabel('k (для переменной ширины окна)')
    plt.ylabel('LOO ошибка')
    plt.title('Эмпирический риск (LOO) vs k')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()