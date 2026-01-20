import numpy as np
import matplotlib.pyplot as plt


def plot_svm_pca(model, X, y, resolution=200, title=""):
    # PCA
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean

    cov = np.cov(X_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1]
    P = eigvecs[:, idx[:2]]  # берём 2 главные компоненты

    # Проекция в двумерное пространство
    X2 = X_centered @ P     

    # Создание сетки
    x_min, x_max = X2[:,0].min() - 1, X2[:,0].max() + 1
    y_min, y_max = X2[:,1].min() - 1, X2[:,1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Обратная проекция сетки в исходное пространство 
    grid = np.c_[xx.ravel(), yy.ravel()]          # (N,2)
    grid_full = grid @ P.T + X_mean               # (N,d)

    # Вычисляем решающую функцию для всей сетки
    # Метод decision поддерживает линейное, RBF и полиномиальное ядра
    Z = model.decision(grid_full).reshape(xx.shape)

    plt.figure(figsize=(7,7))
    plt.contourf(xx, yy, Z, 40, cmap="RdBu_r", alpha=0.8)
    plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)

    plt.scatter(X2[y==1,0], X2[y==1,1], c="red", s=10, label="+1")
    plt.scatter(X2[y==-1,0], X2[y==-1,1], c="blue", s=10, label="-1")
    plt.legend()
    plt.title(title)
    plt.show()
