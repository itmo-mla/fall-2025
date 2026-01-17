import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def show_support_vectors(X_train, y_train, support_vectors_indices):
    # PCA визуализация с опорными векторами для каждого ядра
    # Применяем PCA к train данным для отображения
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)

    n_kernels = len(support_vectors_indices)
    n_cols = min(3, n_kernels)  # максимум 3 графика в ряд
    n_rows = (n_kernels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (name, sv_indices) in zip(axes, support_vectors_indices.items()):
        # Маска для обычных точек (не опорных векторов)
        non_sv_mask = np.ones(len(X_train_pca), dtype=bool)
        non_sv_mask[sv_indices] = False

        # Обычные точки (круги)
        scatter1 = ax.scatter(X_train_pca[non_sv_mask, 0], X_train_pca[non_sv_mask, 1],
                              c=y_train[non_sv_mask], cmap='spring', alpha=0.5, s=40)

        # Опорные векторы (треугольники)
        ax.scatter(X_train_pca[sv_indices, 0], X_train_pca[sv_indices, 1],
                   c=y_train[sv_indices], cmap='spring', marker='^', s=120,
                   edgecolors='black', linewidths=1.5)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'{name}\nОпорных векторов: {len(sv_indices)}')

    # Добавляем общую легенду
    fig.legend(['Обычные точки', 'Опорные векторы (△)'], loc='upper right', fontsize=10)
    plt.suptitle('PCA визуализация train-данных с опорными векторами', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
