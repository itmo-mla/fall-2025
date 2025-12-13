import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA


def plot_risk(errors, k_range):
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, errors, marker='o')
    plt.title('Эмпирический риск (LOO) для разных k')
    plt.xlabel('k')
    plt.ylabel('LOO Ошибка')
    plt.grid(True)
    plt.savefig("images/risk.png", dpi=150)
    plt.show()


def visualize_prototype_selection(X, y, X_reduced, y_reduced):
    n_classes = len(np.unique(y))
    n_features = X.shape[1]

    if n_classes > n_features + 1:
        # В этом случае снижаем размерность до n_classes - 1 с помощью PCA сначала
        print(f"Число классов ({n_classes}) > число признаков ({n_features}) + 1. Применяю PCA перед LDA.")
        pca = PCA(n_components=min(n_features, n_classes - 1))
        X_pca = pca.fit_transform(X)
        X_red_pca = pca.transform(X_reduced)
        lda = LinearDiscriminantAnalysis(n_components=min(2, n_classes - 1))
        X_2d = lda.fit_transform(X_pca, y)
        X_red_2d = lda.transform(X_red_pca)
    else:
        # Стандартный LDA
        lda = LinearDiscriminantAnalysis(n_components=min(2, n_classes - 1))
        X_2d = lda.fit_transform(X, y)
        X_red_2d = lda.transform(X_reduced)

    plt.figure(figsize=(10, 6))

    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.4, s=30, label='Все объекты')

    plt.scatter(X_red_2d[:, 0], X_red_2d[:, 1], c=y_reduced, cmap='tab10',
                edgecolor='k', s=120, linewidth=1.5, label='Отобранные эталоны')

    plt.title(f"Отбор эталонов (всего: {len(X)}, отобрано: {len(X_reduced)})")
    plt.xlabel("LDA компонента 1")
    plt.ylabel("LDA компонента 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/prototype_selection.png", dpi=150)
    plt.show()