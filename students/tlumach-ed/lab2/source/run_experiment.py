import numpy as np
from utils import load_default_dataset, loo_search_k, plot_risk_vs_k, compare_with_sklearn
from knn_parzen import KNNParzen
from prototypes import greedy_remove
import matplotlib.pyplot as plt


def main():
    X, y = load_default_dataset('wine')
    parzen = KNNParzen()

    k_values = list(range(1, min(51, X.shape[0])))
    print('Запускаю LOO-поиск по k')
    errors = loo_search_k(parzen, X, y, k_values)
    best_idx = int(np.argmin(errors))
    best_k = k_values[best_idx]
    best_err = errors[best_idx]
    print(f'Лучший k = {best_k}, LOO ошибка = {best_err:.4f}')

    plot_risk_vs_k(k_values, errors, outpath='risk_vs_k.png')
    print('График risk_vs_k.png сохранён')

    # сравнение со sklearn (эталонная реализация kNN)
    acc_sklearn = compare_with_sklearn(X, y, best_k)
    print(f'Accuracy sklearn KNN (k={best_k}) на всей выборке: {acc_sklearn:.4f}')

    # Отбор эталонов (жадное удаление)
    print('Запускаю жадное удаление эталонов...')
    classifier_ctor = lambda: KNNParzen()
    proto_remove, err_remove = greedy_remove(X, y, classifier_ctor)
    print(f'После удаления: {len(proto_remove)} эталонов, LOO ошибка = {err_remove:.4f}')

    # Визуализация (PCA 2D) — для интуиции
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X)
        plt.figure(figsize=(7, 6))
        for cls in np.unique(y):
            mask = y == cls
            plt.scatter(X2[mask, 0], X2[mask, 1], label=f'class {cls}', alpha=0.5)
        prot_mask = np.zeros(X.shape[0], dtype=bool)
        prot_mask[proto_remove] = True
        plt.scatter(X2[prot_mask, 0], X2[prot_mask, 1], edgecolors='k', s=80, facecolors='none', label='prototypes')
        plt.legend()
        plt.title('PCA 2D + выбранные эталоны (после удаления)')
        plt.xlabel('PC1'); plt.ylabel('PC2')
        plt.tight_layout(); plt.savefig('prototypes_pca.png', dpi=150); plt.close()
        print('Визуализация prototypes_pca.png сохранена')
    except Exception as e:
        print('PCA не доступна:', e)

    # Сравним качество KNN с/без отбора эталонов (на всей выборке, как приближённая метрика)
    parzen.fit(X, y)
    preds_full = parzen.predict(X, k=best_k)
    acc_full = np.mean(preds_full == y)
    parzen.fit(X[proto_remove], y[proto_remove])
    preds_proto = parzen.predict(X, k=best_k)
    acc_proto = np.mean(preds_proto == y)
    print(f'KNN-Parzen accuracy на всех данных (k={best_k}): {acc_full:.4f}')
    print(f'KNN-Parzen accuracy с выбранными эталонами (k={best_k}): {acc_proto:.4f}')


if __name__ == '__main__':
    main()