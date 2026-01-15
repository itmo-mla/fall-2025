import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA


def vis_LOO_errors(loo_errors, step=5):
    k_values = range(1, len(loo_errors) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, loo_errors, marker='o', linestyle='-')
    plt.title("Эмпирический риск (LOO ошибка) для разных k")
    plt.xlabel("k")
    plt.ylabel("LOO ошибка")
    
    # Подписи на оси X через step
    plt.xticks(k_values[::step])
    
    plt.grid(True)
    plt.show()


def visualize_knn_predictions(X, y, k_list, n_clusters_list, knn_class):
    """
    Визуализация KNN с прототипами в 2D (через PCA).
    - Точки окрашены по предсказаниям модели.
    - Крестики-прототипы того же цвета.
    - Фон убран.
    """

    # Приведение к numpy
    X_np = X.to_numpy() if hasattr(X, "to_numpy") else X
    y_np = y.to_numpy() if hasattr(y, "to_numpy") else y

    # PCA на 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_np)

    classes = np.unique(y_np)
    class_to_int = {cls: i for i, cls in enumerate(classes)}
    n_classes = len(classes)
    cmap = ListedColormap(plt.get_cmap(None, n_classes).colors)

    # Сетка под subplots
    fig, axes = plt.subplots(len(k_list), len(n_clusters_list),
                             figsize=(5 * len(n_clusters_list), 4 * len(k_list)),
                             sharex=True, sharey=True)
    if len(k_list) == 1:
        axes = axes[np.newaxis, :]
    if len(n_clusters_list) == 1:
        axes = axes[:, np.newaxis]

    for i, k in enumerate(k_list):
        for j, n_clusters in enumerate(n_clusters_list):
            ax = axes[i, j]

            # Обучаем KNN на исходных признаках
            model = knn_class(k)
            model.cluster_fit(X_np, y_np, n_clusters=n_clusters)

            # Предсказания для точек в PCA
            # Используем KNN на оригинальных признаках, но окрашиваем в PCA
            y_pred = model.predict(X_np)
            y_pred_int = np.array([class_to_int[p] for p in y_pred])

            # Рисуем точки по предсказанным цветам
            for cls in classes:
                cls_idx = class_to_int[cls]
                ax.scatter(X_2d[y_pred_int == cls_idx, 0],
                           X_2d[y_pred_int == cls_idx, 1],
                           label=f'Class {cls}',
                           color=cmap(cls_idx),
                           edgecolor='k', s=50)

            # Прототипы (крестики) в PCA
            proto_X_pca = pca.transform(model.X_train)
            proto_y_int = np.array([class_to_int[p] for p in model.y_train])
            for cls in classes:
                cls_idx = class_to_int[cls]
                ax.scatter(proto_X_pca[proto_y_int == cls_idx, 0],
                           proto_X_pca[proto_y_int == cls_idx, 1],
                           marker='X',
                           color=cmap(cls_idx),
                           s=180,
                           edgecolor='k')

            ax.set_title(f'k={k}, clusters={n_clusters}')
            ax.grid(True)

    axes[-1, 0].set_xlabel('PCA Component 1')
    axes[-1, 0].set_ylabel('PCA Component 2')
    plt.tight_layout()
    plt.show()
