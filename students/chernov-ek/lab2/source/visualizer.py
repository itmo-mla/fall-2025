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


def visualize_knn_predictions(
        knn_class,
        X_train, X_test, y_train, y_test,
        k_list,
        max_prototypes, e=0.1
    ):
    """
    Визуализация KNN с прототипами в 2D (через PCA).
    - Точки окрашены по предсказаниям модели.
    - Крестики-прототипы того же цвета.
    """
    # PCA на 2D
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    classes = np.unique(y_train)
    class_to_int = {cls: i for i, cls in enumerate(classes)}
    n_classes = len(classes)
    cmap = ListedColormap(plt.get_cmap(None, n_classes).colors)

    # Сетка под subplots
    _, axes = plt.subplots(1, len(k_list),
                           figsize=(4*len(k_list), 5),
                           sharex=True, sharey=True)

    for i, k in enumerate(k_list):
        ax = axes[i]

        # Обучаем KNN на исходных признаках
        model = knn_class(k)
        model.ccv_fit(X_train, y_train, max_prototypes, e)

        # Предсказания для точек в PCA
        y_pred = model.predict(X_test)
        y_pred = np.array([class_to_int[p] for p in y_pred])

        # Рисуем точки по предсказанным цветам
        for cls in classes:
            cls_idx = class_to_int[cls]
            ax.scatter(X_test[y_pred == cls_idx, 0],
                        X_test[y_pred == cls_idx, 1],
                        label=f'Class {cls}',
                        color=cmap(cls_idx),
                        edgecolor='k', s=50)

        # Прототипы (крестики) в PCA
        y_proto = np.array([class_to_int[p] for p in model.y_train])
        for cls in classes:
            cls_idx = class_to_int[cls]
            ax.scatter(model.X_train[y_proto == cls_idx, 0],
                       model.X_train[y_proto == cls_idx, 1],
                       marker='X',
                       color=cmap(cls_idx),
                       s=180,
                       edgecolor='k')

        ax.set_title(f'k={k}')
        ax.grid(True)

    plt.tight_layout()
    plt.show()
