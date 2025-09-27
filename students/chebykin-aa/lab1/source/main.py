import random
import shutil
import os

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, hinge_loss
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from core.model import BinaryClassificator

def categorize(
    margins,
    threshold_low: float = -0.3,
    threshold_high: float = 0.3
)-> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Метод, разделяющий отступы на 3 группы(Шумовые, Пограничные, Надежные)"""
    noisy = margins < threshold_low
    border = (margins >= threshold_low) & (margins <= threshold_high)
    reliable = margins > threshold_high
    return noisy, border, reliable

def draw_margins_plot(
    train_margins: np.ndarray,
    test_margins: np.ndarray,
    plot_path: str,
    threshold_low: float = -0.3,
    threshold_high: float = 0.3
):
    """Метод для создания графика отступов на тренировочной/тестовой выборках"""
    # Сортируем марджины по значениям
    train_margins = np.sort(train_margins)
    test_margins = np.sort(test_margins)

    # Получим индексы объектов в выборках
    train_idx = np.arange(len(train_margins))
    test_idx = np.arange(len(test_margins))

    # Разделим объекты по значению отступа
    train_noisy, train_border, train_reliable = categorize(train_margins, threshold_low, threshold_high)
    test_noisy, test_border, test_reliable = categorize(test_margins, threshold_low, threshold_high)

    # Отрисуем графики отступов для тренировочной/тестовой выборок
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    axes[0].bar(train_idx[train_noisy], train_margins[train_noisy], color='red', label='Шумовые', width=0.9)
    axes[0].bar(train_idx[train_border], train_margins[train_border], color='yellow', label='Пограничные', width=0.9)
    axes[0].bar(train_idx[train_reliable], train_margins[train_reliable], color='green', label='Надежные', width=0.9)
    axes[0].plot(train_idx, train_margins, linestyle='-', color='blue')
    axes[0].set_xlabel('Индекс элемента')
    axes[0].set_ylabel('Отступ')
    axes[0].set_title('График отступов на тренировочных данных')
    axes[0].legend()

    axes[1].bar(test_idx[test_noisy], test_margins[test_noisy], color='red', label='Шумовые', width=0.9)
    axes[1].bar(test_idx[test_border], test_margins[test_border], color='yellow', label='Пограничные', width=0.9)
    axes[1].bar(test_idx[test_reliable], test_margins[test_reliable], color='green', label='Надежные', width=0.9)
    axes[1].plot(test_idx, test_margins, linestyle='-', color='blue')
    axes[1].set_xlabel('Индекс элемента')
    axes[1].set_ylabel('Отступ')
    axes[1].set_title('График отступов на тестовых данных')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)

def draw_train_stats(
    train_loss: dict,
    train_q: dict,
    lr: dict,
    plot_path: str
):
    """Метод для создания графиков, характеризующих обучение моделей"""
    # Отрисуем графики отступов для тренировочной/тестовой выборок
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    axes[0].plot(train_loss.keys(), train_loss.values(), linestyle='-', color='red')
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=10))
    axes[0].set_xlabel('Номер эпохи')
    axes[0].set_ylabel('Значение функции потерь')
    axes[0].set_title('График изменения значения функции потерь во время обучения')

    axes[1].plot(train_q.keys(), train_q.values(), linestyle='-', color='red')
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=10))
    axes[1].set_xlabel('Номер эпохи')
    axes[1].set_ylabel('Значение эмпирического риска')
    axes[2].set_title('График изменения значения эмпирического риска во время обучения')

    axes[2].plot(lr.keys(), lr.values(), linestyle='-', color='red')
    axes[2].xaxis.set_major_locator(MaxNLocator(nbins=10))
    axes[2].set_xlabel('Номер эпохи')
    axes[2].set_ylabel('Значение шага обучения')
    axes[2].set_title('График изменения значения шага обучения во время обучения')

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)

def calculate_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    split: str,
    file_path: str
):
    """Метод, вычисляющий качество предсказания моделей"""
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)
    q = hinge_loss(labels, preds)

    metrics_str = (
        f"split: {split}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
        f"Q: {q:.4f}\n"
        f"\n"
    )

    # Write to file
    with open(file_path, "a") as file:
        file.write(metrics_str)

def main():
    # Зафиксируем seed
    seed = 1234
    np.random.seed(seed)
    random.seed(seed)
    # Создадим директорию для хранения результатов
    results_path = f"{os.getcwd()}/students/chebykin-aa/lab1/source/results"
    shutil.rmtree(results_path)
    os.makedirs(results_path)

    # Получим датасет 
    data = load_breast_cancer()
    features = pd.DataFrame(data.data, columns = data.feature_names)
    print(features.head(), end="\n\n")
    print(features.describe(), end="\n\n")
    # Конвертируем метки в {-1; 1},
    labels = pd.Series(data.target)
    labels = labels.map({0: -1, 1: 1})
    print("Статистика меток:")
    print(f'Уникальные значения: {labels.unique()}')
    print(f'Распределение по классам:\n{labels.value_counts()}', end="\n\n")
    # Отнормируем признаки в [-1; 1]
    scaler = MinMaxScaler(feature_range = (-1, 1))
    features = pd.DataFrame(scaler.fit_transform(features), columns = features.columns, index = features.index)
    print(features.describe(), end="\n\n")

    # Проверим, линейно ли разделимы классы
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features)
    df_pca = pd.DataFrame(pca_features, columns=['PC1', 'PC2'])
    plt.figure(figsize=(8,6))
    plt.scatter(df_pca['PC1'], df_pca['PC2'], c=labels, cmap='viridis', s=100, edgecolor='k')
    plt.xlabel('компонента 1')
    plt.ylabel('компонента 2')
    plt.title('График главных компонент, показывающих линейную разделимость классов')
    plt.tight_layout()
    plt.savefig(f"{results_path}/class_boundaries.png")
    plt.close()

    # Разделим датасет на выборки
    X_train, X_test, y_train, y_test = [
        arr.to_numpy() for arr in train_test_split(features, labels, test_size = 0.2, random_state = seed)
    ]
    print("Размер выборок:")
    print(f"X_train.shape: {X_train.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_test.shape: {y_test.shape}")

    # Инициализируем собственные модели с оптимизацией h
    custom_model1 = BinaryClassificator(
        init_type = "none",
        subsampling_type = "margin_abs",
        lr = 1e-4,
        reg_coef = 0.3,
        m = 3,
        momentum = 0.7,
        nesterov = True,
        h_optimization = True,
        n_iters = 150
    )

    custom_model2 = BinaryClassificator(
        init_type = "corr",
        subsampling_type = "margin_abs",
        lr = 1e-4,
        reg_coef = 0.3,
        m = 3,
        momentum = 0.7,
        nesterov = True,
        h_optimization = True,
        n_iters = 150
    )

    custom_model3 = BinaryClassificator(
        init_type = "multi_start",
        subsampling_type = "margin_abs",
        lr = 1e-4,
        reg_coef = 0.3,
        m = 3,
        momentum = 0.7,
        nesterov = True,
        h_optimization = True,
        n_iters = 150
    )

    custom_model4 = BinaryClassificator(
        init_type = "multi_start",
        subsampling_type = "random",
        lr = 1e-4,
        reg_coef = 0.3,
        m = 3,
        momentum = 0.7,
        nesterov = True,
        h_optimization = True,
        n_iters = 150
    )

    # Обучаем и тестируем их
    for idx, custom_model in enumerate([custom_model1, custom_model2, custom_model3, custom_model4]):
        custom_model.fit(X_train, y_train)
        loss_values, q_values, lr_values = custom_model.get_train_info()
        train_preds, train_labels = custom_model.predict(X_train)
        test_preds, test_labels = custom_model.predict(X_test)
        draw_margins_plot(
            (train_preds * y_train),
            (test_preds * y_test),
            plot_path = f"{results_path}/own_binclf{idx+1}_with_h_opt_margins.png"
        )
        draw_train_stats(
            loss_values,
            q_values,
            lr_values,
            plot_path = f"{results_path}/own_binclf{idx+1}_with_h_opt_train_stats.png"
        )
        calculate_metrics(
            train_labels,
            y_train,
            split = "train",
            file_path = f"{results_path}/own_binclf{idx+1}_with_h_opt_metrics.txt"
        )
        calculate_metrics(
            test_labels,
            y_test,
            split = "test",
            file_path = f"{results_path}/own_binclf{idx+1}_with_h_opt_metrics.txt"
        )

    # Инициализируем собственные модели без оптимизации h
    custom_model1 = BinaryClassificator(
        init_type = "none",
        subsampling_type = "margin_abs",
        lr = 1e-4,
        reg_coef = 0.3,
        m = 3,
        momentum = 0.7,
        nesterov = True,
        h_optimization = False,
        n_iters = 150
    )

    custom_model2 = BinaryClassificator(
        init_type = "corr",
        subsampling_type = "margin_abs",
        lr = 1e-4,
        reg_coef = 0.3,
        m = 3,
        momentum = 0.7,
        nesterov = True,
        h_optimization = False,
        n_iters = 150
    )

    custom_model3 = BinaryClassificator(
        init_type = "multi_start",
        subsampling_type = "margin_abs",
        lr = 1e-4,
        reg_coef = 0.3,
        m = 3,
        momentum = 0.7,
        nesterov = True,
        h_optimization = False,
        n_iters = 150
    )

    custom_model4 = BinaryClassificator(
        init_type = "multi_start",
        subsampling_type = "random",
        lr = 1e-4,
        reg_coef = 0.3,
        m = 3,
        momentum = 0.7,
        nesterov = True,
        h_optimization = False,
        n_iters = 150
    )

    # Обучаем и тестируем их
    for idx, custom_model in enumerate([custom_model1, custom_model2, custom_model3, custom_model4]):
        custom_model.fit(X_train, y_train)
        loss_values, q_values, lr_values = custom_model.get_train_info()
        train_preds, train_labels = custom_model.predict(X_train)
        test_preds, test_labels = custom_model.predict(X_test)
        draw_margins_plot(
            (train_preds * y_train),
            (test_preds * y_test),
            plot_path = f"{results_path}/own_binclf{idx+1}_without_h_opt_margins.png"
        )
        draw_train_stats(
            loss_values,
            q_values,
            lr_values,
            plot_path = f"{results_path}/own_binclf{idx+1}_without_h_opt_train_stats.png"
        )
        calculate_metrics(
            train_labels,
            y_train,
            split = "train",
            file_path = f"{results_path}/own_binclf{idx+1}_without_h_opt_metrics.txt"
        )
        calculate_metrics(
            test_labels,
            y_test,
            split = "test",
            file_path = f"{results_path}/own_binclf{idx+1}_without_h_opt_metrics.txt"
        )

    # Обучим и протестируем модели из библиотеки sklearn
    svm = SGDClassifier(
        loss="hinge",
        penalty="l2",
        eta0=1e-4,
        learning_rate="constant",
        max_iter=150,
        tol=1e-3
    )
    log_reg = SGDClassifier(
        loss="log_loss", 
        penalty="l2",
        eta0=1e-4,
        learning_rate="constant",
        max_iter=150,
        tol=1e-3
    )
    for name, model in zip(["svm", "logreg"], [svm, log_reg]):
        model.fit(X_train, y_train)
        train_preds, train_labels = model.decision_function(X_train), model.predict(X_train)
        test_preds, test_labels = model.decision_function(X_test), model.predict(X_test)
        draw_margins_plot(
           (train_preds * y_train),
           (test_preds * y_test),
            plot_path = f"{results_path}/{name}_margins.png"
        )
        calculate_metrics(
            train_labels,
            y_train,
            split = "train",
            file_path = f"{results_path}/{name}_metrics.txt"
        )
        calculate_metrics(
            test_labels,
            y_test,
            split = "test",
            file_path = f"{results_path}/{name}_metrics.txt"
        )

if __name__ == "__main__":
    main()

