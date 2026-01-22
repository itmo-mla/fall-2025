import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from knn import KNNParzenWindowEfficient


def compare_with_sklearn(X_train, y_train, X_test, y_test, k):

    print("\n" + "------ Сравнение с sklearn KNeighborsClassifier ------")
    
    print(f"\n1. Собственная реализация KNN (k={k}, balanced)...")
    our_knn = KNNParzenWindowEfficient(k=k, class_weights='balanced')
    our_knn.fit(X_train, y_train)
    our_predictions = our_knn.predict(X_test)
    
    our_metrics = calculate_metrics(y_test, our_predictions, label="Собственная реализация (balanced)")
    
    # sklearn реализация (с uniform weights)
    print(f"\n2. sklearn KNeighborsClassifier (k={k}, weights='uniform')...")
    sklearn_knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='euclidean')
    sklearn_knn.fit(X_train, y_train)
    sklearn_predictions = sklearn_knn.predict(X_test)
    
    sklearn_metrics = calculate_metrics(y_test, sklearn_predictions, label="sklearn (uniform)")
    
    # sklearn реализация (с distance weights)
    print(f"\n3. sklearn KNeighborsClassifier (k={k}, weights='distance')...")
    sklearn_knn_dist = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
    sklearn_knn_dist.fit(X_train, y_train)
    sklearn_dist_predictions = sklearn_knn_dist.predict(X_test)
    
    sklearn_dist_metrics = calculate_metrics(y_test, sklearn_dist_predictions, label="sklearn (distance)")
    
    print("\n" + "------ Сводная таблица метрик ------")
    
    header = f"{'Модель':<25} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-score':>12}"
    separator = f"{'─' * 80}"
    
    print(header)
    print(separator)
    
    # Собственная реализация
    print(
        f"{'Наша (Парзен)':<25} "
        f"{our_metrics['accuracy']:>12.4f} {our_metrics['precision']:>12.4f} "
        f"{our_metrics['recall']:>12.4f} {our_metrics['f1']:>12.4f}"
    )
    
    # sklearn uniform
    print(
        f"{'sklearn (uniform)':<25} "
        f"{sklearn_metrics['accuracy']:>12.4f} {sklearn_metrics['precision']:>12.4f} "
        f"{sklearn_metrics['recall']:>12.4f} {sklearn_metrics['f1']:>12.4f}"
    )
    
    # sklearn distance
    print(
        f"{'sklearn (distance)':<25} "
        f"{sklearn_dist_metrics['accuracy']:>12.4f} {sklearn_dist_metrics['precision']:>12.4f} "
        f"{sklearn_dist_metrics['recall']:>12.4f} {sklearn_dist_metrics['f1']:>12.4f}"
    )
    
    print(separator)
    
    return {
        'our': our_metrics,
        'sklearn_uniform': sklearn_metrics,
        'sklearn_distance': sklearn_dist_metrics
    }


def calculate_metrics(y_true, y_pred, label=""):

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    if label:
        print(f"\n{label}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-score:  {f1:.4f}")
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': y_pred
    }


def compare_with_without_prototypes(X_train, y_train, X_test, y_test, prototype_indices, k):
    
    print("\n" + "------ Сравнение KNN с полной выборкой и с эталонами ------")
    
    # KNN на полной обучающей выборке
    print(f"\n1. KNN на полной обучающей выборке ({len(X_train)} объектов)...")
    knn_full = KNNParzenWindowEfficient(k=k, class_weights='balanced')
    knn_full.fit(X_train, y_train)
    full_predictions = knn_full.predict(X_test)
    
    full_metrics = calculate_metrics(y_test, full_predictions, label="KNN (полная выборка)")
    
    # KNN на эталонах
    X_prototypes = X_train[prototype_indices]
    y_prototypes = y_train[prototype_indices]
    
    print(f"\n2. KNN на эталонах ({len(X_prototypes)} объектов)...")
    knn_prototypes = KNNParzenWindowEfficient(k=min(k, len(X_prototypes)), class_weights='balanced')
    knn_prototypes.fit(X_prototypes, y_prototypes)
    proto_predictions = knn_prototypes.predict(X_test)
    
    proto_metrics = calculate_metrics(y_test, proto_predictions, label="KNN (эталоны)")
    
    # Вычисляем разницу в метриках
    acc_diff = proto_metrics['accuracy'] - full_metrics['accuracy']
    prec_diff = proto_metrics['precision'] - full_metrics['precision']
    rec_diff = proto_metrics['recall'] - full_metrics['recall']
    f1_diff = proto_metrics['f1'] - full_metrics['f1']
    
    compression = len(X_prototypes) / len(X_train)
    
    print("\n" + "------ Сводка ------")
    
    # Заголовок таблицы
    header = f"{'Модель':<25} {'Размер':>12} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-score':>12}"
    separator = f"{'─' * 80}"
    
    print(header)
    print(separator)
    
    # Полная выборка
    print(
        f"{'KNN (полная)':<25} {len(X_train):>12} "
        f"{full_metrics['accuracy']:>12.4f} {full_metrics['precision']:>12.4f} "
        f"{full_metrics['recall']:>12.4f} {full_metrics['f1']:>12.4f}"
    )
    
    # Эталоны
    print(
        f"{'KNN (эталоны)':<25} {len(X_prototypes):>12} "
        f"{proto_metrics['accuracy']:>12.4f} {proto_metrics['precision']:>12.4f} "
        f"{proto_metrics['recall']:>12.4f} {proto_metrics['f1']:>12.4f}"
    )
    
    print(separator)
    
    # Разница
    print(
        f"{'Разница':<25} {f'{compression:.1%}':>12} "
        f"{acc_diff:>+12.4f} {prec_diff:>+12.4f} "
        f"{rec_diff:>+12.4f} {f1_diff:>+12.4f}"
    )
    
    print(separator)
    
    return {
        'full': full_metrics,
        'prototypes': proto_metrics,
        'compression_ratio': compression,
        'accuracy_diff': acc_diff
    }


def plot_comparison_bar(comparison_results, save_path=None):
    """
    Строит столбчатую диаграмму сравнения метрик.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    models = ['our', 'sklearn_uniform', 'sklearn_distance']
    labels = ['Собственная (Парзен)', 'sklearn (uniform)', 'sklearn (distance)']
    
    data = np.array([[comparison_results[model][metric] for metric in metrics] for model in models])
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i in range(len(models)):
        offset = (i - 1) * width
        ax.bar(x + offset, data[i], width, label=labels[i], color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Метрики', fontsize=12)
    ax.set_ylabel('Значение', fontsize=12)
    ax.set_title('Сравнение реализаций KNN', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-score'])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сравнения сохранен в: {save_path}")
    
    plt.close()


def plot_prototype_comparison(full_metrics, proto_metrics, compression_ratio, save_path=None):
    """
    Строит график сравнения KNN с полной выборкой и с эталонами.
    
    Args:
        full_metrics: метрики для полной выборки
        proto_metrics: метрики для эталонов
        compression_ratio: степень сжатия
        save_path: путь для сохранения
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    
    full_data = [full_metrics[m] for m in metrics]
    proto_data = [proto_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, full_data, width, label='Полная выборка', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, proto_data, width, label=f'Эталоны ({compression_ratio:.1%})', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Метрики', fontsize=12)
    ax.set_ylabel('Значение', fontsize=12)
    ax.set_title('Сравнение KNN: полная выборка vs эталоны', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    # Добавляем значения на столбцах
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сравнения с эталонами сохранен в: {save_path}")
    
    plt.close()


def print_confusion_matrix_detailed(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n------ {title} ------")
    print("                  Predicted")
    print("             Class 0    Class 1")
    print(f"Actual  0    {cm[0, 0]:6d}     {cm[0, 1]:6d}")
    print(f"        1    {cm[1, 0]:6d}     {cm[1, 1]:6d}")
    print(f"{'─' * 40}")
    
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")
    print(f"\nSpecificity (TNR): {specificity:.4f}")
    print(f"Sensitivity (TPR): {sensitivity:.4f}")
    print(f"{'─' * 40}")


def analyze_errors(X, y_true, y_pred, title="Анализ ошибок классификации", save_path=None):
    from sklearn.decomposition import PCA
    
    errors = y_true != y_pred
    n_errors = np.sum(errors)
    accuracy = np.mean(y_true == y_pred)
    
    # PCA для визуализации
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(12, 8))
    
    # Правильно классифицированные объекты
    correct_mask = ~errors
    plt.scatter(
        X_pca[correct_mask & (y_true == 0), 0],
        X_pca[correct_mask & (y_true == 0), 1],
        c='lightcoral', alpha=0.5, s=30, label='Класс 0 (правильно)', edgecolors='none'
    )
    plt.scatter(
        X_pca[correct_mask & (y_true == 1), 0],
        X_pca[correct_mask & (y_true == 1), 1],
        c='lightblue', alpha=0.5, s=30, label='Класс 1 (правильно)', edgecolors='none'
    )
    
    # Ошибки классификации
    if n_errors > 0:
        plt.scatter(
            X_pca[errors, 0], X_pca[errors, 1],
            c='black', marker='X', s=200,
            label=f'Ошибки ({n_errors}, {n_errors/len(y_true)*100:.1f}%)',
            edgecolors='yellow', linewidths=2, zorder=10
        )
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    
    info_text = f'Accuracy: {accuracy:.3f}\nОшибок: {n_errors}/{len(y_true)}'
    plt.text(
        0.02, 0.98, info_text,
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontsize=10
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Анализ ошибок сохранен в: {save_path}")
    
    plt.close()


def visualize_prototype_selection_process(X, y, prototype_history, save_path=None):
    from sklearn.decomposition import PCA
    
    if not prototype_history or len(prototype_history) == 0:
        print("Нет истории для визуализации")
        return
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    n_steps = len(prototype_history)
    if n_steps <= 4:
        steps_to_show = list(range(n_steps))
    else:
        steps_to_show = [
            0,
            n_steps // 3,
            2 * n_steps // 3,
            n_steps - 1
        ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, step_idx in enumerate(steps_to_show):
        ax = axes[idx]
        prototype_indices = prototype_history[step_idx]
        
        all_indices = set(range(len(X)))
        non_prototype_mask = np.array([i not in prototype_indices for i in range(len(X))])
        
        ax.scatter(
            X_pca[non_prototype_mask & (y == 0), 0],
            X_pca[non_prototype_mask & (y == 0), 1],
            c='red', alpha=0.3, s=20, label='Класс 0'
        )
        ax.scatter(
            X_pca[non_prototype_mask & (y == 1), 0],
            X_pca[non_prototype_mask & (y == 1), 1],
            c='blue', alpha=0.3, s=20, label='Класс 1'
        )
        
        # Эталоны
        if len(prototype_indices) > 0:
            ax.scatter(
                X_pca[prototype_indices, 0],
                X_pca[prototype_indices, 1],
                c='gold', s=250, marker='*', edgecolors='black',
                linewidth=2, label='Эталоны', zorder=10
            )
        
        ax.set_title(
            f'Итерация {step_idx + 1}: {len(prototype_indices)} эталонов',
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Процесс отбора эталонов', fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Визуализация процесса отбора сохранена в: {save_path}")
    
    plt.close()


def compare_kernels(X_train, y_train, X_test, y_test, k_range=None, kernels=None):

    from knn import KNNParzenWindowEfficient
    
    if k_range is None:
        k_range = range(1, 31)
    
    if kernels is None:
        kernels = ['gaussian', 'rectangular', 'triangular', 'epanechnikov']
    
    print("\n" + "------ Сравнение ядер ------")
    
    results = {}
    
    for kernel in kernels:
        print(f"\nТестирование ядра: {kernel}")
        accuracies = []
        
        for k in k_range:
            knn = KNNParzenWindowEfficient(k=k, kernel=kernel, class_weights='balanced')
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            acc = np.mean(y_pred == y_test)
            accuracies.append(acc)
        
        best_k = k_range[np.argmax(accuracies)]
        best_acc = np.max(accuracies)
        
        results[kernel] = {
            'accuracies': accuracies,
            'best_k': best_k,
            'best_accuracy': best_acc
        }
        
        print(f"  Лучшее k: {best_k}, Accuracy: {best_acc:.4f}")
    
    return results


def plot_kernel_comparison(results, k_range, save_path=None):

    plt.figure(figsize=(12, 7))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, (kernel, data) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        plt.plot(
            list(k_range), data['accuracies'],
            label=f"{kernel} (best k={data['best_k']})",
            linewidth=2, marker='o', markersize=4, color=color, alpha=0.7
        )
        
        best_idx = np.argmax(data['accuracies'])
        plt.plot(
            list(k_range)[best_idx], data['accuracies'][best_idx],
            '*', markersize=15, color=color
        )
    
    plt.xlabel('Количество соседей (k)', fontsize=12)
    plt.ylabel('Accuracy на тесте', fontsize=12)
    plt.title('Сравнение различных ядер Парзена', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сравнения ядер сохранен в: {save_path}")
    
    plt.close()
