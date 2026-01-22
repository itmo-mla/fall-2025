import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

from knn import KNNParzenWindowEfficient


def loo_cross_validation(X, y, k):
    """
    Выполняет Leave-One-Out кросс-валидацию для заданного k.
    
    Args:
        X: матрица признаков
        y: метки классов
        k: количество ближайших соседей
    
    Returns:
        float: ошибка классификации (доля неправильных предсказаний)
    """
    n_samples = len(X)
    errors = 0
    
    for i in range(n_samples):
        # Обучающая выборка без i-го объекта
        X_train_loo = np.delete(X, i, axis=0)
        y_train_loo = np.delete(y, i, axis=0)
        
        # Тестовый объект
        X_test_loo = X[i:i+1]
        y_test_loo = y[i]
        
        # Обучаем и предсказываем
        knn = KNNParzenWindowEfficient(k=k)
        knn.fit(X_train_loo, y_train_loo)
        prediction = knn.predict(X_test_loo)[0]
        
        if prediction != y_test_loo:
            errors += 1
    
    error_rate = errors / n_samples
    return error_rate


def loo_cross_validation_efficient(X, y, k):
    """
    Эффективная реализация LOO с использованием полной матрицы расстояний.
    
    Args:
        X: матрица признаков
        y: метки классов
        k: количество ближайших соседей
    
    Returns:
        float: ошибка классификации
    """
    n_samples = len(X)
    
    # Вычисляем матрицу расстояний один раз
    X_norm_sq = np.sum(X ** 2, axis=1, keepdims=True)
    distances_sq = X_norm_sq + X_norm_sq.T - 2 * np.dot(X, X.T)
    distances_sq = np.maximum(distances_sq, 0)
    distances = np.sqrt(distances_sq)
    
    errors = 0
    
    for i in range(n_samples):
        # Расстояния от i-го объекта до всех остальных (исключая самого себя)
        dists_i = distances[i].copy()
        dists_i[i] = np.inf
        
        k_nearest_indices = np.argsort(dists_i)[:k]
        k_nearest_distances = dists_i[k_nearest_indices]
        k_nearest_labels = y[k_nearest_indices]
        
        # Ширина окна
        h = k_nearest_distances[-1] if len(k_nearest_distances) > 0 and k_nearest_distances[-1] > 0 else 1.0
        
        if h == 0:
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            prediction = unique[np.argmax(counts)]
        else:
            # Вычисляем веса
            weights = np.exp(-(k_nearest_distances ** 2) / (2 * h ** 2))
            
            # Взвешенное голосование
            class_weights = {}
            for label, weight in zip(k_nearest_labels, weights):
                class_weights[label] = class_weights.get(label, 0) + weight
            
            prediction = max(class_weights, key=class_weights.get)
        
        if prediction != y[i]:
            errors += 1
    
    error_rate = errors / n_samples
    return error_rate


def loo_cross_validation_f1_score(X, y, k, class_weights='balanced', pos_label=1):
    """
    Выполняет LOO с вычислением F1-score для заданного класса.
    
    Args:
        X: матрица признаков
        y: метки классов
        k: количество ближайших соседей
        class_weights: веса классов для KNN
        pos_label: метка положительного класса для F1-score
    
    Returns:
        float: F1-score для положительного класса
    """
    n_samples = len(X)
    predictions = np.zeros(n_samples, dtype=y.dtype)
    
    # Вычисляем матрицу расстояний один раз
    X_norm_sq = np.sum(X ** 2, axis=1, keepdims=True)
    distances_sq = X_norm_sq + X_norm_sq.T - 2 * np.dot(X, X.T)
    distances_sq = np.maximum(distances_sq, 0)
    distances = np.sqrt(distances_sq)
    
    # Вычисляем веса классов вручную для LOO
    if class_weights == 'balanced':
        class_counts = np.bincount(y)
        total = len(y)
        cw = {cls: total / (len(np.unique(y)) * count) for cls, count in enumerate(class_counts)}
    else:
        cw = {cls: 1.0 for cls in np.unique(y)}
    
    for i in range(n_samples):
        # Расстояния от i-го объекта до всех остальных
        dists_i = distances[i].copy()
        dists_i[i] = np.inf
        
        # Находим k ближайших соседей
        k_nearest_indices = np.argsort(dists_i)[:k]
        k_nearest_distances = dists_i[k_nearest_indices]
        k_nearest_labels = y[k_nearest_indices]
        
        # Ширина окна
        h = k_nearest_distances[-1] if len(k_nearest_distances) > 0 and k_nearest_distances[-1] > 0 else 1.0
        
        if h == 0:
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            predictions[i] = unique[np.argmax(counts)]
        else:
            # Вычисляем веса с учетом ядра и весов классов
            kernel_weights = np.exp(-(k_nearest_distances ** 2) / (2 * h ** 2))
            
            # Взвешенное голосование
            class_votes = {}
            for label, weight in zip(k_nearest_labels, kernel_weights):
                class_weight = cw.get(label, 1.0)
                class_votes[label] = class_votes.get(label, 0) + weight * class_weight
            
            predictions[i] = max(class_votes, key=class_votes.get)
    
    # Вычисляем F1-score для положительного класса
    f1 = f1_score(y, predictions, pos_label=pos_label, zero_division=0)
    return f1


def select_optimal_k(X, y, k_range=None, use_efficient=True, verbose=True, 
                     metric='error', class_weights='balanced', pos_label=1):
    """
    Подбирает оптимальное значение k методом LOO кросс-валидации.
    
    Args:
        X: матрица признаков
        y: метки классов
        k_range: диапазон значений k для тестирования
        use_efficient: использовать эффективную реализацию LOO
        verbose: выводить прогресс
        metric: 'error' для минимизации ошибки, 'f1' для максимизации F1-score
        class_weights: веса классов для KNN (используется при metric='f1')
        pos_label: метка положительного класса для F1-score
    
    Returns:
        dict: результаты с оптимальным k и историей метрик
    """
    if k_range is None:
        k_range = range(1, min(51, len(X)))
    
    k_values = list(k_range)
    loo_scores = []
    
    if metric == 'f1':
        print(f"\nПодбор оптимального k методом LOO с метрикой F1-score для класса {pos_label}...")
        print(f"Используются веса классов: {class_weights}")
        
        for i, k in enumerate(k_values):
            if verbose and (i % 5 == 0 or i == len(k_values) - 1):
                print(f"Тестирование k={k} ({i+1}/{len(k_values)})...", end='\r')
            
            f1 = loo_cross_validation_f1_score(X, y, k, class_weights=class_weights, pos_label=pos_label)
            loo_scores.append(f1)
        
        if verbose:
            print()
        
        # Находим оптимальное k (максимальный F1-score)
        optimal_idx = np.argmax(loo_scores)
        optimal_k = k_values[optimal_idx]
        optimal_score = loo_scores[optimal_idx]
        
        print(f"\nОптимальное k = {optimal_k} с F1-score = {optimal_score:.4f}")
        
        return {
            'optimal_k': optimal_k,
            'optimal_score': optimal_score,
            'optimal_error': 1 - optimal_score,
            'k_values': k_values,
            'loo_scores': loo_scores,
            'loo_errors': [1 - s for s in loo_scores],
            'metric': 'f1'
        }
    else:
        loo_func = loo_cross_validation_efficient if use_efficient else loo_cross_validation
        
        print(f"\nПодбор оптимального k методом LOO (всего {len(k_values)} значений)...")
        
        for i, k in enumerate(k_values):
            if verbose and (i % 5 == 0 or i == len(k_values) - 1):
                print(f"Тестирование k={k} ({i+1}/{len(k_values)})...", end='\r')
            
            error = loo_func(X, y, k)
            loo_scores.append(error)
        
        if verbose:
            print()
        
        # Находим оптимальное k (минимальная ошибка)
        optimal_idx = np.argmin(loo_scores)
        optimal_k = k_values[optimal_idx]
        optimal_error = loo_scores[optimal_idx]
        
        print(f"\nОптимальное k = {optimal_k} с ошибкой LOO = {optimal_error:.4f}")
        
        return {
            'optimal_k': optimal_k,
            'optimal_error': optimal_error,
            'optimal_score': 1 - optimal_error,
            'k_values': k_values,
            'loo_errors': loo_scores,
            'loo_scores': [1 - e for e in loo_scores],
            'metric': 'error'
        }


def plot_loo_errors(k_values, loo_errors, optimal_k, save_path=None):
    """
    Строит график эмпирического риска (ошибки LOO) в зависимости от k.
    
    Args:
        k_values: список значений k
        loo_errors: соответствующие ошибки LOO
        optimal_k: оптимальное значение k
        save_path: путь для сохранения графика
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(k_values, loo_errors, 'b-o', linewidth=2, markersize=5, alpha=0.7, label='LOO ошибка')
    
    # Отмечаем оптимальное k
    optimal_idx = k_values.index(optimal_k)
    optimal_error = loo_errors[optimal_idx]
    plt.plot(optimal_k, optimal_error, 'r*', markersize=20, label=f'Оптимальное k={optimal_k}')
    
    plt.xlabel('Количество соседей (k)', fontsize=12)
    plt.ylabel('Эмпирический риск (LOO ошибка)', fontsize=12)
    plt.title('Подбор параметра k методом Leave-One-Out кросс-валидации', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    plt.annotate(
        f'k={optimal_k}\nошибка={optimal_error:.4f}',
        xy=(optimal_k, optimal_error),
        xytext=(optimal_k + len(k_values) * 0.1, optimal_error + 0.02),
        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График LOO ошибок сохранен в: {save_path}")
    
    plt.close()


def analyze_k_sensitivity(k_values, loo_errors):
    """
    Анализирует чувствительность модели к выбору k.
    
    Args:
        k_values: список значений k
        loo_errors: соответствующие ошибки LOO
    
    Returns:
        dict: статистика по чувствительности
    """
    errors_array = np.array(loo_errors)
    
    top_5_indices = np.argsort(errors_array)[:5]
    top_5_k = [k_values[i] for i in top_5_indices]
    top_5_errors = [errors_array[i] for i in top_5_indices]
    
    # Статистика
    stats = {
        'mean_error': np.mean(errors_array),
        'std_error': np.std(errors_array),
        'min_error': np.min(errors_array),
        'max_error': np.max(errors_array),
        'top_5_k': top_5_k,
        'top_5_errors': top_5_errors,
    }
    
    print("\n" + "------ Анализ чувствительности к параметру k ------")
    print(f"Средняя ошибка LOO: {stats['mean_error']:.4f}")
    print(f"Стандартное отклонение: {stats['std_error']:.4f}")
    print(f"Минимальная ошибка: {stats['min_error']:.4f}")
    print(f"Максимальная ошибка: {stats['max_error']:.4f}")
    print("\nТоп-5 значений k:")
    for i, (k, err) in enumerate(zip(top_5_k, top_5_errors), 1):
        print(f"  {i}. k={k:2d} -> ошибка={err:.4f}")
    print(f"{'─' * 60}")
    
    return stats

