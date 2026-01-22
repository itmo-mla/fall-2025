import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

from knn import KNNParzenWindowEfficient


def condensed_nearest_neighbor(X, y, k=1, max_iterations=100, verbose=True):
    """
    Алгоритм отбора эталонов Condensed Nearest Neighbor (CNN).
    
    Алгоритм:
    1. Начинаем с одного случайного объекта из каждого класса
    2. Классифицируем все остальные объекты с помощью текущих эталонов
    3. Добавляем в эталоны объекты, которые были неправильно классифицированы
    4. Повторяем, пока не будет достигнута сходимость
    
    Args:
        X: матрица признаков
        y: метки классов
        k: количество соседей для KNN
        max_iterations: максимальное количество итераций
        verbose: выводить прогресс
    
    Returns:
        dict: словарь с индексами эталонов, эталонами и метками
    """
    n_samples = len(X)
    classes = np.unique(y)
    
    if verbose:
        print(f"\nАлгоритм отбора эталонов CNN")
        print(f"Исходное количество объектов: {n_samples}")
        print(f"Количество классов: {len(classes)}")
    
    prototype_indices = []
    for c in classes:
        class_indices = np.where(y == c)[0]
        prototype_indices.append(np.random.choice(class_indices))
    
    prototype_indices = set(prototype_indices)
    
    for iteration in range(max_iterations):
        old_size = len(prototype_indices)
        
        X_prototypes = X[list(prototype_indices)]
        y_prototypes = y[list(prototype_indices)]
        
        knn = KNNParzenWindowEfficient(k=min(k, len(X_prototypes)))
        knn.fit(X_prototypes, y_prototypes)
        
        predictions = knn.predict(X)
        
        misclassified = np.where(predictions != y)[0]
        
        prototype_indices.update(misclassified)
        
        new_size = len(prototype_indices)
        
        if verbose:
            print(f"Итерация {iteration + 1}: {new_size} эталонов (добавлено {new_size - old_size})")
        
        # Проверка сходимости
        if new_size == old_size:
            if verbose:
                print("Сходимость достигнута!")
            break
    
    prototype_indices = sorted(list(prototype_indices))
    X_prototypes = X[prototype_indices]
    y_prototypes = y[prototype_indices]
    
    compression_ratio = len(prototype_indices) / n_samples
    
    if verbose:
        print(f"\nФинальное количество эталонов: {len(prototype_indices)}")
        print(f"Степень сжатия: {compression_ratio:.2%}")
        print("Распределение эталонов по классам:")
        for c in classes:
            count = np.sum(y_prototypes == c)
            print(f"  Класс {c}: {count} эталонов")
    
    return {
        'indices': prototype_indices,
        'X_prototypes': X_prototypes,
        'y_prototypes': y_prototypes,
        'compression_ratio': compression_ratio,
        'n_prototypes': len(prototype_indices)
    }


def stolp_algorithm(X, y, k=5, threshold=0.0, verbose=True):
    """
    Алгоритм отбора эталонов STOLP (STandard Objects for Learning Patterns).
    
    Более продвинутый алгоритм, основанный на отступах объектов.
    
    Args:
        X: матрица признаков
        y: метки классов
        k: количество соседей для KNN
        threshold: порог отступа для выбора эталонов
        verbose: выводить прогресс
    
    Returns:
        dict: словарь с эталонами
    """
    n_samples = len(X)
    classes = np.unique(y)
    
    if verbose:
        print(f"\nАлгоритм отбора эталонов STOLP")
        print(f"Исходное количество объектов: {n_samples}")
    
    # Вычисляем матрицу расстояний
    X_norm_sq = np.sum(X ** 2, axis=1, keepdims=True)
    distances_sq = X_norm_sq + X_norm_sq.T - 2 * np.dot(X, X.T)
    distances = np.sqrt(np.maximum(distances_sq, 0))
    
    # Отступ = расстояние до ближайшего объекта другого класса - расстояние до ближайшего объекта своего класса
    margins = np.zeros(n_samples)
    
    for i in range(n_samples):
        same_class = y == y[i]
        diff_class = ~same_class
        
        # Расстояние до ближайшего объекта своего класса (исключая сам объект)
        dists_same = distances[i].copy()
        dists_same[i] = np.inf
        dists_same[diff_class] = np.inf
        nearest_same = np.min(dists_same) if np.any(same_class & (np.arange(n_samples) != i)) else 0
        
        # Расстояние до ближайшего объекта другого класса
        dists_diff = distances[i].copy()
        dists_diff[same_class] = np.inf
        nearest_diff = np.min(dists_diff) if np.any(diff_class) else np.inf
        
        margins[i] = nearest_diff - nearest_same
    
    # Начинаем с объектов с наименьшими отступами (пограничные объекты)
    # Также добавляем по одному объекту с максимальным отступом из каждого класса
    prototype_indices = set()
    
    for c in classes:
        class_mask = y == c
        class_indices = np.where(class_mask)[0]
        class_margins = margins[class_indices]
        
        # Добавляем объект с минимальным отступом (наиболее сложный)
        min_margin_idx = class_indices[np.argmin(class_margins)]
        prototype_indices.add(min_margin_idx)
        
        # Добавляем объект с максимальным отступом (типичный представитель)
        max_margin_idx = class_indices[np.argmax(class_margins)]
        prototype_indices.add(max_margin_idx)
    
    # Добавляем объекты с отступом ниже порога
    low_margin_indices = np.where(margins < threshold)[0]
    prototype_indices.update(low_margin_indices)
    
    # Проверяем качество и добавляем объекты, которые неправильно классифицируются
    prototype_indices = list(prototype_indices)
    X_prototypes = X[prototype_indices]
    y_prototypes = y[prototype_indices]
    
    knn = KNNParzenWindowEfficient(k=min(k, len(X_prototypes)))
    knn.fit(X_prototypes, y_prototypes)
    predictions = knn.predict(X)
    misclassified = np.where(predictions != y)[0]
    
    prototype_indices = set(prototype_indices)
    prototype_indices.update(misclassified)
    prototype_indices = sorted(list(prototype_indices))
    
    X_prototypes = X[prototype_indices]
    y_prototypes = y[prototype_indices]
    
    compression_ratio = len(prototype_indices) / n_samples
    
    if verbose:
        print(f"\nФинальное количество эталонов: {len(prototype_indices)}")
        print(f"Степень сжатия: {compression_ratio:.2%}")
        print("Распределение эталонов по классам:")
        for c in classes:
            count = np.sum(y_prototypes == c)
            print(f"  Класс {c}: {count} эталонов")
    
    return {
        'indices': prototype_indices,
        'X_prototypes': X_prototypes,
        'y_prototypes': y_prototypes,
        'compression_ratio': compression_ratio,
        'n_prototypes': len(prototype_indices),
        'margins': margins
    }


def visualize_prototypes_pca(X, y, prototype_indices, title="Визуализация эталонов (PCA)", save_path=None):
    """
    Визуализирует эталоны в 2D пространстве с помощью PCA.
    
    Args:
        X: матрица признаков
        y: метки классов
        prototype_indices: индексы эталонов
        title: заголовок графика
        save_path: путь для сохранения
    """
    from sklearn.decomposition import PCA
    
    # Применяем PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Создаем маски
    prototype_mask = np.zeros(len(X), dtype=bool)
    prototype_mask[prototype_indices] = True
    
    plt.figure(figsize=(12, 8))
    
    classes = np.unique(y)
    colors = ['red', 'blue']
    
    for i, c in enumerate(classes):
        # Обычные объекты
        mask = (y == c) & ~prototype_mask
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colors[i], label=f'Класс {c} (обычные)',
            alpha=0.3, s=30, edgecolors='none'
        )
        
        # Эталоны
        mask = (y == c) & prototype_mask
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colors[i], label=f'Класс {c} (эталоны)',
            alpha=1.0, s=150, marker='*', edgecolors='black', linewidth=1.5
        )
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} дисперсии)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} дисперсии)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Добавляем информацию о количестве эталонов
    compression = len(prototype_indices) / len(X)
    info_text = f'Эталонов: {len(prototype_indices)} из {len(X)} ({compression:.1%})'
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
        print(f"Визуализация эталонов сохранена в: {save_path}")
    
    plt.close()


def compare_prototype_methods(X, y, k=5):
    """
    Сравнивает различные методы отбора эталонов.
    
    Args:
        X: матрица признаков
        y: метки классов
        k: количество соседей для KNN
    
    Returns:
        dict: результаты сравнения
    """
    print("\n" + "------ Сравнение методов отбора эталонов ------")
    
    # CNN алгоритм
    print("\n1. Condensed Nearest Neighbor (CNN)")
    cnn_result = condensed_nearest_neighbor(X, y, k=k, verbose=True)
    
    # STOLP алгоритм
    print("\n2. STOLP алгоритм")
    stolp_result = stolp_algorithm(X, y, k=k, threshold=0.0, verbose=True)
    
    print("\n" + "------ Сводка ------")
    print(f"{'Метод':<20} {'Эталонов':<12} {'Сжатие':<12}")
    print(f"{'─' * 60}")
    print(f"{'CNN':<20} {cnn_result['n_prototypes']:<12} {cnn_result['compression_ratio']:<12.2%}")
    print(f"{'STOLP':<20} {stolp_result['n_prototypes']:<12} {stolp_result['compression_ratio']:<12.2%}")
    print(f"{'─' * 60}")
    
    return {
        'cnn': cnn_result,
        'stolp': stolp_result
    }


def compute_distance_matrix(X):
    """
    Вычисляет матрицу попарных евклидовых расстояний.
    
    Args:
        X: матрица признаков (n_samples, n_features)
    
    Returns:
        distances: матрица расстояний (n_samples, n_samples)
    """
    n_samples = X.shape[0]
    X_norm_sq = np.sum(X ** 2, axis=1, keepdims=True)
    distances_sq = X_norm_sq + X_norm_sq.T - 2 * np.dot(X, X.T)
    distances_sq = np.maximum(distances_sq, 0)
    return np.sqrt(distances_sq)


def compute_compactness_profile(X, y, subset_indices, max_k=3):
    """
    Вычисляет профиль компактности Π(m) для подмножества объектов.
    
    Профиль компактности Π(m) - доля объектов из всей выборки, у которых
    m-й ближайший сосед в подмножестве имеет другой класс.
    
    Args:
        X: полная матрица признаков
        y: полные метки классов
        subset_indices: индексы объектов подмножества
        max_k: максимальное m для профиля
    
    Returns:
        profile: массив значений Π(m) для m=0..max_k-1
    """
    n_samples = len(y)
    X_subset = X[subset_indices]
    y_subset = y[subset_indices]
    
    if len(subset_indices) == 0:
        return np.ones(max_k)
    
    # Вычисляем расстояния от всех объектов до объектов подмножества
    # distances[i, j] = расстояние от i-го объекта до j-го объекта подмножества
    distances = np.zeros((n_samples, len(subset_indices)))
    for i in range(n_samples):
        for j, idx in enumerate(subset_indices):
            distances[i, j] = np.linalg.norm(X[i] - X[idx])
    
    profile = np.zeros(max_k)
    
    for m in range(min(max_k, len(subset_indices))):
        error_count = 0
        
        for i in range(n_samples):
            # Находим индексы m+1 ближайших соседей в подмножестве
            neighbor_indices_in_subset = np.argsort(distances[i])[:m+1]
            # m-й ближайший сосед (0-indexed)
            m_neighbor_in_subset = neighbor_indices_in_subset[m]
            m_neighbor_global = subset_indices[m_neighbor_in_subset]
            
            if y[i] != y[m_neighbor_global]:
                error_count += 1
        
        profile[m] = error_count / n_samples
    
    return profile


def compute_ccv_score(X, y, subset_indices, k=3):
    """
    Вычисляет CCV (Complete Cross-Validation) score через профиль компактности.
    
    CCV = Σ_{m=1}^k Π(m) * C(L-1, l-1-m) / C(L, l)
    
    где:
    - Π(m) - профиль компактности
    - C(n, k) - биномиальный коэффициент
    - L - размер полной выборки
    - l - размер подмножества
    
    Args:
        X: матрица признаков
        y: метки классов
        subset_indices: индексы подмножества
        k: максимальное m для профиля
    
    Returns:
        ccv: значение CCV score
    """
    L = len(y)
    l = len(subset_indices)
    
    if l == 0 or l > L:
        return 1.0
    
    profile = compute_compactness_profile(X, y, subset_indices, max_k=k)
    
    ccv = 0.0
    for m in range(min(k, l)):
        # Биномиальный коэффициент
        if l - 1 - m >= 0 and L - 1 >= l - 1 - m:
            weight = comb(L - 1, l - 1 - m, exact=True) / comb(L, l, exact=True)
            ccv += profile[m] * weight
    
    return ccv


def ccv_prototype_selection(X, y, k=3, max_candidates=20, max_iterations=100, verbose=True):
    """
    Алгоритм отбора эталонов через минимизацию CCV (Complete Cross-Validation).
    
    Использует жадную стратегию добавления эталонов с эвристикой для ускорения.
    
    Args:
        X: матрица признаков
        y: метки классов
        k: параметр для CCV (обычно 3)
        max_candidates: количество кандидатов для тестирования на каждой итерации
        max_iterations: максимальное количество итераций
        verbose: выводить прогресс
    
    Returns:
        dict: результаты отбора эталонов
    """
    n_samples = len(X)
    classes = np.unique(y)
    
    if verbose:
        print(f"\nАлгоритм отбора эталонов CCV")
        print(f"Исходное количество объектов: {n_samples}")
        print(f"Количество классов: {len(classes)}")
        print(f"Параметр k: {k}")
    
    # Начальная инициализация
    # Для каждого класса выбираем объект, ближайший к центроиду
    prototype_indices = []
    
    for c in classes:
        class_mask = y == c
        class_X = X[class_mask]
        centroid = np.mean(class_X, axis=0)
        
        distances_to_centroid = np.linalg.norm(class_X - centroid, axis=1)
        closest_idx_in_class = np.argmin(distances_to_centroid)
        closest_idx_global = np.where(class_mask)[0][closest_idx_in_class]
        
        prototype_indices.append(int(closest_idx_global))
    
    ccv_history = []
    remaining = set(range(n_samples)) - set(prototype_indices)
    
    if verbose:
        print(f"Начальная инициализация: {len(prototype_indices)} эталонов (по 1 из каждого класса)")
    
    # Итеративное добавление эталонов
    for iteration in range(max_iterations):
        best_ccv = float('inf')
        best_candidate = None
        
        if len(remaining) == 0:
            break
        
        candidate_scores = []
        
        for candidate in remaining:
            distances_to_prototypes = [
                np.linalg.norm(X[candidate] - X[p])
                for p in prototype_indices
            ]
            importance = min(distances_to_prototypes) if distances_to_prototypes else 0
            candidate_scores.append((candidate, importance))
        
        # Сортируем по важности (от большего к меньшему)
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [c[0] for c in candidate_scores[:min(max_candidates, len(candidate_scores))]]
        
        for candidate in top_candidates:
            test_indices = prototype_indices + [candidate]
            ccv = compute_ccv_score(X, y, test_indices, k)
            
            if ccv < best_ccv:
                best_ccv = ccv
                best_candidate = candidate
        
        if best_candidate is None:
            if verbose:
                print("Не найдено улучшений, останавливаемся")
            break
        
        # Критерий остановки: CCV перестал уменьшаться
        if len(ccv_history) > 0 and best_ccv >= ccv_history[-1]:
            if verbose:
                print(f"Итерация {iteration + 1}: CCV перестал уменьшаться ({best_ccv:.4f} >= {ccv_history[-1]:.4f})")
            break
        
        prototype_indices.append(best_candidate)
        remaining.remove(best_candidate)
        ccv_history.append(best_ccv)
        
        if verbose and (iteration % 5 == 0 or iteration < 5):
            print(f"Итерация {iteration + 1}: {len(prototype_indices)} эталонов, CCV = {best_ccv:.4f}")
    
    X_prototypes = X[prototype_indices]
    y_prototypes = y[prototype_indices]
    compression_ratio = len(prototype_indices) / n_samples
    
    if verbose:
        print(f"\nФинальное количество эталонов: {len(prototype_indices)}")
        print(f"Степень сжатия: {compression_ratio:.2%}")
        print("Распределение эталонов по классам:")
        for c in classes:
            count = np.sum(y_prototypes == c)
            print(f"  Класс {c}: {count} эталонов")
    
    return {
        'indices': prototype_indices,
        'X_prototypes': X_prototypes,
        'y_prototypes': y_prototypes,
        'compression_ratio': compression_ratio,
        'n_prototypes': len(prototype_indices),
        'ccv_history': ccv_history
    }


def plot_ccv_history(ccv_history, save_path=None):
    """
    Строит график изменения CCV score в процессе отбора эталонов.
    
    Args:
        ccv_history: история значений CCV
        save_path: путь для сохранения графика
    """
    plt.figure(figsize=(10, 6))
    
    iterations = range(1, len(ccv_history) + 1)
    plt.plot(iterations, ccv_history, 'b-o', linewidth=2, markersize=5, alpha=0.7)
    
    plt.xlabel('Количество эталонов', fontsize=12)
    plt.ylabel('CCV Score', fontsize=12)
    plt.title('Процесс отбора эталонов методом CCV', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if len(ccv_history) > 0:
        min_idx = np.argmin(ccv_history)
        min_ccv = ccv_history[min_idx]
        plt.plot(min_idx + 1, min_ccv, 'r*', markersize=15, label=f'Минимум: {min_ccv:.4f}')
        plt.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График CCV history сохранен в: {save_path}")
    
    plt.close()
