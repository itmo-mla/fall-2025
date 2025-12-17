import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


class MyKNN:
    def __init__(self, k=5, kernel='gaussian'):
        self.k = k
        self.kernel = kernel
        self.X_train = None
        self.y_train = None
        self.scaler = StandardScaler()

    def _gaussian_kernel(self, r):
        """
        Гауссово ядро K(r) = exp(-2r²)
        """
        return np.exp(-2 * r**2)

    def _epanechnikov_kernel(self, r):
        """
        Квадратичное ядро (Епанечникова) K(r) = (1 - r²)[|r| ≤ 1]
        """
        return np.where(np.abs(r) <= 1, 1 - r**2, 0)

    def _rectangular_kernel(self, r):
        """
        Прямоугольное ядро K(r) = [|r| ≤ 1]
        """
        return np.where(np.abs(r) <= 1, 1, 0)

    def _compute_distances(self, X):
        """
        Вычисление евклидова расстояния между объектами
        ρ(x, x_i) = (∑|x^j - x_i^j|²)^(1/2)
        """
        distances = np.sqrt(np.sum((self.X_train - X[:, np.newaxis])**2, axis=2))
        return distances

    def _get_kernel_func(self):
        kernels = {
            'gaussian': self._gaussian_kernel,
            'epanechnikov': self._epanechnikov_kernel,
            'rectangular': self._rectangular_kernel
        }
        return kernels.get(self.kernel, self._gaussian_kernel)

    def fit(self, X, y):
        self.X_train = self.scaler.fit_transform(X)
        self.y_train = y.values if hasattr(y, 'values') else y
        return self

    def predict(self, X):
        """
        Предсказание для новых объектов

        Основная формула классификации
        a(x; X^ℓ, k, K) = arg max_{y∈Y} ∑_{i=1}^ℓ [y_i = y] K( ρ(x, x_i) / ρ(x, x^{(k+1)}) )

        где:
        K(r) - функция ядра
        ρ(x, x_i) - расстояние до iго объекта
        ρ(x, x^{(k+1)}) - расстояние до (k+1)го соседа (переменная ширина окна)
        """
        X_normalized = self.scaler.transform(X)
        kernel_func = self._get_kernel_func()
        predictions = []
        for x in X_normalized:
            # ρ(x, x_i) = (∑|x^j - x_i^j|²)^(1/2) вычисляем расстояния до всех объектов обучения
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

            # находим k+1 ближайших соседей (для метода парзеновского окна переменной ширины)
            k_neighbors_id = np.argpartition(distances, self.k)[:self.k + 1]
            k_neighbors_distances = distances[k_neighbors_id]

            # ρ(x, x^{(1)}) ≤ ρ(x, x^{(2)}) ≤ ... ≤ ρ(x, x^{(ℓ)}) отранжируем объекты
            sorted_indices = np.argsort(k_neighbors_distances)
            sorted_distances = k_neighbors_distances[sorted_indices]
            sorted_labels = self.y_train[k_neighbors_id[sorted_indices]]

            # h = ρ(x, x^{(k+1)}) переменная ширина окна - расстояние до (k+1)го соседа
            h = sorted_distances[self.k]

            # w(i, x) = K( ρ(x, x^{(i)}) / h) вычисляем веса для k ближайших соседей (исключая (k+1)го)
            weights = []
            for i in range(self.k):
                r = sorted_distances[i] / h if h > 0 else 0
                weight = kernel_func(r)
                weights.append(weight)

            # взвешенное голосование
            class_weights = {}
            for i in range(self.k):
                label = sorted_labels[i]
                weight = weights[i]
                class_weights[label] = class_weights.get(label, 0) + weight
            # a(x) = arg max_{y∈Y} Γ_y(x) выбираем класс с максимальным весом
            predicted_class = max(class_weights.items(), key=lambda x: x[1])[0]
            predictions.append(predicted_class)

        return np.array(predictions)

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


wine_data = pd.read_csv('winequality-red.csv')

print("Размерность данных:", wine_data.shape)
print("\nПервые 5 строк:")
print(wine_data.head())
print("\nИнформация о данных:")
print(wine_data.info())
print("\nСтатистика:")
print(wine_data.describe())


def to_three_classes(y):
    y_new = y.copy()
    y_new[y <= 4] = 0  # низкое
    y_new[(y == 5) | (y == 6)] = 1  # среднее
    y_new[y >= 7] = 2  # высокое
    return y_new


wine_data['quality_3class'] = to_three_classes(wine_data['quality'])

print("Распределение по 3 классам:")
print(wine_data['quality_3class'].value_counts().sort_index())
plt.figure(figsize=(10, 4))
wine_data['quality_3class'].value_counts().sort_index().plot(kind='bar')
plt.title('3 класса качества вин')
plt.xlabel('Класс (0=низкое, 1=среднее, 2=высокое)')
plt.ylabel('Количество')
plt.xticks(rotation=0)
plt.savefig(os.path.join(PLOTS_DIR, "1_wine_3classes.png"), dpi=150, bbox_inches='tight')

X = wine_data.drop(['quality', 'quality_3class'], axis=1)
y = wine_data['quality_3class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nДанные для обучения:")
print(f"X: {X_train.shape}, y: {y_train.shape}")
print(f"Классы в y_train: {y_train.value_counts().sort_index().to_dict()}")


def loo_optimization(X_train, y_train, k_values, kernel='gaussian'):
    """
    LOO(k, X^ℓ) = ∑[a(x_i; X^ℓ \ {x_i}, k) ≠ y_i]
    """
    n_samples = len(X_train)
    loo_scores = []

    print("\n=== Оптимизация параметра по LOO ===")
    X_train_np = X_train.values
    y_train_np = y_train.values
    for k in k_values:
        print(f"Обрабатывается k={k}...", end=' ')
        correct_predictions = 0
        for i in range(n_samples):
            # создаем обучающую выборку без iго объекта
            X_train_loo = np.delete(X_train_np, i, axis=0)
            y_train_loo = np.delete(y_train_np, i, axis=0)

            X_test_loo = X_train_np[i:i + 1]  # берем iй объект как тестовый
            true_label = y_train_np[i]

            # обучаем модель и делаем предсказание
            knn = MyKNN(k=k, kernel=kernel)
            knn.fit(X_train_loo, y_train_loo)
            y_pred = knn.predict(X_test_loo)

            # проверяем правильность предсказания
            if y_pred[0] == true_label:
                correct_predictions += 1

        # вычисляем LOO точность
        loo_accuracy = correct_predictions / n_samples
        loo_scores.append(loo_accuracy)
        print(f"LOO точность = {loo_accuracy:.4f}")

    return loo_scores


k_values = range(1, 21)
loo_scores = loo_optimization(X_train, y_train, k_values)

plt.figure(figsize=(12, 6))
plt.plot(k_values, loo_scores, 'bo-', linewidth=2, markersize=6)
plt.xlabel('Количество соседей (k)')
plt.ylabel('LOO точность')
plt.title('Зависимость точности от параметра k. LOO оптимизация.')
plt.grid(True, alpha=0.3)
best_k_index = np.argmax(loo_scores)
best_k = k_values[best_k_index]
best_score = loo_scores[best_k_index]
plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7,label=f'Лучшее k={best_k} ({best_score:.4f})')
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "2_loo_optimization.png"), dpi=150, bbox_inches='tight')

print(f"\nЛучший параметр: k = {best_k}")
print(f"LOO точность при лучшем k = {best_score:.4f}")


def analyze_parameters(X_train, y_train, X_test, y_test, best_k):
    kernels = ['gaussian', 'epanechnikov', 'rectangular']
    kernel_results = []

    print("\n=== Анализ влияния параметров ===")
    for kernel in kernels:
        knn = MyKNN(k=best_k, kernel=kernel)
        knn.fit(X_train, y_train)
        train_score = knn.score(X_train, y_train)
        test_score = knn.score(X_test, y_test)
        kernel_results.append({'kernel': kernel, 'train_score': train_score, 'test_score': test_score})
        print(f"Ядро {kernel:12} | Обучающая точность: {train_score:.4f} | Тестовая точность: {test_score:.4f}")

    print("\n=== Эмпирический риск для разных K ===")
    train_scores = []
    test_scores = []
    for k in range(1, 21):
        knn = MyKNN(k=k, kernel='gaussian')
        knn.fit(X_train, y_train)
        train_score = knn.score(X_train, y_train)
        test_score = knn.score(X_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        if k % 5 == 0:
            print(f"k={k:2} | Обучающая: {train_score:.4f} | Тестовая: {test_score:.4f}")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 2)
    k_range = range(1, 21)
    plt.plot(k_range, train_scores, 'bo-', label='Обучающая выборка')
    plt.plot(k_range, test_scores, 'ro-', label='Тестовая выборка')
    plt.xlabel('Количество соседей k')
    plt.ylabel('Точность')
    plt.title('Эмпирический риск')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_range)
    plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Лучшее k={best_k}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "3_parameter_analysis.png"), dpi=150, bbox_inches='tight')
    return kernel_results, train_scores, test_scores


kernel_analysis, train_scores, test_scores = analyze_parameters(X_train, y_train, X_test, y_test, best_k)


def compare_with_sklearn(X_train, y_train, X_test, y_test, best_k):
    print("\n=== Сравнение с эталонной реализацией ===")
    my_knn = MyKNN(k=best_k, kernel='gaussian')
    my_knn.fit(X_train, y_train)
    our_train_score = my_knn.score(X_train, y_train)
    our_test_score = my_knn.score(X_test, y_test)

    sklearn_knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
    sklearn_knn.fit(X_train, y_train)
    sklearn_train_score = sklearn_knn.score(X_train, y_train)
    sklearn_test_score = sklearn_knn.score(X_test, y_test)

    print(f"{'Метрика':<25} | {'MyKNN':<15} | {'Sklearn':<15}")
    print("-" * 60)
    print(f"{'Обучающая точность':<25} | {our_train_score:<15.4f} | {sklearn_train_score:<15.4f}")
    print(f"{'Тестовая точность':<25} | {our_test_score:<15.4f} | {sklearn_test_score:<15.4f}")

    my_predictions = my_knn.predict(X_test)
    sklearn_predictions = sklearn_knn.predict(X_test)

    print("\nРаспределение предсказанных классов (моя модель):")
    our_pred_counts = pd.Series(my_predictions).value_counts().sort_index()
    for cls, count in our_pred_counts.items():
        class_name = ['Низкое', 'Среднее', 'Высокое'][cls]
        print(f"{class_name}: {count} объектов")

    print("\nРаспределение истинных классов:")
    true_counts = y_test.value_counts().sort_index()
    for cls, count in true_counts.items():
        class_name = ['Низкое', 'Среднее', 'Высокое'][cls]
        print(f"{class_name}: {count} объектов")

    plt.figure(figsize=(15, 5))

    # график 1 сравнение точности
    plt.subplot(1, 3, 1)
    models = ['MyKNN', 'Sklearn KNN']
    train_scores = [our_train_score, sklearn_train_score]
    test_scores = [our_test_score, sklearn_test_score]

    x_pos = np.arange(len(models))
    width = 0.35

    plt.bar(x_pos - width / 2, train_scores, width, label='Обучающая', alpha=0.7, color='blue')
    plt.bar(x_pos + width / 2, test_scores, width, label='Тестовая', alpha=0.7, color='red')

    plt.xlabel('Модели')
    plt.ylabel('Точность')
    plt.title('Сравнение точности моделей')
    plt.xticks(x_pos, models)
    plt.legend()
    plt.ylim(0.5, 1.0)
    plt.grid(True, alpha=0.3, axis='y')

    # График 2 матрица ошибок MyKNN
    plt.subplot(1, 3, 2)
    cm_my = confusion_matrix(y_test, my_predictions)
    sns.heatmap(cm_my, annot=True, fmt='d', cmap='Blues', xticklabels=['Низкое', 'Среднее', 'Высокое'], yticklabels=['Низкое', 'Среднее', 'Высокое'])
    plt.title('Матрица ошибок\nMyKNN')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')

    # График 3 матрица ошибок sklearn
    plt.subplot(1, 3, 3)
    cm_sklearn = confusion_matrix(y_test, sklearn_predictions)
    sns.heatmap(cm_sklearn, annot=True, fmt='d', cmap='Greens', xticklabels=['Низкое', 'Среднее', 'Высокое'], yticklabels=['Низкое', 'Среднее', 'Высокое'])
    plt.title('Матрица ошибок\nsklearn')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "4_sklearn_comparison.png"), dpi=150, bbox_inches='tight')

    return my_knn, sklearn_knn


my_model, sklearn_model = compare_with_sklearn(X_train, y_train, X_test, y_test, best_k)


class PrototypeSelector:
    """
    Алгоритм отбора эталонов на основе жадной стратегии
    """
    def __init__(self, k=5, kernel='gaussian'):
        self.k = k
        self.kernel = kernel
        self.prototypes = None
        self.prototype_labels = None
        self.scaler = StandardScaler()

    def _compute_loo_score(self, X, y, prototypes_indices):
        """
        Вычисление LOO точности для текущего набора эталонов
        """
        n_samples = len(X)
        correct_predictions = 0

        # используем только выбранные эталоны
        X_prototypes = X[prototypes_indices]
        y_prototypes = y[prototypes_indices]

        # создаем временный KNN только с эталонами
        temp_knn = MyKNN(k=self.k, kernel=self.kernel)
        temp_knn.X_train = X_prototypes
        temp_knn.y_train = y_prototypes

        for i in range(n_samples):
            if i in prototypes_indices:
                correct_predictions += 1  # эталон всегда правильно классифицирует сам себя
                continue

            # предсказываем для i-го объекта
            x_test = X[i:i + 1]
            true_label = y[i]
            distances = np.sqrt(np.sum((X_prototypes - x_test) ** 2, axis=1))
            k_neighbors = min(self.k, len(prototypes_indices))

            if k_neighbors == 0:
                # если нет эталонов, предсказываем наиболее частый класс
                predicted_class = np.bincount(y_prototypes).argmax() if len(y_prototypes) > 0 else 1
            else:
                k_neighbors_id = np.argpartition(distances, k_neighbors - 1)[:k_neighbors]  # k-1 для границ
                k_neighbors_distances = distances[k_neighbors_id]

                # сортируем соседей
                sorted_indices = np.argsort(k_neighbors_distances)
                sorted_distances = k_neighbors_distances[sorted_indices]
                sorted_labels = y_prototypes[k_neighbors_id[sorted_indices]]

                # переменная ширина окна
                if len(prototypes_indices) > self.k:
                    h = sorted_distances[self.k - 1]
                else:
                    h = sorted_distances[-1] if len(sorted_distances) > 0 else 1.0

                # взвешенное голосование
                kernel_func = temp_knn._get_kernel_func()
                class_weights = {}

                for j in range(len(sorted_distances)):
                    r = sorted_distances[j] / h if h > 0 else 0
                    weight = kernel_func(r)
                    label = sorted_labels[j]
                    class_weights[label] = class_weights.get(label, 0) + weight

                if class_weights:
                    predicted_class = max(class_weights.items(), key=lambda x: x[1])[0]
                else:
                    predicted_class = np.bincount(y_prototypes).argmax() if len(y_prototypes) > 0 else 1

            if predicted_class == true_label:
                correct_predictions += 1

        return correct_predictions / n_samples if n_samples > 0 else 0

    def greedy_prototype_selection(self, X, y, max_prototypes=None):
        """
        Жадный отбор эталонов
        """
        X_normalized = self.scaler.fit_transform(X)
        y_np = y.values if hasattr(y, 'values') else y

        n_samples = len(X_normalized)
        if max_prototypes is None:
            max_prototypes = n_samples // 4

        # начинаем с одного эталона от каждого класса
        prototypes_indices = []
        unique_classes = np.unique(y_np)
        for cls in unique_classes:
            class_indices = np.where(y_np == cls)[0]
            if len(class_indices) > 0:
                prototypes_indices.append(class_indices[0])

        print("Начальный набор эталонов:")
        for idx in prototypes_indices:
            print(f"Объект {idx}, класс {y_np[idx]}")
        best_score = self._compute_loo_score(X_normalized, y_np, prototypes_indices)
        print(f"Начальная LOO точность: {best_score:.4f}")

        # жадное добавление эталонов
        scores_history = [best_score]
        prototypes_history = [prototypes_indices.copy()]
        itr = 0
        while len(prototypes_indices) < max_prototypes and len(prototypes_indices) < n_samples:
            itr += 1
            print(f"\nИтерация {itr}: текущее количество эталонов = {len(prototypes_indices)}")
            best_candidate = None
            best_candidate_score = best_score

            # перебираем все возможные кандидаты для добавления
            candidates = [i for i in range(n_samples) if i not in prototypes_indices]
            if not candidates:
                break
            for candidate in candidates:
                candidate_prototypes = prototypes_indices + [candidate]
                score = self._compute_loo_score(X_normalized, y_np, candidate_prototypes)
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = candidate

            # добавляем лучшего кандидата если он улучшает качество
            if best_candidate is not None and best_candidate_score > best_score:
                prototypes_indices.append(best_candidate)
                best_score = best_candidate_score
                scores_history.append(best_score)
                prototypes_history.append(prototypes_indices.copy())
                print(f"Добавлен эталон {best_candidate}, класс {y_np[best_candidate]}")
                print(f"LOO точность: {best_score:.4f}")
            else:
                print("Не удалось улучшить точность, завершаем отбор")
                break

        self.prototypes = X_normalized[prototypes_indices]
        self.prototype_labels = y_np[prototypes_indices]
        return prototypes_indices, scores_history, prototypes_history

    def fit(self, X, y):
        """Обучение с отобранными эталонами. Возвращает историю для визуализации."""
        prototypes_indices, scores_history, prototypes_history = self.greedy_prototype_selection(X, y)
        self.scores_history = scores_history
        self.prototypes_history = prototypes_history
        return self

    def predict(self, X):
        """Предсказание с использованием только эталонов"""
        X_normalized = self.scaler.transform(X)
        predictions = []

        for x in X_normalized:
            distances = np.sqrt(np.sum((self.prototypes - x) ** 2, axis=1))

            # находим k ближайших эталонов
            k_neighbors = min(self.k, len(self.prototypes))

            if k_neighbors == 0:
                predictions.append(1)
                continue

            k_neighbors_id = np.argpartition(distances, k_neighbors - 1)[:k_neighbors]  # k-1 для границ
            k_neighbors_distances = distances[k_neighbors_id]

            # сортируем соседей
            sorted_indices = np.argsort(k_neighbors_distances)
            sorted_distances = k_neighbors_distances[sorted_indices]
            sorted_labels = self.prototype_labels[k_neighbors_id[sorted_indices]]

            # переменная ширина окна
            if len(self.prototypes) > self.k:
                h = sorted_distances[self.k - 1]  # k-1 для границ
            else:
                h = sorted_distances[-1] if len(sorted_distances) > 0 else 1.0

            # взвешенное голосование
            knn = MyKNN(k=self.k, kernel=self.kernel)
            kernel_func = knn._get_kernel_func()
            class_weights = {}

            for j in range(len(sorted_distances)):
                r = sorted_distances[j] / h if h > 0 else 0
                weight = kernel_func(r)
                label = sorted_labels[j]
                class_weights[label] = class_weights.get(label, 0) + weight

            if class_weights:
                predicted_class = max(class_weights.items(), key=lambda x: x[1])[0]
            else:
                predicted_class = 1

            predictions.append(predicted_class)

        return np.array(predictions)

    def score(self, X, y):
        """Оценка точности модели с эталонами"""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


print("\n" + "=" * 70)
print("ОТБОР ЭТАЛОНОВ")
print("=" * 70)

prototype_selector = PrototypeSelector(k=min(5, best_k), kernel='gaussian')
prototype_selector.fit(X_train, y_train)

print("\nСоздание визуализации отобранных эталонов...")

pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(prototype_selector.scaler.transform(X_train))

prototypes_indices = []
unique_classes = np.unique(y_train)
for cls in unique_classes:
    class_indices = np.where(y_train == cls)[0]
    if len(class_indices) > 0:
        prototypes_indices.append(class_indices[0])

selector_for_viz = PrototypeSelector(k=min(5, best_k), kernel='gaussian')
_, scores, prototypes_hist = selector_for_viz.greedy_prototype_selection(X_train, y_train)
final_prototype_indices = prototypes_hist[-1]

X_full_pca = pca.transform(prototype_selector.scaler.transform(X_train))
X_prototypes_pca = X_full_pca[final_prototype_indices]

plt.figure(figsize=(12, 8))

scatter_all = plt.scatter(X_full_pca[:, 0], X_full_pca[:, 1], c=y_train, cmap='viridis', alpha=0.3, s=20, label='Все объекты')
scatter_prototypes = plt.scatter(X_prototypes_pca[:, 0], X_prototypes_pca[:, 1], c=y_train.iloc[final_prototype_indices], cmap='viridis',
                                 edgecolors='red', linewidth=2, s=200, label='Эталоны')

plt.title(f'Отобранные эталоны (всего: {len(final_prototype_indices)}) в пространстве PCA')
plt.xlabel(f'Первая главная компонента (объясняет {pca.explained_variance_ratio_[0]:.1%} дисперсии)')
plt.ylabel(f'Вторая главная компонента (объясняет {pca.explained_variance_ratio_[1]:.1%} дисперсии)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.colorbar(scatter_all, label='Класс качества')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "6_prototypes_pca.png"), dpi=150, bbox_inches='tight')

print(f"\nОтобрано эталонов: {len(prototype_selector.prototypes)}")
print(f"Исходное количество объектов: {len(X_train)}")
print(f"Сжатие: {len(prototype_selector.prototypes) / len(X_train) * 100:.1f}%")

prototype_train_score = prototype_selector.score(X_train, y_train)
prototype_test_score = prototype_selector.score(X_test, y_test)

print(f"\nКачество модели с эталонами:")
print(f"Обучающая точность: {prototype_train_score:.4f}")
print(f"Тестовая точность: {prototype_test_score:.4f}")

plt.figure(figsize=(8, 5))
num_prototypes = [len(p) for p in prototype_selector.prototypes_history]
plt.plot(num_prototypes, prototype_selector.scores_history, 'go-', linewidth=2, markersize=6)
plt.xlabel('Число эталонов')
plt.ylabel('LOO-точность')
plt.title('Изменение LOO-точности при отборе эталонов')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "5_prototype_selection_analysis.png"), dpi=150, bbox_inches='tight')
plt.close()


print("\n" + "=" * 80)
print("СРАВНЕНИЕ KNN С И БЕЗ ОТБОРА ЭТАЛОНОВ")
print("=" * 80)

standard_knn = MyKNN(k=best_k, kernel='gaussian')
standard_knn.fit(X_train, y_train)

y_pred_standard = standard_knn.predict(X_test)
y_pred_prototype = prototype_selector.predict(X_test)

# метрики для обычного KNN
standard_test_accuracy = standard_knn.score(X_test, y_test)
standard_test_balanced = balanced_accuracy_score(y_test, y_pred_standard)

# метрики для KNN с эталонами
prototype_test_accuracy = prototype_selector.score(X_test, y_test)
prototype_test_balanced = balanced_accuracy_score(y_test, y_pred_prototype)

print(f"{'Метрика':<30} | {'Обычный KNN':<12} | {'KNN с эталонами':<15} | {'Изменение'}")
print("-" * 85)
print(f"{'Точность':<30} | {standard_test_accuracy:<12.4f} | {prototype_test_accuracy:<15.4f} | {prototype_test_accuracy - standard_test_accuracy:+.4f}")
print(f"{'Сбалансированная точность':<30} | {standard_test_balanced:<12.4f} | {prototype_test_balanced:<15.4f} | {prototype_test_balanced - standard_test_balanced:+.4f}")
print(f"{'Размер модели':<30} | {len(X_train):<12} | {len(prototype_selector.prototypes):<15} | {len(prototype_selector.prototypes) - len(X_train):+d}")
print(f"{'Сжатие данных':<30} | {'100%':<12} | {len(prototype_selector.prototypes)/len(X_train)*100:<15.1f}% | -{(1 - len(prototype_selector.prototypes)/len(X_train))*100:.1f}%")


print(f"\nСравнение производительности")
start_time = time.time()
standard_knn.predict(X_test)
standard_time = time.time() - start_time

start_time = time.time()
prototype_selector.predict(X_test)
prototype_time = time.time() - start_time

print(f"Время предсказания (обычный KNN): {standard_time:.4f} сек")
print(f"Время предсказания (KNN с эталонами): {prototype_time:.4f} сек")
print(f"Ускорение: {standard_time/prototype_time:.2f}x")
