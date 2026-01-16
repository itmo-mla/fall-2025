import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from sklearn.decomposition import PCA
from collections import Counter
import pandas as pd
import time

data_raw = load_breast_cancer()
df = pd.DataFrame(data_raw.data, columns=data_raw.feature_names)
df['diagnosis'] = data_raw.target_names[data_raw.target]

# Переведем таргетные метки в числовой формат -1 и 1
df['diagnosis'] = df['diagnosis'].map({'malignant': 1, 'benign': -1})
# Отберем признаки, уменьшив размерность
worst_cols = [col for col in df.columns if 'worst' in col]
df_worst = df[worst_cols].drop(columns=['area worst', 'perimeter worst'], errors='ignore')
# Переведем в нумпай
X = df_worst.values
y = df['diagnosis'].values
# Нормализация
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Размер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)

# с метками -1 и 1 формула KNN превращается в проверку знака суммы
# если сумма положительна - класс 1, иначе -1
class ParzenKNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def gaussian_kernel(self, distance, h):
        """
        Гауссово ядро: K(r) = exp(- (r / h)^2)
        Превращает расстояние в вес
        distance = 0, вес = 1 (максимальный)
        distance = h, вес = exp(-1) ~ 0.37
        distance >> h, вес стремится 0

        distance - расстояние от точки до обучающего примера
        h - ширина ядра
        """

        # добавляем eps, чтобы избежать деления на ноль
        eps = 1e-10
        h = h + eps
        return np.exp(- (distance / h) ** 2)
    
    def predict(self, X_test):
        predictions = []

        for x in X_test:
            # Считаем Евклидово расстояние от текущей точки до всех точек
            # sqrt(sum((x_train - x_test)^2))
            # axis = 1 - суммируем квадраты разностей по столбцам (по признакам)
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

            # получаем индексы точек, отсортированных от ближайших к дальним
            sorted_indices = np.argsort(distances)

            # первые k индексов - голосующие соседи
            k_indices = sorted_indices[:self.k]

            # возьмем расстояния и метки соседей
            k_distances = distances[k_indices]
            k_labels = self.y_train[k_indices]

            # Адаптивная ширина окна (h)
            # Берем расстояние до k+1 (дальнего) соседа
            # в плотных скоплениях окно будет узкое, в разреженных шире

            h_index = self.k 
            
            if h_index < len(distances):
                # Если k+1 сосед существует
                h = distances[sorted_indices[h_index]]
            else:
                # Если выборка слишком маленькая (например, len(X_omega) <= k), 
                # берем самого дальнего соседа (индекс -1)
                h = distances[sorted_indices[-1]]

            
            # h = distances[sorted_indices[self.k]]

            # расстояние в веса - чем ближе сосед, тем больший вес он имеет
            weights = self.gaussian_kernel(k_distances, h)

            # взвешенное голосование
            # умножаем вес соседа на его метку, потом суммируем
            weighted_votes = weights * k_labels
            total_score = np.sum(weighted_votes)

            # предсказываем класс по знаку
            pred = np.sign(total_score)
            
            if pred == 0:
                pred = -1  # если сумма равна нулю, отдаем предпочтение классу 1

            predictions.append(pred)

        return np.array(predictions)
    

model = ParzenKNN(k=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# print("Истинные метки:", y_test)
# print("Предсказания:", y_pred)


def run_loo_score(X, y, k_max=20):
    """
    Функция для подбора k методом скользящего контроля Leave-One-Out
    """

    n_samples = X.shape[0]
    errors_list = []

    # перебираем k от 1 до k_max
    k_range = range(1, k_max+1)

    for k in k_range:
        mistakes = 0

        # модель с текущим k
        model = ParzenKNN(k=k)

        for i in range(n_samples):
            # исключаем объект i из обучения
            mask = np.ones(n_samples, dtype=bool)
            mask[i] = False

            X_train_loo = X[mask]
            y_train_loo = y[mask]

            # объект для теста - i
            X_val = X[i].reshape(1, -1)
            y_val = y[i]

            model.fit(X_train_loo, y_train_loo)

            prediction = model.predict(X_val)[0]

            if prediction != y_val:
                mistakes += 1
        
        # эмпирический риск (доля ошибок) для текущего k
        risk = mistakes / n_samples
        errors_list.append(risk)
        print(f"k={k}, ошибки: {mistakes}, риск: {risk:.4f}")

    return k_range, errors_list

# подбираем k от 1 до 20
k_vals, risk_vals = run_loo_score(X_train, y_train, k_max=20)

min_risk_idx = np.argmin(risk_vals)
best_k = k_vals[min_risk_idx]
print(f"Лучшее k по LOO: {best_k} с риском {risk_vals[min_risk_idx]:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(k_vals, risk_vals, marker='o', linestyle='-', color='b', label='LOO Error')

# Отметим точку минимума
plt.scatter(best_k, risk_vals[min_risk_idx], color='red', s=100, zorder=5, label=f'Best k={best_k}')

plt.title('График эмпирического риска (LOO) от параметра k')
plt.xlabel('Количество соседей k')
plt.ylabel('Доля ошибок (Empirical Risk)')
plt.xticks(k_vals)
plt.grid(True)
plt.legend()
# plt.show()
# plt.savefig('loo_risk_plot.png')

# сравнение с KNeighborsClassifier из sklearn
model = ParzenKNN(k=best_k)
time_now = time.time()
model.fit(X_train, y_train)
time_after_fit = time.time()
print(f"Время обучения собственной модели: {time_after_fit - time_now:.4f} секунд")
y_pred = model.predict(X_test)

sklearn_model = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
time_now = time.time()
sklearn_model.fit(X_train, y_train)
time_after_fit = time.time()
print(f"Время обучения KNeighborsClassifier из sklearn: {time_after_fit - time_now:.4f} секунд")
sk_preds = sklearn_model.predict(X_test)

print("Сравнение с KNeighborsClassifier из sklearn:\n\n")
print("Метрики для собственного ParzenKNN:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")

print("Метрики для KNeighborsClassifier из sklearn:")
print(f"Accuracy: {accuracy_score(y_test, sk_preds):.4f}")
print(f"F1 Score: {f1_score(y_test, sk_preds):.4f}")
print(f"Recall: {recall_score(y_test, sk_preds):.4f}")


def compute_margins(X, y, model):
    """
    Считаем отступы для всех объектов обучающей выборки.
    Margin = y_i * (sum весов свой класс - sum w чужой класс)
    В бинарной реализации predict возвращает знак суммы,
    Нам нужно само число до взятия знака, чтобы оценить уверенность.
    """
    margins = []

    dist_matrix = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2))
    n = len(X)
    for i in range(n):
        dists = dist_matrix[i]
        sorted_indices = np.argsort(dists)

        # берем k соседей, исключая сам объект (индекс 0)
        # надо посмотреть как соеди голосуют за объект без учета своего голоса
        neighbour_indices = sorted_indices[1:model.k+1]

        neighbour_y = y[neighbour_indices]
        neighbour_dists = dists[neighbour_indices]

        # ширина окна
        if model.k + 1 < n:
            h = dists[sorted_indices[model.k + 1]]
        else:
            h = dists[sorted_indices[-1]]

        weights = model.gaussian_kernel(neighbour_dists, h)

        # скалярное произведение
        # если y[i] == 1, а neighbour == 1 -> +weight
        # если y[i] == 1, а neighbour == -1 -> -weight

        vote_score = np.sum(weights * neighbour_y)

        # отступ - умножаем на истинный класс
        margin = y[i] * vote_score
        margins.append(margin)

    return np.array(margins)


def stolp(X, y, k, max_etalons=50):

    # инициализация модели для расчета отступов

    model = ParzenKNN(k=k)
    margins = compute_margins(X, y, model)

    # убираем объекты с отрицатаельным отступом (шумовые)
    clean_indices = np.where(margins > 0)[0]
    X_clean = X[clean_indices]
    y_clean = y[clean_indices]
    margins_clean = margins[clean_indices]
    print(f"Удалено шумовых объектов: {len(X) - len(X_clean)}")

    # начальное множество эталонов
    # берем по одному объекту из класса с наибольшим отступом
    omega_indices = []
    unique_classes = np.unique(y_clean)

    for clsass in unique_classes:
        cls_indices = np.where(y_clean == clsass)[0]
        best_local_idx = np.argmax(margins_clean[cls_indices])
        best_global_idx = cls_indices[best_local_idx]
        omega_indices.append(best_global_idx)

    # наращивание множества
    # используется список индекса относительно X_clean
    omega_indices = list(omega_indices)
    while len(omega_indices) < max_etalons:
        X_omega = X_clean[omega_indices]
        y_omega = y_clean[omega_indices]

        # обучаем только на эталонах
        # k не может быть больше числа эталонов
        curr_k = min(k, len(omega_indices))
        model = ParzenKNN(k=curr_k)
        model.fit(X_omega, y_omega)

        # классифицируем чистую выборку
        preds = model.predict(X_clean)
        
        errors_mask = (preds != y_clean)
        errors_indices = np.where(errors_mask)[0]

        # если ошибок нет, выходим
        if len(errors_indices) == 0:
            break

        # ищем ошибочный объект, которого нет в эталонах
        # добавляем самый сложный объект (минимальный отступ)
        # он лежит на границе

        candidates = [idx for idx in errors_indices if idx not in omega_indices]

        if not candidates:
            break

        # среди кандидатов ищем того, у кого был минимальный отступ в исходном множестве
        candidates_margins = margins_clean[candidates]
        best_candidates_local = np.argmin(candidates_margins)
        idx_to_add = candidates[best_candidates_local]
        omega_indices.append(idx_to_add)

    print(f"Итоговое число эталонов: {len(omega_indices)}")
    return X_clean[omega_indices], y_clean[omega_indices]

X_etalons, y_etalons = stolp(X_train, y_train, k=best_k, max_etalons=50)


# 1. Сжатие для визуализации
pca = PCA(n_components=2)
# Обучаем PCA на всем трейне
X_train_pca = pca.fit_transform(X_train)
# Проецируем эталоны (нам нужно найти их координаты, проще спроецировать заново)
X_etalons_pca = pca.transform(X_etalons)

plt.figure(figsize=(12, 8))

# Рисуем ВСЕ точки (как фон) - тусклые
plt.scatter(X_train_pca[y_train == -1, 0], X_train_pca[y_train == -1, 1], 
            color='blue', alpha=0.15, label='Доброкачественные (все)')
plt.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], 
            color='red', alpha=0.15, label='Злокачественные (все)')

# Рисуем ЭТАЛОНЫ - яркие и большие
plt.scatter(X_etalons_pca[y_etalons == -1, 0], X_etalons_pca[y_etalons == -1, 1], 
            color='blue', marker='o', s=100, edgecolors='k', label='Эталоны доброкачественных')
plt.scatter(X_etalons_pca[y_etalons == 1, 0], X_etalons_pca[y_etalons == 1, 1], 
            color='red', marker='^', s=100, edgecolors='k', label='Эталоны злокачественных')

plt.title(f'Визуализация STOLP (PCA 2D)\nСжатие: {len(X_train)} -> {len(X_etalons)} объектов')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('stolp_etalons_pca.png')

# Модель на эталонах
# k берем поменьше, так как точек стало мало, или такое же, если эталонов > k
k_stolp = min(best_k, len(X_etalons)) 
stolp_model = ParzenKNN(k=k_stolp)
stolp_model.fit(X_etalons, y_etalons)
stolp_preds = stolp_model.predict(X_test)
acc_stolp = accuracy_score(y_test, stolp_preds)

print(f"\nТочность на полной выборке ({len(X_train)} точек): {accuracy_score(y_test, y_pred):.4f}")
print(f"Точность на эталонах ({len(X_etalons)} точек): {acc_stolp:.4f}")
print(f"Сжатие данных в {len(X_train) / len(X_etalons):.1f} раз")