import kagglehub
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

def calculate_metrics(y_true, y_pred):
    """
    Вычисляет основные метрики классификации

    Parameters:
    y_true : array-like, истинные метки классов
    y_pred : array-like, предсказанные метки классов
    """

    # Проверка на одинаковую длину массивов
    if len(y_true) != len(y_pred):
        raise ValueError("Массивы должны иметь одинаковую длину")

    # Вычисление метрик
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Вывод результатов
    print("МЕТРИКИ КЛАССИФИКАЦИИ")
    print("=" * 30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

EPS = 1e-5
MAX_ITERS = 1e5

def drop_outliers(df:pd.DataFrame, columns:list):
    """
    Функция для удаления выбросов.
    На вход принимает дасатет и список колонок для очистки.
    Возвращает очищенный от выбросов датасет.
    """
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q2 = df[column].quantile(0.75)
        IQR = Q2 - Q1
        df = df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q2 + 1.5 * IQR)]

    return df

def column_normalisation(df:pd.DataFrame, columns:list):

    for column in columns:
        median = df[column].median()
        IQR = df[column].quantile(0.75) - df[column].quantile(0.25)
        IQR = IQR if IQR != 0 else 1e-6

        df[column] = df[column].apply(lambda x: (x - median)/IQR)

    return df

dataset_path = kagglehub.dataset_download("muhammedderric/fitness-classification-dataset-synthetic")
csv_path = os.path.join(dataset_path, 'fitness_dataset.csv')
df = pd.read_csv(csv_path)
df['bias'] = 1
df = drop_outliers(df, ['heart_rate', 'weight_kg', 'blood_pressure'])
df = column_normalisation(df, ['age', 'height_cm', 'weight_kg', 'heart_rate', 'blood_pressure', 'sleep_hours', 'nutrition_quality', 'activity_index'])
replace_vals = {
    'smokes': {
        'no': -1,
        'yes': 1,
        '0': -1,
        '1': 1
    },
    'gender': {
        'F': -1,
        'M': 1
    },
    'is_fit': {
        0: -1
    }
}
df['sleep_hours'].fillna(df['sleep_hours'].median(), inplace=True)
df = df.replace(replace_vals)
corr = df.corr().to_csv('t.csv')
print(corr)

def distance(x_1, x_2, p):
    return np.linalg.norm(x_1 - x_2, ord=p, axis=-1)

def kernel(u):
    return (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * u**2)

def indicator(x,y):
    return 1 if x == y else 0

def plot_empirical_risk(risks, k_values):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, risks, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('k (количество соседей)')
    plt.ylabel('Эмпирический риск')
    plt.title('Зависимость эмпирического риска от k')
    plt.grid(True, alpha=0.3)
    plt.show()

class KNN:

    def __init__(self,  p:int, k_optimizator=False, standart_selection=False, k:int=2, tol:int = 0.001):
        self.k = k
        self.p = p
        self.k_optimizator = k_optimizator
        self.standart_selection = standart_selection
        self.tol = tol
        self.comb_cache = {}

    def _predict_single(self, x:np.array, k:int):
        distances = []
        for i in range(self.x_train.shape[0]):
            dist = distance(x, self.x_train[i], self.p)
            distances.append((dist, i))

        distances.sort(key=lambda x: x[0])
        window_width = distances[k-1][0]

        class_sums = [0 for _ in self.classes]

        for dist, idx in distances[:k]:
            u = dist / window_width
            kernel_val = kernel(u)

            for j, class_label in enumerate(self.classes):
                class_sums[j] += kernel_val * indicator(class_label, self.y_train[idx])

        max_index = np.argmax(class_sums)
        predicted_class = self.classes[max_index]
        return predicted_class

    def _k_optimizator(self, k_arr:list):
        res_k = None
        best_err = 1e5
        arr = []
        N = self.x_train.shape[0]

        for k in k_arr:
            error_count = 0
            print(k)
            for i in range(N):
                x_temp = np.delete(self.x_train, i, axis=0)
                y_temp = np.delete(self.y_train, i, axis=0)

                original_X = self.x_train
                original_y = self.y_train

                self.x_train = x_temp
                self.y_train = y_temp

                prediction = self._predict_single(original_X[i], k)

                self.x_train = original_X
                self.y_train = original_y

                if prediction != original_y[i]:
                    error_count += 1

            if error_count < best_err:
                best_err = error_count
                res_k = k
            arr.append(error_count/N)

        self.k = res_k
        return arr

    def _get_comb(self, n, k):
        key = (n, k)
        if key not in self.comb_cache:
            self.comb_cache[key] = math.comb(n, k)
        return self.comb_cache[key]

    def distance_matrix(self):
        n = self.x_train.shape[0]
        self._dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = distance(self.x_train[i], self.x_train[j], self.p)
                self._dist_matrix[i, j] = d
                self._dist_matrix[j, i] = d

    def T(self, i_train=None):
        distances = self._dist_matrix[i_train].astype(float)
        distances[i_train] = np.inf
        L = len(distances)
        l = L - self.k
        idxs = np.argpartition(distances, self.k - 1)[:self.k]
        idxs = idxs[np.argsort(distances[idxs])]

        m_range = np.arange(1, len(idxs) + 1)
        indicators = (self.y_train[i_train] != self.y_train[idxs]).astype(float)

        comb_vals = np.array([self._get_comb(L-1-m, l-1) for m in m_range])
        c = self._get_comb(L-1, l)

        s = np.sum(comb_vals * indicators / c)
        return s

    def CCV(self):
        L = self.x_train.shape[0]
        if L == 0:
            return 0.0
        total = 0.0
        for i in range(L):
            total += self.T(i_train=i)
        return total / L

    def optimal_X(self, tol=0.001):
        ccv_curr = self.CCV()
        removed_global = []
        iteration = 0
        while True:
            L = self.x_train.shape[0]
            mn = 1e7
            ind = -1
            for i in range(L):
                mask = np.ones(L, dtype=bool)
                mask[i] = False

                x_temp = self.x_train[mask]
                y_temp = self.y_train[mask]
                dist_temp = self._dist_matrix[np.ix_(mask, mask)]

                original_x = self.x_train
                original_y = self.y_train
                original_dist = self._dist_matrix

                self.x_train = x_temp
                self.y_train = y_temp
                self._dist_matrix = dist_temp

                res = self.CCV()

                self.x_train = original_x
                self.y_train = original_y
                self._dist_matrix = original_dist

                if res < mn:
                    mn = res
                    ind = i

            if ind == -1 or mn >= ccv_curr - tol:
                break

            global_ind = np.where(self.active_mask)[0]
            global_ind_remove = global_ind[ind]
            removed_global.append(global_ind_remove)
            self.active_mask[global_ind_remove] = False

            print(f"Iter {iteration}: CCV {ccv_curr:.6f} → {mn:.6f}")
            keep_mask = np.ones(self.x_train.shape[0], dtype=bool)
            keep_mask[ind] = False

            self.x_train = self.x_train[keep_mask]
            self.y_train = self.y_train[keep_mask]
            self._dist_matrix = self._dist_matrix[np.ix_(keep_mask, keep_mask)]
            ccv_curr = mn

            iteration += 1
            if iteration > 500:
                break
        return removed_global


    def fit(self, X_train:pd.DataFrame, y_train:pd.DataFrame, arr:list=[]):
        self.x_train = X_train.to_numpy()
        self.y_train = y_train.to_numpy()
        self.distance_matrix()
        self.active_mask = np.ones(self.x_train.shape[0], dtype=bool)
        self.classes = pd.unique(y_train)
        L = []
        del_vals = []
        if self.k_optimizator:
            if not arr:
                print("Не переданы параметры для поиска!")
                return L
            L = self._k_optimizator(arr)
        if self.standart_selection:
            del_vals = self.optimal_X(self.tol)
        return L, del_vals

    def predict(self, X_test: pd.DataFrame):
        X_np = X_test.to_numpy()
        res = []

        for x in X_np:
            prediction = self._predict_single(x, self.k)
            res.append(prediction)

        return res




y_train = df['is_fit']
X_train = df.drop('is_fit', axis=1)
model = KNN(2, True, True, None, 0.0015)
L, arr = model.fit(X_train, y_train, [1,3,6,11,15])
model1 = KNeighborsClassifier(n_neighbors = model.k, p = 2, weights='distance')
model1.fit(X_train, y_train)
print(arr)
plot_empirical_risk(L, [1,3,6,11,15])
y_pred = model.predict(X_train)
print(calculate_metrics(y_train, y_pred))

# ВИЗУАЛИЗАЦИЯ
feat1 = 'activity_index'
feat2 = 'nutrition_quality'

col1_idx = X_train.columns.get_loc(feat1)
col2_idx = X_train.columns.get_loc(feat2)

X_train_orig = X_train.copy().to_numpy()
y_train_orig = y_train.copy().to_numpy()
initial_n = len(y_train_orig)

removed_global = np.array(arr, dtype=int)
mask_removed = np.zeros(initial_n, dtype=bool)
mask_removed[removed_global] = True

X_kept = X_train_orig[~mask_removed]
X_removed = X_train_orig[mask_removed]
y_kept = y_train_orig[~mask_removed]
y_removed = y_train_orig[mask_removed]

X_train_2d = pd.DataFrame(X_train_orig[:, [col1_idx, col2_idx]], columns=[feat1, feat2])
model_2d = KNN(p=2, k=model.k)
model_2d.fit(X_train_2d, pd.Series(y_train_orig))

if len(removed_global) > 0:
    keep_mask = np.ones(len(y_train_orig), dtype=bool)
    keep_mask[removed_global] = False
    model_2d.x_train = X_train_orig[keep_mask][:, [col1_idx, col2_idx]]
    model_2d.y_train = y_train_orig[keep_mask]
    model_2d.distance_matrix()

h = 0.02
x_min, x_max = X_train_orig[:, col1_idx].min() - 0.5, X_train_orig[:, col1_idx].max() + 0.5
y_min, y_max = X_train_orig[:, col2_idx].min() - 0.5, X_train_orig[:, col2_idx].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_df = pd.DataFrame(grid, columns=[feat1, feat2])

Z = np.array(model_2d.predict(grid_df))
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.2, levels=[-1.5, -0.5, 0.5, 1.5], colors=['lightblue', 'lightcoral'])
plt.contour(xx, yy, Z, levels=[-0.5, 0.5], colors='k', linestyles='--', linewidths=1.5)
plt.scatter(X_kept[y_kept == -1, col1_idx], X_kept[y_kept == -1, col2_idx],
            c='blue', marker='.', facecolors='none', edgecolors='blue',
            s=80, linewidths=1.5, label='Not Fit (остался)')
plt.scatter(X_kept[y_kept == 1, col1_idx], X_kept[y_kept == 1, col2_idx],
            c='red', marker='.', s=80, linewidths=2, label='Fit (остался)')
plt.scatter(X_removed[y_removed == -1, col1_idx], X_removed[y_removed == -1, col2_idx],
            c='blue', marker='o', facecolors='none',
            s=120, linewidths=2.5, label='Not Fit (удалён)')
plt.scatter(X_removed[y_removed == 1, col1_idx], X_removed[y_removed == 1, col2_idx],
            c='red', marker='x', s=120, linewidths=3,
            label='Fit (удалён)')
plt.title(f'Отбор эталонов: {feat1} vs {feat2}\nКрасный × = Fit (1), Синий ○ = Not Fit (-1)')
plt.xlabel(feat1)
plt.ylabel(feat2)
plt.legend()
plt.grid(alpha=0.3)
plt.show()