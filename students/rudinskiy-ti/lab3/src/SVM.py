import kagglehub
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

EPS = 1e-5
MAX_ITERS = 1e5

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

def separation_plot(x_train, y_train, model):
    model.fit(x_train, y_train)
    print("SV count:", np.sum(model.lmb > model.tol)) 
    print("margin SV count:", np.sum((model.lmb > model.tol) & (model.lmb < model.C - model.tol))) 
    print("bias:", model.bias)
    h = 0.02  
    x_min, x_max = x_train.iloc[:, 0].min() - 1, x_train.iloc[:, 0].max() + 1
    y_min, y_max = x_train.iloc[:, 1].min() - 1, x_train.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    Z = model.decision_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.6)
    plt.contour(xx, yy, Z, levels=[0], colors='k', linestyles='--', linewidths=2)
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='k', linewidth=2, label='Support Vectors')
    colors = ['red' if label == -1 else 'blue' for label in y_train]
    plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=colors, alpha=0.7, label='Data points')
    plt.xlabel(features_to_plot[0])
    plt.ylabel(features_to_plot[1])
    plt.title('SVM: Разделяющая гиперплоскость (гауссово ядро)')
    plt.legend()
    plt.colorbar(label='Decision function')
    plt.grid(True)
    plt.show()

class SVM:

    def __init__(self, max_iter, tol, C, kernel_type, gamma=0.01):
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
        self.kernel_type = kernel_type
        self.gamma = gamma

    def kernel(self, x, y):
        if self.kernel_type == 'simple':
            return np.dot(x,y)
        if self.kernel_type == 'gauss':
            return np.exp(-self.gamma*np.linalg.norm(x - y)**2)

    def fit(self, x_train, y_train):
        if isinstance(x_train, pd.DataFrame):
            self.x_np = x_train.to_numpy()
        elif isinstance(x_train, np.ndarray):
            self.x_np = x_train

        if isinstance(y_train, pd.Series):
            self.y_np = y_train.to_numpy()
        elif isinstance(y_train, pd.DataFrame):
            self.y_np = y_train.to_numpy().ravel()
        elif isinstance(y_train, np.ndarray):
            self.y_np = y_train.ravel()

        n = self.x_np.shape[0]
        self.K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.K[i, j] = self.kernel(self.x_np[i], self.x_np[j])

        self.lmb = self._find_weigths(self.K, self.y_np)
        idxs = np.where((self.lmb > self.tol) & (self.lmb < self.C - self.tol))[0]
        sv_indices = np.where(self.lmb > self.tol)[0]
        if (len(idxs) == 0):
          idxs = sv_indices
        self.support_vectors_ = self.x_np[sv_indices]
        arr = []
        for i in idxs:
            s = np.sum(self.lmb * self.y_np * self.K[:, i])
            arr.append(s - self.y_np[i])
        self.bias = float(np.median(np.array(arr)))
        

    def _find_weigths(self, K, y_np):
        L = K.shape[0]
        Y = np.dot(y_np[:, None], y_np[None, :])

        def Z(lmb):
            Q = lmb.T @ (Y * K) @ lmb
            return -np.sum(lmb) + 0.5 * Q

        cons = ({'type': 'eq', 'fun': lambda lmb:  np.dot(lmb, y_np)})
        bounds = [(0, self.C) for _ in range(L)]
        res = scipy.optimize.minimize(Z, np.zeros(L), method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': self.max_iter})
        return res.x

    def decision_func(self, x_test):
        L = self.x_np.shape[0]
        if isinstance(x_test, pd.DataFrame):
            x_test_np = x_test.to_numpy()
        elif isinstance(x_test, np.ndarray):
            x_test_np = x_test
        M = x_test_np.shape[0]
        K_test = np.zeros((M, L))
        for i in range(M):
            for j in range(L):
                K_test[i, j]= self.kernel(x_test_np[i],self.x_np[j])
        res = (self.lmb*self.y_np ) @ K_test.T - self.bias
        return res

    def predict(self, x_test):
        return np.sign(self.decision_func(x_test))

dataset_path = kagglehub.dataset_download("muhammedderric/fitness-classification-dataset-synthetic")
csv_path = os.path.join(dataset_path, 'fitness_dataset.csv')
df = pd.read_csv(csv_path)
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

Y = df['is_fit']
X = df.drop('is_fit', axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)
x_np = X_train
y_np = y_train.to_numpy()
model = SVM(50, 1e-4, 0.5, 'gauss', 0.015)
model.fit(X_train, y_train)
res = model.predict(X_test)
svm_sklearn = SVC(
    kernel='rbf',
    C=0.5,
    tol=1e-4,
    max_iter=5000,
    random_state=42,
)
svm_sklearn.fit(X_train, y_train)
res_sklearn = svm_sklearn.predict(X_test)
print(calculate_metrics(y_test, res))
print(calculate_metrics(y_test, res_sklearn))
features_to_plot = ['heart_rate', 'activity_index']
X_plot = X_train[features_to_plot]
y_plot = y_train
separation_plot(X_plot, y_plot, model)