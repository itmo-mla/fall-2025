import kagglehub
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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

def sigm_func(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

class LogisticRegression:

    def __init__(self, tol, lr, max_iter, method):
        self.tol = tol
        self.lr = lr
        self.method = method
        self.max_iter = max_iter
        self.w = None
    
    def _weights_init_(self, n):
        low = -1/(2*n)
        high = 1/(2*n)
        self.w = np.random.uniform(low=low, high=high, size=(n, 1)).ravel()
    
    def fit(self, x_np, y_np):
        m, n = x_np.shape
        x_np = np.hstack([np.ones((m, 1)), x_np])
        arr = []
        if self.method == 'NR':
            self._weights_init_(n + 1)
            for cnt in range(self.max_iter):
                prev_w = self.w.copy()
                z = y_np * (x_np @ self.w) 
                sigm = sigm_func(z)
                M = y_np @ (x_np @ self.w)
                Q = np.log(1 + np.exp(-M))
                arr.append(Q)
                dQ = - x_np.T @ ((1 - sigm) * y_np)
                d = (1 - sigm) * sigm
                ddQ = x_np.T @ (d[:, None] * x_np)
                delta = np.linalg.solve(ddQ, dQ)
                self.w = prev_w - self.lr * delta
                if np.linalg.norm(self.w - prev_w) < self.tol:
                    break
        elif self.method == 'IRLS':
            prev_sigm = None
            self.w = np.linalg.solve(x_np.T @ x_np, x_np.T @ y_np)
            for cnt in range(self.max_iter):
                M = y_np @ (x_np @ self.w)
                Q = np.log(1 + np.exp(-M))
                arr.append(Q)
                z = y_np * (x_np @ self.w) 
                sigm = sigm_func(z)
                gamma = np.sqrt((1 - sigm) * sigm)
                D = np.diag(gamma)
                x_np_ = D @ x_np
                y_np_ = y_np * np.sqrt((1 - sigm) / sigm)
                delta = np.linalg.solve(x_np_.T @ x_np_, x_np_.T @ y_np_)
                self.w = self.w + self.lr * delta
                if prev_sigm is not None:
                    if np.max(np.abs(sigm - prev_sigm)) < self.tol:
                        print(np.max(np.abs(sigm - prev_sigm)))
                        break
                prev_sigm = sigm.copy()
        return arr
    
    def predict(self, x_np):
        m = x_np.shape[0]
        x_np = np.hstack([np.ones((m, 1)), x_np])
        z = x_np @ self.w
        pred = np.sign(z)
        pred[pred == 0] = 1
        return pred
    
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

model = LogisticRegression(1e-3, 1e-2, 150, 'NR')
arr = model.fit(X_train.to_numpy(), y_train.to_numpy())
y_pred = model.predict(X_test.to_numpy())
calculate_metrics(y_test, y_pred)
plt.plot(arr)
plt.show()

sk_model = SklearnLR(
    tol=1e-3,
    max_iter=150,
    solver='newton-cg',     
    fit_intercept=True 
)
sk_model.fit(X_train, y_train)
y_pred_sk = sk_model.predict(X_test)
calculate_metrics(y_test, y_pred_sk)