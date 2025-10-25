import kagglehub
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import SGDClassifier

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

class SGD:

    def __init__(self, max_iters:int, tol:float, tau:float, lmb:float, gamma:float, h_max:float, type_of_selection:str, type_of_weight:str, n_starts:int):
        """
            Инициализация модели стохастического спуска. 
            Описание входных параметров:
                max_iters - максимальное количество циклов метода
                tol - точность сходимости
                tau - коэффициент регуляризации
                lmb - коэффициент при реккурентной оценки лосса
                gamma - коэффициент при методе моментов Нестерова
                type_of_selection - тип предъявления объектов ('rand' - случайно, 'margin_md' - по модулю отсупа)
                type_of_weight - тип инициализации весов ('rand' - случайно, 'corr' - через корреляцию)
        """
        self.max_iters = max_iters
        self.tol = tol
        self.tau = tau
        self.lmb = lmb
        self.gamma = gamma
        self.h_max = h_max
        self.type_of_selection = type_of_selection
        self.type_of_weight = type_of_weight
        self.n_starts = n_starts
        self.w = None

    def _weights(self, n:int, X:pd.DataFrame=None, y:pd.DataFrame=None):
        """
            Инициализация весов модели. 
            Описание входных параметров:
                n - количество признаков
                x_j - объект для инициализации весов 
                y_j - таргет для инициализации весов 
        """
        if self.type_of_weight == 'rand':
            low = -1/2*n
            high = 1/2*n
            self.w = np.random.uniform(low=low, high=high, size=(n, 1)).ravel()
        elif self.type_of_weight == 'corr':
            X_np = X.to_numpy()
            y_np = y.to_numpy()
            arr = []
            for i in range(X_np.shape[1]):
                norm_f = X_np[:, i] @ X_np[:, i]
                scal = y_np @ X_np[:, i]
                arr.append(scal/norm_f)
            self.w = np.array(arr)
        elif self.type_of_weight == 'multistart':
            low = -1/2*n
            high = 1/2*n

            best_loss =  np.inf
            best_w = None
            for i in range(self.n_starts):
                self.w = np.random.uniform(low=low, high=high, size=(n, 1)).ravel()
                self.fit(X,y)
                X_np = X.to_numpy()
                y_np = y.to_numpy()
                L = self._compute_loss(X_np, self.w, y_np)
                Q = np.sum(L)/len(L)
                if Q < best_loss:
                    best_loss = Q
                    best_w = self.w
            self.w = best_w

    def _compute_margins(self, x:np.array, w:np.array, y:np.array):
        """
            Расчет отступа объекта 
            Описание входных параметров:
                x - объект
                w - веса модели
                y - таргет
        """
        return y * (x @ w)
    
    def _compute_loss(self, x:np.array, w:np.array, y:np.array):
        """
            Расчет функции оишбки 
            Описание входных параметров:
                x - объект
                w - веса модели
                y - таргет
        """
        M = self._compute_margins(x, w, y)
        return np.log2(1 + np.exp(-M) + 1e-5) + self.tau*np.sum(w**2)/2
    
    def _compute_gradient(self, x:np.array, w:np.array, y:np.array):
        """
            Расчет градиента функции оишбки 
            Описание входных параметров:
                x - объект
                w - веса модели
                y - таргет
        """
        M = self._compute_margins(x, w, y)
        dLdM = -1/(np.log(2)*(1 + np.exp(M) + 1e-5))
        dMdw = y*x
        dL2dw = self.tau*w
        return dLdM*dMdw + dL2dw
    
    def _golden_section_method(self, a:float, b:float, x:np.array, w:np.array, y:np.array, gradient:np.array):
        """
            Метод оптимизации для поиска шага h в наискорейшем спуске
            Описание входных параметров:
                a - левая граница отрезка
                b - правая граница отрезка
                x - объект
                w - веса модели
                y - таргет
                gradient - направление градиента
        """
        PHI = 0.5*(1 + 5**0.5)
        
        iters = 1
        while iters < 50:
            h1 = b - (b - a)/PHI
            y1 = self._compute_loss(x, w - h1*gradient, y)
            h2 = a + (b - a)/PHI
            y2 = self._compute_loss(x, w - h2*gradient, y)
            if y1 >= y2: a = h1
            else: b = h2

            if abs(b-a) < 1e-2:
                break
            iters += 1
        return (a + b)/2
    
    def _probs_from_margins(self, M:float):
        """
            Перевод показателя отступа в вероятность
            Описание входных параметров:
                M - отступ объекта
        """
        a = np.abs(M)
        p = 1.0 / (1.0 + np.exp(a))
        S = p.sum()
        return p / S
    
    def _select_object(self, X:np.array, y:np.array):
        """
            Выбор объекта для обучения
            Описание входных параметров:
                X - признаки модели
                y - таргеты
        """
        i = 0
        if self.type_of_selection == 'rand':
                i = np.random.choice(X.shape[0])
        elif self.type_of_selection == 'margin_md':
            M_all = self._compute_margins(X, self.w, y)
            p_all = self._probs_from_margins(M_all) 
            i = np.random.choice(X.shape[0], p=p_all)
        return i

    def _stochastic_gradient_descent(self, X_np:np.array, y_np:np.array):
        """
            Обучение через стохастический градиентный спуск
            Описание входных параметров:
                X_np - признаки модели
                y_np - таргеты
        """
        v = np.zeros_like(self.w)
        indices = np.random.choice(X_np.shape[0], size=min(50, X_np.shape[0]), replace=False)
        Q = np.mean([self._compute_loss(X_np[i], self.w, float(y_np[i])) for i in indices])

        h = self.h_max
        iters = 1
        Q_arr = []
        
        while iters < self.max_iters:
            
            j = self._select_object(X_np, y_np)
            x_j = X_np[j].ravel()
            y_j = float(y_np[j])

            ksi_j = self._compute_loss(x_j, self.w, y_j)
            grad = self._compute_gradient(x_j, self.w - h*self.gamma*v, y_j)
            Q_arr.append(Q)
            v = self.gamma*v + (1 - self.gamma)*grad
            h = self._golden_section_method(0, self.h_max, x_j, self.w, y_j, v)
            self.w = self.w*(1 - h*self.tau) - h*v

            Q_new = self.lmb*ksi_j + (1 - self.lmb) * Q
            
            if abs(Q_new - Q) < self.tol:
                break
            Q = Q_new
            
            iters += 1
        return Q_arr
    
    def fit(self, X:pd.DataFrame, y:pd.DataFrame):
        """
            Обучение модели
            Описание входных параметров:
                X_np - тренировочный датасет
                y_np - таргеты
        """
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        if self.w is None:
            j = np.random.choice(X_np.shape[0])
            x_j = X_np[j].ravel()
            y_j = float(y_np[j])
            self._weights(X_np.shape[1], X, y)
        arr = self._stochastic_gradient_descent(X_np, y_np)
        return arr

    def predict(self, X:pd.DataFrame):
        """
            Предсказание признаков
            Описание входных параметров:
                X_np - датасет для предсказания
        """
        return np.sign(X @ self.w)

def plot_margins(model:SGD, X_test:pd.DataFrame, y_test:pd.DataFrame):
    """
    Вывод графика
    Описание входных параметров:
        model - обученная модель
    """
    arr = []
    c = 0
    for i, row in X_test.iterrows():
        arr.append(model._compute_margins(row, model.w, y_test.to_list()[c]))
        c += 1
    arr.sort()
    
    plt.figure(figsize=(10, 6))
    plt.plot(arr, label='Отступ классификатора')
    plt.axhline(y=0, color='r', linestyle='--', label='y = 0')
    plt.xlabel('Итерация')
    plt.ylabel('Отступ классификатора')
    plt.title('График отступов классификатора')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

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



Y = df['is_fit']
X = df.drop('is_fit', axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)
# Kf = KFold(n_splits=20, shuffle=True, random_state=42)
# accuracy = 0
# precision = 0
# recall = 0
# f1 = 0
# accuracy_sgd = 0
# precision_sgd = 0
# recall_sgd = 0
# f1_sgd = 0
# for i, j in Kf.split(X):
#     m = SGD(100000, 1e-5, 0.001, 5e-2, 0.9, 0.01, 'rand', 'corr', 10)
#     sgd_classifier = SGDClassifier(
#         loss='log_loss',               
#         penalty='l2',               
#         alpha=0.001,                
#         max_iter=100000,            
#         tol=1e-5,                   
#         learning_rate='constant',   
#         eta0=0.05,                  
#         n_iter_no_change=5,         
#         random_state=42,
#         early_stopping=False
#     )
#     sgd_classifier.fit(X_train, y_train)
#     m.fit(X.iloc[i], Y.iloc[i])
#     y_pred = m.predict(X.iloc[j])
#     y_pred_sgd = sgd_classifier.predict(X.iloc[j])

#     metrics = calculate_metrics(Y.iloc[j], y_pred)
#     accuracy += metrics['accuracy']/20
#     precision += metrics['precision']/20
#     recall += metrics['recall']/20
#     f1 += metrics['f1_score']/20

#     metrics_sgd = calculate_metrics(Y.iloc[j], y_pred_sgd)
#     accuracy_sgd += metrics_sgd['accuracy']/20
#     precision_sgd += metrics_sgd['precision']/20
#     recall_sgd += metrics_sgd['recall']/20
#     f1_sgd += metrics_sgd['f1_score']/20

# print("МЕТРИКИ КЛАССИФИКАЦИИ")
# print("=" * 30)

# print(f"Accuracy:  {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall:    {recall:.4f}")
# print(f"F1-score:  {f1:.4f}")
# print("=" * 30)

# print("Реализация библиотеки")
# print(f"Accuracy:  {accuracy_sgd:.4f}")
# print(f"Precision: {precision_sgd:.4f}")
# print(f"Recall:    {recall_sgd:.4f}")
# print(f"F1-score:  {f1_sgd:.4f}")
model = SGD(100000, 1e-5, 0.001, 5e-2, 0.9, 0.01, 'margin_md', 'multistart', 10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plot_margins(model,X_test, y_test)