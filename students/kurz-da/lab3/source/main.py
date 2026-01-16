import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd


# 1. Загрузить сырые X, y (без scaling и PCA)
def load_raw_data(path="data.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    y = df["diagnosis"].map({"M": 1, "B": -1}).values
    X = df.drop(columns=["diagnosis"]).values.astype(float)
    return X, y

# 2. Разделить
X, y = load_raw_data()
Xtr_raw, Xte_raw, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Fit scaler и PCA только на train
scaler = StandardScaler()
Xtr_scaled = scaler.fit_transform(Xtr_raw)
Xte_scaled = scaler.transform(Xte_raw)

pca = PCA(n_components=2)
Xtr = pca.fit_transform(Xtr_scaled)
Xte = pca.transform(Xte_scaled)


class DualSVM:
    def __init__(self, C=1.0, kernel='linear', gamma=0.5):
        self.C = C        # Константа регуляризации
        self.kernel = kernel
        self.gamma = gamma # Параметр для RBF ядра
        self.lambdas = None
        self.w0 = 0.0     # Смещение (bias)
        self.support_vectors_ = None
        self.support_labels_ = None
        self.support_lambdas_ = None

    def _kernel_function(self, X1, X2):
        """
        Реализация ядер.
        "Трюк с ядром" позволяет строить нелинейную границу (напр. rbf),
        оставаясь в dual-формулировке

        вес w - это линейный вектор в исходном пространстве
        если заменим x_i -> ϕ(x_i), w будет жить в пространстве φ, возможно бесконечномерном

        Лямбды - двойственные переменные (множители Лагранжа).
        Трюк заключается в том, что мы вычисляем K(x_i, x_j), минуя ϕ
        """
        # решение соответствует обычной линейной разделяющей гиперплоскости (2D после PCA)
        if self.kernel == 'linear':
            # K(x, x') = <x, x'>
            return np.dot(X1, X2.T)
        
        elif self.kernel == 'rbf':
            # K(x, x') = exp(-gamma * ||x - x'||^2)
            # Вычисляем матрицу евклидовых расстояний
            if X1.ndim == 1: X1 = X1.reshape(1, -1)
            if X2.ndim == 1: X2 = X2.reshape(1, -1)
            
            # (a-b)^2 = a^2 + b^2 - 2ab
            sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                       np.sum(X2**2, axis=1) - \
                       2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma * sq_dists)

    def fit(self, X, y):
        n_samples = X.shape[0]
        
        # РЕШЕНИЕ ДВОЙСТВЕННОЙ ЗАДАЧИ ПО ЛЯМБДА
        # 1. Матрица Грама: K_ij = K(x_i, x_j)
        K = self._kernel_function(X, X)
        
        # 2. Целевая функция (Objective) для минимизации
        # Мы хотим максимизировать L(lambda).
        # minimize( -L ) = minimize( 0.5 * sum(Li Lj yi yj Kij) - sum(Li) )
        
        # P_ij = y_i * y_j * K_ij
        # матричная форма второго слагаемого двойственной функции
        # np.outer(y, y) -> матрица y_i y_j
        # K - матрица ядра
        # нужно чтобы вся двойст. функция записывалась компактно
        P = np.outer(y, y) * K
        
        # минус двойственной целевой функции SVM
        # dual-SVM максимизирует, scipy.optimize.minimize минимизирует
        # поэтому мы минимизируем -L(lambda)
        def objective(lambdas):
            # 0.5 * lambda^T * P * lambda - sum(lambda)
            # первый член - геометрия
            # второй - максимизация зазора
            return 0.5 * np.dot(lambdas, np.dot(P, lambdas)) - np.sum(lambdas)
        
        def objective_grad(lambdas):
            # Градиент для ускорения оптимизации: P * lambda - 1
            # Передача в SLSQP ускоряет сходимость
            # повышает стабильность
            # уменьщает число итераций
            return np.dot(P, lambdas) - np.ones(n_samples)

        # 3. Ограничения (Constraints)
        # sum(lambda_i * y_i) = 0 
        # type eq - ограниечение равенство. Такое, что fun(λ)=0
        # Потому что условие стационарности - ∂L/∂b = 0 => ∑_i λ_i y_i = 0
        # В scipy.optimize.minimize ограничения задаются в виде функций, которые:
        # для eq должны быть равны нулю
        # для ineq должны быть ≥ 0
        constraints = [{'type': 'eq', 'fun': lambda lambdas: np.dot(lambdas, y)}]
        
        # 4. Границы (Bounds)
        # Для каждой λᵢ задаётся ограничение:
        # 0 <= lambda_i <= C 
        # Это soft-margin SVM
        # нижняя граница λ_i ≥ 0 — стандартное требование для множителей Лагранжа
        # верхняя граница λ_i ≤ C — результат добавления штрафа за ошибки (slack variables)

        # маленькая λ_i → точка не влияет на границу
        # λ_i = 0 → точка не опорная
        # 0 < λ_i < C → точка лежит на margin
        # λ_i = C → точка нарушает margin или классифицирована с ошибкой
        bounds = [(0, self.C) for _ in range(n_samples)]
        
        # 5. Решение оптимизационной задачи
        print(f"Решаем двойственную задачу (ядро: {self.kernel})...")
        # Начальное приближение
        initial_lambdas = np.zeros(n_samples)
        
        # Численно решаем двойственную задачу SVM, 
        # находя оптимальные множители Лагранжа λ_i при заданных ограничениях
        result = minimize(fun=objective, 
                          x0=initial_lambdas, 
                          method='SLSQP', # Sequential Least Squares Programming
                        # это оптимизатор, который умеет решать задачи с ограничениями, 
                        # то есть когда нужно минимизировать функцию, 
                        # одновременно соблюдая равенства и границы на переменные.
                          jac=objective_grad, 
                          bounds=bounds, 
                          constraints=constraints)
        
        if not result.success:
            print("Внимание: Оптимизация не сошлась!")
            
        self.lambdas = result.x

        # Решаем dual-задачу SVM по λ_i — множителям Лагранжа.
        # Ограничение Σ_i λ_i y_i = 0 следует из условия стационарности по смещению b,
        # а ограничения 0 ≤ λ_i ≤ C отражают soft-margin постановку и баланс между максимизацией зазора и штрафом за ошибки.

        # условие стационарности - это требование, чтобы в точке оптимума
        # производная целевой функции по оптимизируемой переменной была равна нулю
        
        # 6. Отбор опорных векторов 
        # Оставляем только те, где lambda > 0 (на практике > 1e-5)
        mask = self.lambdas > 1e-5
        self.support_vectors_ = X[mask]
        self.support_labels_ = y[mask]
        self.support_lambdas_ = self.lambdas[mask]
        
        print(f"Найдено опорных векторов: {len(self.support_vectors_)}")
        
        # 7. Расчет смещения w0 (bias)
        # w0 = <w, x_i> - y_i для любого опорного вектора,
        # который лежит строго на границе (0 < lambda < C).
        # <w, x_i> раскрывается как sum(lambda_j * y_j * K(x_j, x_i))
        
        # Ищем индексы граничных векторов (не нарушителей)
        margin_mask = (self.support_lambdas_ > 1e-5) & (self.support_lambdas_ < self.C - 1e-5)
        
        if np.any(margin_mask):
            # Берем индексы внутри множества опорных
            sv_boundary = self.support_vectors_[margin_mask]
            sy_boundary = self.support_labels_[margin_mask]
            
            # Считаем w0 как среднее по всем граничным векторам для устойчивости
            w0_vals = []
            for i in range(len(sv_boundary)):
                # sum(lambda_j * y_j * K(x_j, x_boundary))
                # Вычисляем ядро между всеми опорными и текущим граничным
                k_vals = self._kernel_function(self.support_vectors_, sv_boundary[i].reshape(1,-1)).flatten()
                
                prediction_no_bias = np.sum(self.support_lambdas_ * self.support_labels_ * k_vals)
                # w0 = prediction_no_bias - y_i
                w0_vals.append(sy_boundary[i] - prediction_no_bias)
                
            self.w0 = np.mean(w0_vals)
        else:
            # Если нет строгих граничных (редкий случай, переобучение), берем любой опорный
            self.w0 = 0 # fallback

    def predict(self, X):
        """
        Формула классификации:
        a(x) = sign( sum(lambda_i * y_i * K(x, x_i)) - w0 )
        """
        if self.support_vectors_ is None:
            return np.zeros(X.shape[0])
            
        # Считаем ядро K(x, x_i) между новыми точками и опорными векторами
        K_matrix = self._kernel_function(X, self.support_vectors_)
        
        # Взвешенная сумма
        # (N_samples, N_sv) dot (N_sv, ) -> (N_samples, )
        decision = np.dot(K_matrix, self.support_lambdas_ * self.support_labels_) + self.w0
        
        return np.sign(decision)

    def decision_function(self, X):
        # Возвращает расстояние до разделяющей плоскости (без sign) для визуализации
        K_matrix = self._kernel_function(X, self.support_vectors_)
        return np.dot(K_matrix, self.support_lambdas_ * self.support_labels_) + self.w0
    

def plot_svm_contours(model, X, y, title="SVM"):
    plt.figure(figsize=(10, 6))
    
    # Сетка для фона
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Предсказание для каждой точки сетки (расстояние до границы)
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Заливка полей принятия решений
    plt.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['blue', 'red'], alpha=0.2)
    
    # Отрисовка линий: граница (0) и отступы (-1, 1) [cite: 66, 67]
    # На слайде 5 показана полоса {x : -1 <= <w,x> - w0 <= 1}
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')
    
    # Точки данных
    plt.scatter(X[y==-1][:,0], X[y==-1][:,1], color='blue', label='Class -1', s=30, alpha=0.7)
    plt.scatter(X[y==1][:,0], X[y==1][:,1], color='red', label='Class +1', s=30, alpha=0.7)
    
    # Выделение опорных векторов 
    plt.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1],
                s=150, linewidth=1.5, facecolors='none', edgecolors='green', label='Support Vectors')
    
    plt.title(title)
    plt.legend()
    plt.savefig(f"svm_{title.lower().replace(' ', '_')}.png")
    plt.show()

# --- ЭКСПЕРИМЕНТ 1: Линейное ядро (Linear SVM) ---
# Сравнение с эталонным решением
print("--- Линейное Ядро ---")
my_linear_svm = DualSVM(C=1.0, kernel='linear')
my_linear_svm.fit(Xtr, ytr)
plot_svm_contours(my_linear_svm, Xtr, ytr, "SVM (Линейное ядро)")

sk_linear = SVC(kernel='linear', C=1.0)
sk_linear.fit(Xtr, ytr)

print(f"Accuracy линейного ядра: {accuracy_score(yte, my_linear_svm.predict(Xte)):.4f}")
print(f"Accuracy линейного ядра Sklearn: {accuracy_score(yte, sk_linear.predict(Xte)):.4f}")

# --- ЭКСПЕРИМЕНТ 2: Трюк с ядром (RBF Kernel) ---
# Реализация нелинейного классификатора 
print("\n--- RBF Ядро (нелинейное) ---")
my_rbf_svm = DualSVM(C=1.0, kernel='rbf', gamma=0.5)
my_rbf_svm.fit(Xtr, ytr)
plot_svm_contours(my_rbf_svm, Xtr, ytr, "SVM (RBF ядро)")

sk_rbf = SVC(kernel='rbf', C=1.0, gamma=0.5)
sk_rbf.fit(Xtr, ytr)

print(f"RBF Accuracy: {accuracy_score(yte, my_rbf_svm.predict(Xte)):.4f}")
print(f"Sklearn RBF Accuracy: {accuracy_score(yte, sk_rbf.predict(Xte)):.4f}")


