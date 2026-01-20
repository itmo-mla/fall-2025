import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from sklearn.svm import SVC
import os

if not os.path.exists('plots'):
    os.makedirs('plots')

X, y = datasets.make_moons(n_samples=300, noise=0.15, random_state=42)
y = np.where(y == 0, -1, 1)

plt.figure(figsize=(8, 6))
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', label='Class -1', alpha=0.6)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class +1', alpha=0.6)
plt.title('Исходные данные')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('plots/01.png', dpi=300, bbox_inches='tight')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Размер обучающей выборки: {X_train_scaled.shape}")
print(f"Размер тестовой выборки: {X_test_scaled.shape}")


class MySVM:
    def __init__(self, C=1.0, kernel='linear', gamma=None, degree=3, coef0=1):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

        self.lambdas_ = None  # двойственные переменные λ
        self.b_ = None  # bias
        self.support_vectors_ = None  # опорные векторы
        self.support_labels_ = None  # метки опорных векторов
        self.X_train = None  # обучающая выборка (для ядра)
        self.y_train = None  # метки обучающей выборки
        self.support_lambdas_ = None  # λ для опорных векторов

    def _kernel(self, X1, X2):
        if self.kernel == 'linear':
            # Линейное ядро: K(x, y) = x·y
            return X1 @ X2.T

        elif self.kernel == 'poly':
            # Полиномиальное ядро: K(x, y) = (γ·x·y + coef0)^d
            return (self.gamma * (X1 @ X2.T) + self.coef0) ** self.degree

        elif self.kernel == 'rbf':
            # RBF (Гауссово) ядро: K(x, y) = exp(-γ·||x-y||²)
            if self.gamma is None:
                self.gamma = 1.0 / (X1.shape[1] * X1.var())
            norm1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            norm2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            pairwise_dist = norm1 + norm2 - 2 * X1 @ X2.T
            pairwise_dist = np.maximum(pairwise_dist, 0)
            return np.exp(-self.gamma * pairwise_dist)

        else:
            raise ValueError(f"Неподдерживаемое ядро: {self.kernel}")

    def _objective_function(self, lambdas, X, y):
        # Вычисляем ядерную матрицу
        K = self._kernel(X, X)

        # Векторизованное вычисление целевой функции
        quadratic_term = 0.5 * np.dot(lambdas * y, np.dot(K, lambdas * y))
        linear_term = np.sum(lambdas)
        result = quadratic_term - linear_term

        if np.isnan(result) or np.isinf(result):
            print(f"!!! objective = {result}!!!")
            print(f"    max(K) = {K.max():.6f}, min(K) = {K.min():.6f}")
            print(f"    max(λ) = {lambdas.max():.6f}, min(λ) = {lambdas.min():.6f}")

        return result

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Сохраняем данные для предсказаний
        self.X_train = X.copy()
        self.y_train = y.copy()

        # 1 - начальное приближение - нулевой вектор λ
        initial_lambdas = np.ones(n_samples) * 0.01

        # 2 - ограничения для двойственных переменных
        # 0 ≤ λ_i ≤ C
        bounds = [(0, self.C) for _ in range(n_samples)]

        # sum(λ_i * y_i) = 0
        constraints = {
            'type': 'eq',
            'fun': lambda lambdas: np.dot(lambdas, y),
            'jac': lambda lambdas: y  # якобиан для ускорения
        }

        # минимизация целевой функции
        print("Начинаем оптимизацию двойственной задачи...")

        initial_value = self._objective_function(initial_lambdas, X, y)
        print(f"Начальное значение целевой функции: {initial_value:.6f}")
        constraint_value = np.dot(initial_lambdas, y)
        print(f"Значение ограничения sum(λ*y): {constraint_value:.6f}\n")

        result = minimize(fun=self._objective_function, x0=initial_lambdas, args=(X, y), method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-10, 'disp': True})

        print(f"\nРезультат оптимизации:")
        print(f"{result.message}")
        print(f"Число итераций - {result.nit}")
        print(f"Финальное значение функции - {result.fun:.6f}")

        # 4 - сохраняем найденные λ
        self.lambdas_ = result.x

        # 5 - находим опорные векторы (λ_i > 1e-5)
        support_indices = np.where(self.lambdas_ > 1e-5)[0]
        self.support_vectors_ = X[support_indices]
        self.support_labels_ = y[support_indices]
        self.support_lambdas_ = self.lambdas_[support_indices]

        print(f"\nНайдено опорных векторов: {len(support_indices)}")
        print(f"Минимальное λ - {self.lambdas_.min():.6e}")
        print(f"Максимальное λ - {self.lambdas_.max():.6e}")
        print(f"Среднее λ - {self.lambdas_.mean():.6e}")

        # 6 - вычисляем смещение b по граничным опорным векторам
        b_values = []
        if len(support_indices) > 0:
            # используем только опорные векторы для эффективности
            K_support = self._kernel(self.support_vectors_, X)
            for i, idx in enumerate(support_indices):
                lambda_i = self.lambdas_[idx]
                # берем только граничные опорные векторы (0 < λ < C)
                if 1e-5 < lambda_i < self.C - 1e-5:
                    # вычисляем f(x_i) без b: sum_j λ_j y_j K(x_j, x_i)
                    f_without_b = np.sum(self.lambdas_ * y * K_support[i, :])
                    # для граничных SV: y_i * f(x_i) = 1 => b = y_i - f(x_i) без b
                    b = self.support_labels_[i] - f_without_b
                    b_values.append(b)

            if b_values:
                self.b_ = np.mean(b_values)
                print(f"Смещение b вычислено по {len(b_values)} граничным опорным векторам: {self.b_:.6f}")
            else:
                print("Нет граничных опорных векторов (все λ на границах [0, C]).")
                print("Вычисляю b по всем опорным векторам...")
                for i, idx in enumerate(support_indices):
                    f_without_b = np.sum(self.lambdas_ * y * K_support[i, :])
                    b = self.support_labels_[i] - f_without_b
                    b_values.append(b)
                if b_values:
                    self.b_ = np.median(b_values)
                    print(f"Смещение b (медиана по всем SV): {self.b_:.6f}")
                else:
                    self.b_ = 0.0
                    print("Не удалось вычислить b, установлено в 0")
        else:
            self.b_ = 0.0
            print("Нет опорных векторов!!! b установлено в 0")

        return self

    def decision_function(self, X):
        if self.support_vectors_ is None or len(self.support_vectors_) == 0:
            return np.zeros(X.shape[0]) + self.b_

        K = self._kernel(self.support_vectors_, X)
        scores = np.dot(self.support_lambdas_ * self.support_labels_, K) + self.b_
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

    def get_params(self):
        return {
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'b': self.b_,
            'n_support_vectors': len(self.support_vectors_) if self.support_vectors_ is not None else 0
        }


def plot_decision_boundary(model, X, y, title="Граница решения SVM", filename=None):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))

    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label='Class -1', edgecolors='k', alpha=0.7)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Class +1', edgecolors='k', alpha=0.7)

    if hasattr(model, 'support_vectors_') and model.support_vectors_ is not None and len(model.support_vectors_) > 0:
        plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=150, facecolors='none', edgecolors='black',
                    linewidths=2, label=f'Опорные векторы ({len(model.support_vectors_)})')

    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    if filename:
        plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')

    plt.show()


print("\n" + "=" * 60)
print("SVM")
print("=" * 60)

# создаем и обучаем модель
svm_custom = MySVM(C=1.0, kernel='linear')
svm_custom.fit(X_train_scaled, y_train)

# предсказания
y_train_pred = svm_custom.predict(X_train_scaled)
y_test_pred = svm_custom.predict(X_test_scaled)

# вычисляем точность
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nРезультаты линейного SVM (моя реализация)")
print(f"Точность на обучающей выборке: {train_acc:.4f}")
print(f"Точность на тестовой выборке:  {test_acc:.4f}")
print(f"Количество опорных векторов:  {len(svm_custom.support_vectors_)}")
print(f"Смещение b: {svm_custom.b_:.6f}")

plot_decision_boundary(svm_custom, X_train_scaled, y_train,
                       title=f"Линейный MySVM\nТочность: {train_acc:.3f} (train), {test_acc:.3f} (test)",
                       filename='02.png')

print("\n" + "=" * 60)
print("СРАВНЕНИЕ ЛИНЕЙНОГО И RBF ЯДЕР")
print("=" * 60)

# Линейное ядро (уже обучили)
print("1. Линейное ядро (C=1.0)")
print(f"  Точность: train={train_acc:.4f}, test={test_acc:.4f}")
print(f"  Опорных векторов: {len(svm_custom.support_vectors_)}")
print(f"  Граничных SV: {sum((svm_custom.support_lambdas_ > 1e-5) & (svm_custom.support_lambdas_ < 0.999))}")

# RBF ядро
print("\n2. RBF ядро (C=1.0, gamma='auto')")
svm_rbf = MySVM(C=1.0, kernel='rbf', gamma=None)
svm_rbf.fit(X_train_scaled, y_train)

y_train_pred_rbf = svm_rbf.predict(X_train_scaled)
y_test_pred_rbf = svm_rbf.predict(X_test_scaled)

train_acc_rbf = accuracy_score(y_train, y_train_pred_rbf)
test_acc_rbf = accuracy_score(y_test, y_test_pred_rbf)

print(f"  Точность: train={train_acc_rbf:.4f}, test={test_acc_rbf:.4f}")
print(f"  Опорных векторов: {len(svm_rbf.support_vectors_)}")
if svm_rbf.support_lambdas_ is not None:
    boundary_sv = sum((svm_rbf.support_lambdas_ > 1e-5) & (svm_rbf.support_lambdas_ < 0.999))
    print(f"  Граничных SV: {boundary_sv}")


def plot_on_axis(model, X, y, title, ax, show_sv=True):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label='Class -1', edgecolors='k', alpha=0.6)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Class +1', edgecolors='k', alpha=0.6)

    if show_sv and model.support_vectors_ is not None and len(model.support_vectors_) > 0:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   s=100, facecolors='none', edgecolors='black',
                   linewidths=1.5, label=f'SV ({len(model.support_vectors_)})')

    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# линейное ядро
plot_on_axis(svm_custom, X_train_scaled, y_train,
             f"Линейное ядро\nТочность: {train_acc:.3f} (train), {test_acc:.3f} (test)",
             axes[0])

# RBF ядро
plot_on_axis(svm_rbf, X_train_scaled, y_train,
             f"RBF ядро\nТочность: {train_acc_rbf:.3f} (train), {test_acc_rbf:.3f} (test)",
             axes[1])

plt.tight_layout()
plt.savefig('plots/03.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("ВЛИЯНИЕ ПАРАМЕТРА C ДЛЯ RBF ЯДРА")
print("=" * 60)

C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
results = []

for C in C_values:
    svm_temp = MySVM(C=C, kernel='rbf', gamma=None)
    svm_temp.fit(X_train_scaled, y_train)

    train_acc_temp = accuracy_score(y_train, svm_temp.predict(X_train_scaled))
    test_acc_temp = accuracy_score(y_test, svm_temp.predict(X_test_scaled))

    results.append({'C': C, 'train_acc': train_acc_temp, 'test_acc': test_acc_temp,
                    'n_sv': len(svm_temp.support_vectors_) if svm_temp.support_vectors_ is not None else 0
                    })
    print(f"C={C:6.2f}: train={train_acc_temp:.4f}, test={test_acc_temp:.4f}, SV={results[-1]['n_sv']}")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# График точности
C_vals = [r['C'] for r in results]
train_accs = [r['train_acc'] for r in results]
test_accs = [r['test_acc'] for r in results]

ax1.semilogx(C_vals, train_accs, 'o-', label='Train accuracy', linewidth=2)
ax1.semilogx(C_vals, test_accs, 's-', label='Test accuracy', linewidth=2)
ax1.set_xlabel('Параметр C (лог. шкала)')
ax1.set_ylabel('Точность')
ax1.set_title('Влияние параметра C на точность (RBF ядро)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# График числа опорных векторов
n_svs = [r['n_sv'] for r in results]
ax2.semilogx(C_vals, n_svs, 'o-', color='green', linewidth=2)
ax2.set_xlabel('Параметр C (лог. шкала)')
ax2.set_ylabel('Число опорных векторов')
ax2.set_title('Влияние параметра C на число опорных векторов')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/04.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nВизуализация границ для разных значений C...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, C in enumerate(C_values):
    if i < len(axes):
        svm_temp = MySVM(C=C, kernel='rbf', gamma=None)
        svm_temp.fit(X_train_scaled, y_train)

        plot_on_axis(svm_temp, X_train_scaled, y_train,
                     f"RBF ядро, C={C}\nТочность: {results[i]['train_acc']:.3f} (train), {results[i]['test_acc']:.3f} (test)",
                     axes[i], show_sv=False)

for i in range(len(C_values), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig('plots/05.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("СРАВНЕНИЕ С ЭТАЛОННОЙ РЕАЛИЗАЦИЕЙ")
print("=" * 60)

print("1. Линейное ядро (C=1.0):")
sklearn_linear = SVC(kernel='linear', C=1.0)
sklearn_linear.fit(X_train_scaled, y_train)

sk_linear_train = sklearn_linear.score(X_train_scaled, y_train)
sk_linear_test = sklearn_linear.score(X_test_scaled, y_test)

print(f"   sklearn: train={sk_linear_train:.4f}, test={sk_linear_test:.4f}")
print(f"   Мой:     train={train_acc:.4f}, test={test_acc:.4f}")
print(f"   Разница: train={abs(sk_linear_train - train_acc):.4f}, test={abs(sk_linear_test - test_acc):.4f}")

print("\n2. RBF ядро (C=1.0, gamma='scale'):")
sklearn_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
sklearn_rbf.fit(X_train_scaled, y_train)

sk_rbf_train = sklearn_rbf.score(X_train_scaled, y_train)
sk_rbf_test = sklearn_rbf.score(X_test_scaled, y_test)

print(f"   sklearn: train={sk_rbf_train:.4f}, test={sk_rbf_test:.4f}")
print(f"   Мой:     train={train_acc_rbf:.4f}, test={test_acc_rbf:.4f}")
print(f"   Разница: train={abs(sk_rbf_train - train_acc_rbf):.4f}, test={abs(sk_rbf_test - test_acc_rbf):.4f}")


print("-" * 60)
print(f"{'Метод':<20} {'Train Acc':<12} {'Test Acc':<12} {'SV Count':<12}")
print("-" * 60)
print(f"{'Мой линейный':<20} {train_acc:<12.4f} {test_acc:<12.4f} {len(svm_custom.support_vectors_):<12}")
print(f"{'Sklearn линейный':<20} {sk_linear_train:<12.4f} {sk_linear_test:<12.4f} {sklearn_linear.n_support_.sum():<12}")
print(f"{'Мой RBF':<20} {train_acc_rbf:<12.4f} {test_acc_rbf:<12.4f} {len(svm_rbf.support_vectors_):<12}")
print(f"{'Sklearn RBF':<20} {sk_rbf_train:<12.4f} {sk_rbf_test:<12.4f} {sklearn_rbf.n_support_.sum():<12}")
