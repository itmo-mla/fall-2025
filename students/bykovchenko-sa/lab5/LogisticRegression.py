import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.stats import ttest_rel, wilcoxon
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

print("\n" + "=" * 70)
print("Анализ датасета BREAST CANCER WISCONSIN")
print("=" * 70)
print(f"Размерность данных: {X.shape}")
print(f"Количество признаков: {X.shape[1]}")
print(f"Количество объектов: {X.shape[0]}")
print(f"\nЦелевая переменная: {target_names}")
print(f"Распределение классов {np.bincount(y)}")
print(f"    - Злокачественные (Malignant): {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")
print(f"    - Доброкачественные (Benign): {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['target_name'] = df['target'].map({0: 'malignant', 1: 'benign'})

print("\nОсновные статистики по признакам:")
print(df.describe().T[['mean', 'std', 'min', 'max']].head(8))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nРазмер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"Распределение классов в обучающей выборке: {np.bincount(y_train)}")
print(f"Распределение классов в тестовой выборке: {np.bincount(y_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Визуализация распределения признаков
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Признак 1 worst radius
axes[0].hist(X_train_scaled[y_train == 0, 20], alpha=0.7, label='Malignant', bins=25, color='red', density=True)
axes[0].hist(X_train_scaled[y_train == 1, 20], alpha=0.7, label='Benign', bins=25, color='blue', density=True)
axes[0].set_xlabel('worst radius (scaled)')
axes[0].set_ylabel('Плотность')
axes[0].set_title('Распределение worst radius по классам')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Признак 2 worst concave points
axes[1].hist(X_train_scaled[y_train == 0, 27], alpha=0.7, label='Malignant', bins=25, color='red', density=True)
axes[1].hist(X_train_scaled[y_train == 1, 27], alpha=0.7, label='Benign', bins=25, color='blue', density=True)
axes[1].set_xlabel('worst concave points (scaled)')
axes[1].set_ylabel('Плотность')
axes[1].set_title('Распределение worst concave points по классам')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '1_distribution_features.png'), dpi=300, bbox_inches='tight')
plt.show()


class MyLogisticRegression:
    """Логистическая регрессия с методом Ньютона-Рафсона"""

    def __init__(self, max_iter=100, tol=1e-8, reg_param=0.1, alpha=0.5):
        """
        Параметры:
        max_iter: максимальное число итераций
        tol: критерий остановки (изменение весов)
        reg_param: параметр регуляризации L2 (λ)
        alpha: коэффициент шага (learning rate)
        """
        self.max_iter = max_iter
        self.tol = tol
        self.reg_param = reg_param
        self.alpha = alpha
        self.w = None
        self.loss_history = []
        self.grad_norm_history = []

    def sigmoid(self, z):
        """Сигмоидная функция σ(z) = 1/(1 + e^{-z})

        Соответствует обратной логит-функции в GLM
        g(μ) = log(μ/(1-μ)) = w^T x  =>  μ = σ(w^T x) = 1/(1 + e^{-w^T x})
        """
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    def compute_loss(self, X, y):
        """Вычисление функции потерь с регуляризацией L2

        L(M) = log(1 + e^{-M}) - логарифмическая функция потерь
        где M_i = w^T x_i y_i

        Для бинарной классификации с y_i ∈ {0,1}:
        L(w) = -1/n Σ [y_i log(σ(w^T x_i)) + (1-y_i) log(1-σ(w^T x_i))]

        С регуляризацией: L_reg(w) = L(w) + λ/2 ||w||^2
        """
        z = X @ self.w
        p = self.sigmoid(z)

        loss = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))
        reg_loss = 0.5 * self.reg_param * np.sum(self.w[1:] ** 2)

        return loss + reg_loss

    def fit(self, X, y, verbose=False):
        """Обучение модели методом Ньютона-Рафсона

        α^{t+1} = α^t - h_t [Q''(α^t)]^{-1} Q'(α^t)

        Для логистической регрессии:
        Q'(w) = X^T (σ(Xw) - y)  - градиент
        Q''(w) = X^T D X         - гессиан, D = diag(σ_i(1-σ_i))
        """
        X_bias = np.c_[np.ones(X.shape[0]), X]
        n_samples, n_features = X_bias.shape
        np.random.seed(42)
        self.w = np.random.randn(n_features) * 0.01

        if verbose:
            print(f"\nНачало обучения Newton-Raphson:")
            print(f"    Размерность данных: {X_bias.shape}")
            print(f"    Параметр регуляризации λ: {self.reg_param}")
            print(f"    Коэффициент шага α: {self.alpha}")

        for i in range(self.max_iter):
            # 1. Прямой проход: вычисление вероятностей
            z = X_bias @ self.w     # линейная комбинация
            p = self.sigmoid(z)     # σ(z) = 1/(1+e^{-z})

            # 2. Вычисление градиента
            # ∇L(w) = 1/n X^T (p - y)
            gradient = X_bias.T @ (p - y) / n_samples

            # Добавляем регуляризацию к градиенту
            gradient[1:] += self.reg_param * self.w[1:] / n_samples

            # 3. Вычисление гессиана
            # H = 1/n X^T D X, где D = diag(p_i(1-p_i))
            W = np.diag(p * (1 - p))    # диагональная весовая матрица
            hessian = (X_bias.T @ W @ X_bias) / n_samples

            # Добавляем регуляризацию к гессиану
            hessian[1:, 1:] += self.reg_param * np.eye(n_features - 1) / n_samples

            # Стабилизация гессиана для предотвращения вырожденности
            hessian += np.eye(n_features) * 1e-6

            # 4. Решение системы: H * Δw = ∇L
            # Δw = H^{-1} ∇L
            try:
                delta_w = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                # Если матрица вырождена, используем псевдообратную
                if verbose and i == 0:
                    print(" Предупреждение!!! Использование псевдообратной матрицы")
                delta_w = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

            # 5. Обновление весов: w_new = w - α * Δw
            # где α - коэффициент шага
            self.w -= self.alpha * delta_w

            # Сохранение истории
            loss = self.compute_loss(X_bias, y)
            self.loss_history.append(loss)
            self.grad_norm_history.append(np.linalg.norm(gradient))

            # Проверка критерия остановки
            if np.linalg.norm(delta_w) < self.tol:
                if verbose:
                    print(f"    Сходимость на итерации {i + 1}, loss {loss:.6f}")
                break

        if verbose and i == self.max_iter - 1:
            print(f"    Достигнут лимит итераций ({self.max_iter}), loss {loss:.6f}")

        return self

    def predict_proba(self, X):
        """Предсказание вероятностей"""
        X_bias = np.c_[np.ones(X.shape[0]), X]
        return self.sigmoid(X_bias @ self.w)

    def predict(self, X, threshold=0.5):
        """Предсказание классов"""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_params(self):
        """Получение параметров модели"""
        return {
            'weights': self.w,
            'bias': self.w[0],
            'n_iterations': len(self.loss_history),
            'final_loss': self.loss_history[-1] if self.loss_history else None
        }


class LogisticRegressionIRLS:
    """Логистическая регрессия с методом IRLS"""

    def __init__(self, max_iter=100, tol=1e-8, reg_param=0.1):
        """
        Параметры:
        max_iter: максимальное число итераций
        tol: критерий остановки (изменение весов)
        reg_param: параметр регуляризации L2 (λ)
        """
        self.max_iter = max_iter
        self.tol = tol
        self.reg_param = reg_param
        self.w = None
        self.loss_history = []

    def sigmoid(self, z):
        """Стабильная реализация сигмоидной функции"""
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    def compute_loss(self, X, y):
        """Вычисление функции потерь с регуляризацией L2"""
        z = X @ self.w
        p = self.sigmoid(z)
        loss = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))
        reg_loss = 0.5 * self.reg_param * np.sum(self.w[1:] ** 2)
        return loss + reg_loss

    def fit(self, X, y, verbose=False):
        """Обучение модели методом IRLS (Iteratively Reweighted Least Squares)

        Алгоритм из лекции:
        1. w := (F^T F)^{-1} F^T y — нулевое приближение
        2. Для t = 1, 2, 3, ...:
           σ_i = σ(y_i w^T x_i)
           γ_i = √((1-σ_i)σ_i)
           F̃ = diag(γ_1,...,γ_ℓ) F
           ỹ_i = y_i √((1-σ_i)/σ_i)
           w := w + h_t (F^T F)^{-1} F^T ỹ

        В реализации ниже используется эквивалентная формулировка через
        взвешенные МНК на каждой итерации.
        """
        X_bias = np.c_[np.ones(X.shape[0]), X]  # матрица признаков F
        n_samples, n_features = X_bias.shape

        # Инициализация весов
        # w_0 = (X^T X)^{-1} X^T y (обычная линейная регрессия)
        self.w = np.linalg.lstsq(X_bias, y, rcond=None)[0]

        if verbose:
            print(f"\nНачало обучения IRLS:")
            print(f"    Размерность данных: {X_bias.shape}")
            print(f"    Параметр регуляризации λ: {self.reg_param}")

        for i in range(self.max_iter):
            # Вычисление вероятностей
            z = X_bias @ self.w
            p = self.sigmoid(z)     # σ_i = σ(w^T x_i)

            # Диагональная весовая матрица W
            # W_ii = p_i(1-p_i) = (1-σ_i)σ_i
            W_diag = p * (1 - p)

            # working response
            # z_working = z + (y - p)/W_diag
            epsilon = 1e-12
            z_working = z + (y - p) / (W_diag + epsilon)

            # Взвешенная матрица признаков (F̃ = diag(γ_i) F)
            W_sqrt = np.sqrt(W_diag)    # γ_i = √(W_ii)
            X_weighted = X_bias * W_sqrt[:, np.newaxis]     # F̃

            # Взвешенный отклик
            y_weighted = z_working * W_sqrt     # ỹ

            # Регуляризационная матрица
            reg_matrix = np.eye(n_features) * self.reg_param
            reg_matrix[0, 0] = 0

            # Решение взвешенной линейной регрессии
            # (F̃^T F̃ + λI) w = F̃^T ỹ
            XTWX = X_weighted.T @ X_weighted + reg_matrix
            XTWy = X_weighted.T @ y_weighted

            try:
                w_new = np.linalg.solve(XTWX, XTWy)
            except np.linalg.LinAlgError:
                w_new = np.linalg.lstsq(XTWX, XTWy, rcond=None)[0]

            # Проверка изменения весов
            weight_change = np.linalg.norm(w_new - self.w)
            self.w = w_new

            # Сохранение функции потерь
            loss = self.compute_loss(X_bias, y)
            self.loss_history.append(loss)

            # Проверка критерия остановки
            if weight_change < self.tol:
                if verbose:
                    print(f"    Сходимость на итерации {i + 1}, loss: {loss:.6f}")
                break

        if verbose and i == self.max_iter - 1:
            print(f"    Достигнут лимит итераций ({self.max_iter}), loss: {loss:.6f}")

        return self

    def predict_proba(self, X):
        """Предсказание вероятностей"""
        X_bias = np.c_[np.ones(X.shape[0]), X]
        return self.sigmoid(X_bias @ self.w)

    def predict(self, X, threshold=0.5):
        """Предсказание классов"""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_params(self):
        """Получение параметров модели"""
        return {
            'weights': self.w,
            'bias': self.w[0],
            'n_iterations': len(self.loss_history),
            'final_loss': self.loss_history[-1] if self.loss_history else None
        }


print("\n" + "=" * 70)
print("ОБУЧЕНИЕ МОДЕЛЕЙ")
print("=" * 70)

# Измерение времени выполнения
timing_results = {}

print("\n1. Обучение модели Newton-Raphson:")
start_time = time.time()
lr_newton = MyLogisticRegression(max_iter=50, tol=1e-6, reg_param=0.1, alpha=0.5)
lr_newton.fit(X_train_scaled, y_train, verbose=True)
newton_time = time.time() - start_time
timing_results['Newton-Raphson'] = newton_time
print(f"    Время обучения: {newton_time:.3f} сек")

print("\n2. Обучение модели IRLS:")
start_time = time.time()
lr_irls = LogisticRegressionIRLS(max_iter=50, tol=1e-6, reg_param=0.1)
lr_irls.fit(X_train_scaled, y_train, verbose=True)
irls_time = time.time() - start_time
timing_results['IRLS'] = irls_time
print(f"    Время обучения: {irls_time:.3f} сек")

print("\n3. Обучение модели Scikit-learn:")
start_time = time.time()
lr_sklearn = SklearnLogisticRegression(penalty='l2', C=10.0, max_iter=1000, solver='lbfgs', random_state=42, fit_intercept=True)
lr_sklearn.fit(X_train_scaled, y_train)
sklearn_time = time.time() - start_time
timing_results['Scikit-learn'] = sklearn_time
print(f"    Время обучения: {sklearn_time:.3f} сек")

# Предсказания
y_pred_newton = lr_newton.predict(X_test_scaled)
y_pred_irls = lr_irls.predict(X_test_scaled)
y_pred_sklearn = lr_sklearn.predict(X_test_scaled)

# Вероятности
prob_newton = lr_newton.predict_proba(X_test_scaled)
prob_irls = lr_irls.predict_proba(X_test_scaled)
prob_sklearn = lr_sklearn.predict_proba(X_test_scaled)[:, 1]

print("\n" + "=" * 70)
print("СРАВНЕНИЕ ПАРАМЕТРОВ МОДЕЛЕЙ")
print("=" * 70)

newton_params = lr_newton.get_params()
irls_params = lr_irls.get_params()

print(f"\nNewton-Raphson параметры:")
print(f"    Bias (intercept): {newton_params['bias']:.6f}")
print(f"    Количество итераций: {newton_params['n_iterations']}")
print(f"    Финальная функция потерь: {newton_params['final_loss']:.6f}")
print(f"    Норма весов: {np.linalg.norm(newton_params['weights']):.4f}")

print(f"\nIRLS параметры:")
print(f"    Bias (intercept): {irls_params['bias']:.6f}")
print(f"    Количество итераций: {irls_params['n_iterations']}")
print(f"    Финальная функция потерь: {irls_params['final_loss']:.6f}")
print(f"    Норма весов: {np.linalg.norm(irls_params['weights']):.4f}")

print(f"\nScikit-learn параметры:")
print(f"    Bias (intercept): {lr_sklearn.intercept_[0]:.6f}")
sklearn_weights_full = np.concatenate([[lr_sklearn.intercept_[0]], lr_sklearn.coef_[0]])
print(f"    Норма весов: {np.linalg.norm(sklearn_weights_full):.4f}")


print(f"\nСравнение разности весов между моделями:")
print("-" * 60)

# Вычисляем разность весов
diff_newton_irls = np.mean(np.abs(newton_params['weights'][1:] - irls_params['weights'][1:]))
diff_newton_sklearn = np.mean(np.abs(newton_params['weights'][1:] - lr_sklearn.coef_[0]))
diff_irls_sklearn = np.mean(np.abs(irls_params['weights'][1:] - lr_sklearn.coef_[0]))

print(f"Средняя абсолютная разность весов:")
print(f"    Newton-Raphson vs IRLS: {diff_newton_irls:.6f}")
print(f"    Newton-Raphson vs Scikit-learn: {diff_newton_sklearn:.6f}")
print(f"    IRLS vs Scikit-learn: {diff_irls_sklearn:.6f}")

print(f"\nМаксимальная абсолютная разность весов:")
print(f"    Newton-Raphson vs IRLS: {np.max(np.abs(newton_params['weights'][1:] - irls_params['weights'][1:])):.6f}")
print(f"    Newton-Raphson vs Scikit-learn: {np.max(np.abs(newton_params['weights'][1:] - lr_sklearn.coef_[0])):.6f}")
print(f"    IRLS vs Scikit-learn: {np.max(np.abs(irls_params['weights'][1:] - lr_sklearn.coef_[0])):.6f}")

print(f"\nСравнение первых 5 весов (включая bias):")
print(f"{'Признак':<25} {'Newton':<12} {'IRLS':<12} {'Scikit-learn':<12}")
print("-" * 65)
print(f"{'Bias':<25} {newton_params['bias']:<12.6f} {irls_params['bias']:<12.6f} {lr_sklearn.intercept_[0]:<12.6f}")

for i in range(4):
    feat_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
    print(
        f"{feat_name:<25} {newton_params['weights'][i + 1]:<12.6f} {irls_params['weights'][i + 1]:<12.6f} {lr_sklearn.coef_[0][i]:<12.6f}")


print(f"\nТоп-10 важных признаков по абсолютному значению весов:")

newton_weights = newton_params['weights'][1:]
newton_importance = np.abs(newton_weights)
newton_top_idx = np.argsort(newton_importance)[-10:][::-1]
print(f"\nNewton-Raphson:")
for i, idx in enumerate(newton_top_idx):
    sign = '+' if newton_weights[idx] > 0 else '-'
    print(f"  {i + 1:2d}. {feature_names[idx]:<25} {sign} {abs(newton_weights[idx]):.4f}")


sklearn_importance = np.abs(lr_sklearn.coef_[0])
sklearn_top_idx = np.argsort(sklearn_importance)[-10:][::-1]
print(f"\nScikit-learn:")
for i, idx in enumerate(sklearn_top_idx):
    sign = '+' if lr_sklearn.coef_[0][idx] > 0 else '-'
    print(f"  {i + 1:2d}. {feature_names[idx]:<25} {sign} {abs(lr_sklearn.coef_[0][idx]):.4f}")


def evaluate_model(y_true, y_pred, y_prob, model_name):
    """Оценка качества модели"""
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"\n{model_name}:")
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  Precision:          {precision:.4f}")
    print(f"  Recall (Sensitivity): {sensitivity:.4f}")
    print(f"  Specificity:        {specificity:.4f}")
    print(f"  F1-Score:           {f1:.4f}")
    print(f"  ROC-AUC:            {roc_auc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={tn:2d}  FP={fp:2d}")
    print(f"    FN={fn:2d}  TP={tp:2d}")

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr
    }


print("\n" + "=" * 70)
print("ОЦЕНКА КАЧЕСТВА КЛАССИФИКАЦИИ")
print("=" * 70)

# Оценка всех моделей
results = {}
results['Newton-Raphson'] = evaluate_model(y_test, y_pred_newton, prob_newton, "Newton-Raphson")
results['IRLS'] = evaluate_model(y_test, y_pred_irls, prob_irls, "IRLS")
results['Scikit-learn'] = evaluate_model(y_test, y_pred_sklearn, prob_sklearn, "Scikit-learn")

print("\n" + "=" * 70)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 70)

summary_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results],
    'Precision': [results[m]['precision'] for m in results],
    'Recall': [results[m]['recall'] for m in results],
    'Specificity': [results[m]['specificity'] for m in results],
    'F1-Score': [results[m]['f1_score'] for m in results],  # Исправлено
    'ROC-AUC': [results[m]['roc_auc'] for m in results],
    'Time (s)': [timing_results[m] for m in results]
}).set_index('Model')

print(summary_df.round(4))


fig = plt.figure(figsize=(18, 12))

# 1. График сходимости методов
ax1 = plt.subplot(2, 3, 1)
ax1.plot(lr_newton.loss_history, 'b-', linewidth=2, label='Newton-Raphson', marker='o', markersize=4)
ax1.plot(lr_irls.loss_history, 'g-', linewidth=2, label='IRLS', marker='s', markersize=4)
ax1.set_xlabel('Итерация')
ax1.set_ylabel('Log-Loss')
ax1.set_title('Сходимость методов оптимизации', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# 2. ROC-кривые
ax2 = plt.subplot(2, 3, 2)
colors = {'Newton-Raphson': 'blue', 'IRLS': 'green', 'Scikit-learn': 'red'}
for name, result in results.items():
    ax2.plot(result['fpr'], result['tpr'], color=colors[name], lw=2, label=f'{name} (AUC = {result["roc_auc"]:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC-кривые', fontsize=12, fontweight='bold')
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3)

# 3. Важность признаков
ax3 = plt.subplot(2, 3, 3)
top_n = 10
idx_top_newton = np.argsort(np.abs(newton_weights))[-top_n:][::-1]
features_top = [feature_names[i] for i in idx_top_newton]
weights_newton_top = newton_weights[idx_top_newton]
weights_sklearn_top = lr_sklearn.coef_[0][idx_top_newton]
x = np.arange(top_n)
width = 0.35
ax3.bar(x - width / 2, weights_newton_top, width, label='Newton-Raphson', color='blue', alpha=0.7)
ax3.bar(x + width / 2, weights_sklearn_top, width, label='Scikit-learn', color='red', alpha=0.7)
ax3.set_xlabel('Признаки')
ax3.set_ylabel('Веса')
ax3.set_title(f'Топ-{top_n} важных признаков', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(features_top, rotation=45, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. Сравнение точности
ax4 = plt.subplot(2, 3, 4)
models = list(results.keys())
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x_pos = np.arange(len(models))
bar_width = 0.2

metric_mapping = {
    'Accuracy': 'accuracy',
    'Precision': 'precision',
    'Recall': 'recall',
    'F1-Score': 'f1_score'
}

for i, metric in enumerate(metrics):
    metric_key = metric_mapping[metric]
    values = [results[m][metric_key] for m in models]
    ax4.bar(x_pos + i * bar_width, values, bar_width, label=metric)

ax4.set_xlabel('Модель')
ax4.set_ylabel('Значение')
ax4.set_title('Сравнение метрик качества', fontsize=12, fontweight='bold')
ax4.set_xticks(x_pos + bar_width * 1.5)
ax4.set_xticklabels(models)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# 5. Время выполнения
ax5 = plt.subplot(2, 3, 5)
times = [timing_results[m] for m in models]
colors_time = ['blue', 'green', 'red']
bars = ax5.bar(models, times, color=colors_time, alpha=0.7)
ax5.set_ylabel('Время (секунды)')
ax5.set_title('Время обучения моделей', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

for bar, t in zip(bars, times):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
             f'{t:.3f}s', ha='center', va='bottom')

# 6. Матрицы ошибок
ax6 = plt.subplot(2, 3, 6)
cm_sklearn = results['Scikit-learn']['confusion_matrix']
sns.heatmap(cm_sklearn, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Количество'}, ax=ax6)
ax6.set_xlabel('Предсказанный класс')
ax6.set_ylabel('Истинный класс')
ax6.set_title('Матрица ошибок (Scikit-learn)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '2_main_results.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("СТАТИСТИЧЕСКИЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
print("=" * 70)

print("\n1. Сравнение предсказанных вероятностей:")
print("-" * 65)

# Тест на равенство средних (парный t-тест)
t_stat_newton_sklearn, p_val_newton_sklearn = ttest_rel(prob_newton, prob_sklearn)
t_stat_irls_sklearn, p_val_irls_sklearn = ttest_rel(prob_irls, prob_sklearn)
t_stat_newton_irls, p_val_newton_irls = ttest_rel(prob_newton, prob_irls)

print(f"\nПарный t-тест:")
print(f"  Newton-Raphson vs Scikit-learn:")
print(f"    t-статистика: {t_stat_newton_sklearn:.6f}, p-значение: {p_val_newton_sklearn:.6f}")
print(f"    Средняя разность: {np.mean(prob_newton - prob_sklearn):.6f}")
print(f"    Стандартная ошибка: {np.std(prob_newton - prob_sklearn, ddof=1) / np.sqrt(len(prob_newton)):.6f}")

print(f"\n  IRLS vs Scikit-learn:")
print(f"    t-статистика: {t_stat_irls_sklearn:.6f}, p-значение: {p_val_irls_sklearn:.6f}")
print(f"    Средняя разность: {np.mean(prob_irls - prob_sklearn):.6f}")
print(f"    Стандартная ошибка: {np.std(prob_irls - prob_sklearn, ddof=1) / np.sqrt(len(prob_irls)):.6f}")

print(f"\n  Newton-Raphson vs IRLS:")
print(f"    t-статистика: {t_stat_newton_irls:.6f}, p-значение: {p_val_newton_irls:.6f}")
print(f"    Средняя разность: {np.mean(prob_newton - prob_irls):.6f}")

# Непараметрический тест Вилкоксона
print(f"\nТест Вилкоксона (непараметрический):")
w_stat_newton_sklearn, w_p_newton_sklearn = wilcoxon(prob_newton - prob_sklearn)
w_stat_irls_sklearn, w_p_irls_sklearn = wilcoxon(prob_irls - prob_sklearn)

print(f"  Newton-Raphson vs Scikit-learn")
print(f"    W-статистика: {w_stat_newton_sklearn:.2f}, p-значение: {w_p_newton_sklearn:.6f}")

print(f"\n  IRLS vs Scikit-learn:")
print(f"    W-статистика: {w_stat_irls_sklearn:.2f}, p-значение: {w_p_irls_sklearn:.6f}")

print(f"\n2. Корреляционный анализ:")
print("-" * 65)

corr_newton_sklearn = np.corrcoef(prob_newton, prob_sklearn)[0, 1]
corr_irls_sklearn = np.corrcoef(prob_irls, prob_sklearn)[0, 1]
corr_newton_irls = np.corrcoef(prob_newton, prob_irls)[0, 1]

print(f"    Корреляция Пирсона:")
print(f"        Newton-Raphson и Scikit-learn: {corr_newton_sklearn:.6f}")
print(f"        IRLS и Scikit-learn: {corr_irls_sklearn:.6f}")
print(f"        Newton-Raphson и IRLS: {corr_newton_irls:.6f}")

# Анализ согласованности классификаций
print(f"\n3. Анализ согласованности классификаций:")
print("-" * 65)

agree_newton_sklearn = np.mean(y_pred_newton == y_pred_sklearn)
agree_irls_sklearn = np.mean(y_pred_irls == y_pred_sklearn)
agree_newton_irls = np.mean(y_pred_newton == y_pred_irls)

print(f"    Процент совпадения предсказаний:")
print(f"        Newton-Raphson и Scikit-learn: {agree_newton_sklearn:.4f} ({agree_newton_sklearn * 100:.1f}%)")
print(f"        IRLS и Scikit-learn: {agree_irls_sklearn:.4f} ({agree_irls_sklearn * 100:.1f}%)")
print(f"        Newton-Raphson и IRLS: {agree_newton_irls:.4f} ({agree_newton_irls * 100:.1f}%)")

print(f"\n4. Анализ расхождений:")
print("-" * 65)
diff_newton_sklearn = y_pred_newton != y_pred_sklearn
diff_indices = np.where(diff_newton_sklearn)[0]

if len(diff_indices) > 0:
    print(f"Расхождения между Newton-Raphson и Scikit-learn: {len(diff_indices)}")
    print("Подробности расхождений:")
    for idx in diff_indices[:5]:
        print(f"    Объект {idx}: true={y_test[idx]}, "
              f"NR={y_pred_newton[idx]}({prob_newton[idx]:.3f}), "
              f"SK={y_pred_sklearn[idx]}({prob_sklearn[idx]:.3f})")
else:
    print(" Расхождений между Newton-Raphson и Scikit-learn не обнаружено")

print("\n" + "=" * 70)
print("АНАЛИЗ ВЛИЯНИЯ РЕГУЛЯРИЗАЦИИ")
print("=" * 70)

reg_params = [0.001, 0.01, 0.1, 1.0, 10.0]
reg_results = []

print(f"\n{'λ':>8} {'1/λ (C)':>10} {'Newton Acc':>12} {'IRLS Acc':>12} {'Sklearn Acc':>12}")
print("-" * 60)

for reg in reg_params:
    # Newton-Raphson
    lr_nr = MyLogisticRegression(max_iter=50, reg_param=reg, alpha=0.5)
    lr_nr.fit(X_train_scaled, y_train)
    y_pred_nr = lr_nr.predict(X_test_scaled)
    acc_nr = accuracy_score(y_test, y_pred_nr)

    # IRLS
    lr_i = LogisticRegressionIRLS(max_iter=50, reg_param=reg)
    lr_i.fit(X_train_scaled, y_train)
    y_pred_i = lr_i.predict(X_test_scaled)
    acc_i = accuracy_score(y_test, y_pred_i)

    # Scikit-learn
    lr_s = SklearnLogisticRegression(C=1 / reg, max_iter=1000, penalty='l2', solver='lbfgs', random_state=42)
    lr_s.fit(X_train_scaled, y_train)
    y_pred_s = lr_s.predict(X_test_scaled)
    acc_s = accuracy_score(y_test, y_pred_s)

    reg_results.append({
        'lambda': reg,
        'C': 1 / reg,
        'newton_acc': acc_nr,
        'irls_acc': acc_i,
        'sklearn_acc': acc_s
    })

    print(f"{reg:>8.3f} {1 / reg:>10.1f} {acc_nr:>12.4f} {acc_i:>12.4f} {acc_s:>12.4f}")

# Визуализация влияния регуляризации
fig3, ax3 = plt.subplots(1, 2, figsize=(12, 5))

# Точность в зависимости от λ
lambdas = [r['lambda'] for r in reg_results]
ax3[0].plot(lambdas, [r['newton_acc'] for r in reg_results], 'bo-', label='Newton-Raphson', linewidth=2)
ax3[0].plot(lambdas, [r['irls_acc'] for r in reg_results], 'gs-', label='IRLS', linewidth=2)
ax3[0].plot(lambdas, [r['sklearn_acc'] for r in reg_results], 'r^-', label='Scikit-learn', linewidth=2)
ax3[0].set_xscale('log')
ax3[0].set_xlabel('λ (параметр регуляризации)')
ax3[0].set_ylabel('Accuracy')
ax3[0].set_title('Влияние регуляризации на точность', fontsize=12, fontweight='bold')
ax3[0].legend()
ax3[0].grid(True, alpha=0.3)

# Норма весов в зависимости от λ
norm_newton = []
norm_sklearn = []

for reg in reg_params:
    # Newton
    lr_temp = MyLogisticRegression(max_iter=50, reg_param=reg)
    lr_temp.fit(X_train_scaled, y_train)
    norm_newton.append(np.linalg.norm(lr_temp.w))

    # Scikit-learn
    lr_temp_sk = SklearnLogisticRegression(C=1 / reg, max_iter=1000, penalty='l2')
    lr_temp_sk.fit(X_train_scaled, y_train)
    norm_sklearn.append(np.linalg.norm(np.concatenate([[lr_temp_sk.intercept_[0]], lr_temp_sk.coef_[0]])))

ax3[1].plot(lambdas, norm_newton, 'bo-', label='Newton-Raphson', linewidth=2)
ax3[1].plot(lambdas, norm_sklearn, 'r^-', label='Scikit-learn', linewidth=2)
ax3[1].set_xscale('log')
ax3[1].set_xlabel('λ')
ax3[1].set_ylabel('Норма весов')
ax3[1].set_title('Влияние регуляризации на норму весов', fontsize=12, fontweight='bold')
ax3[1].legend()
ax3[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '3_regularization_analysis.png'), dpi=300, bbox_inches='tight')
plt.show()
