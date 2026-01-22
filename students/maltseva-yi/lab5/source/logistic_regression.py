import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_curve, roc_auc_score)
import time
import seaborn as sns

class LogisticRegressionNR:
    # Логистическая регрессия с оптимизацией методом Ньютона-Рафсона c добавлением регуляризации L2
    
    def __init__(self, C=1.0, max_iter=100, tol=1e-4, verbose=False):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.w = None
        self.loss_history = []
        
    def sigmoid(self, z):
        # Для предотвращения переполнения
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def log_loss(self, X, y, w):
        z = X @ w
        # Используем стабильную версию
        loss = np.mean(np.log(1 + np.exp(-y * z)))
        
        if self.C > 0:
            reg_loss = (1 / (2 * self.C)) * np.sum(w[1:]**2)
            loss += reg_loss / X.shape[0]
            
        return loss
    
    def fit(self, X, y):

        n_samples, n_features = X.shape
        
        # Добавляем свободный член
        X_ext = np.hstack([np.ones((n_samples, 1)), X])
        
        self.w = np.zeros(n_features + 1)

        y_bipolar = 2 * y - 1
        
        for iter in range(self.max_iter):

            z = X_ext @ self.w
            sigma = self.sigmoid(y_bipolar * z)
            
            grad = -X_ext.T @ ((1 - sigma) * y_bipolar) / n_samples
            
            # Добавляем регуляризацию к градиенту (кроме intercept)
            if self.C > 0:
                reg_grad = np.zeros_like(self.w)
                reg_grad[1:] = (1 / self.C) * self.w[1:] / n_samples
                grad += reg_grad
            
            # Вычисляем гессиан
            weights = (1 - sigma) * sigma
            H = (X_ext.T * weights) @ X_ext / n_samples
            
            # Добавляем регуляризацию к гессиану (кроме intercept)
            if self.C > 0:
                H_reg = np.eye(H.shape[0])
                H_reg[0, 0] = 0
                H += (1 / self.C) * H_reg / n_samples
            
            H += 1e-8 * np.eye(H.shape[0])
            
            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(H) @ grad
            
            self.w -= delta
            
            # Вычисляем потерю
            current_loss = self.log_loss(X_ext, y_bipolar, self.w)
            self.loss_history.append(current_loss)
            
            if np.linalg.norm(delta) < self.tol:
                break
                
            if self.verbose and iter % 10 == 0:
                print(f"Iter {iter + 1}, Loss: {current_loss:.6f}")
        
        return self
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        X_ext = np.hstack([np.ones((n_samples, 1)), X])
        probabilities = self.sigmoid(X_ext @ self.w)
        return np.column_stack([1 - probabilities, probabilities])
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


class LogisticRegressionIRLS:
    # Логистическая регрессия с оптимизацией методом IRLS c добавлением регуляризации L2
    
    def __init__(self, C=1.0, max_iter=100, tol=1e-4, verbose=False):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.w = None
        self.loss_history = []
        
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def fit(self, X, y):

        n_samples, n_features = X.shape

        X_ext = np.hstack([np.ones((n_samples, 1)), X])
        
        self.w = np.zeros(n_features + 1)
        
        for iter in range(self.max_iter):
            # Вычисляем линейную комбинацию
            z = X_ext @ self.w
            
            # Вычисляем вероятности
            p = self.sigmoid(z)
            
            # Вычисляем веса объектов
            weights = p * (1 - p)
            weights = np.maximum(weights, 1e-6)

            working_response = z + (y - p) / weights

            W = np.diag(weights)
            
            XTW = X_ext.T @ W
            XTWX = XTW @ X_ext
            
            if self.C > 0:
                reg_matrix = np.eye(XTWX.shape[0])
                reg_matrix[0, 0] = 0  # Не регуляризуем intercept
                XTWX += (1 / self.C) * reg_matrix
            
            XTWX += 1e-8 * np.eye(XTWX.shape[0])
            
            try:
                w_new = np.linalg.solve(XTWX, XTW @ working_response)
            except np.linalg.LinAlgError:
                w_new = np.linalg.pinv(XTWX) @ (XTW @ working_response)
            
            # Проверка сходимости
            delta = np.linalg.norm(w_new - self.w)
            self.w = w_new

            epsilon = 1e-15
            p_clipped = np.clip(p, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
            
            # Добавляем регуляризацию к потере
            if self.C > 0:
                reg_loss = (1 / (2 * self.C)) * np.sum(self.w[1:]**2)
                loss += reg_loss / n_samples
                
            self.loss_history.append(loss)
            
            if delta < self.tol:
                break
                
            if self.verbose and iter % 10 == 0:
                print(f"Iter {iter + 1}, Loss: {loss:.6f}, Delta: {delta:.6f}")
        
        return self
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        X_ext = np.hstack([np.ones((n_samples, 1)), X])
        probabilities = self.sigmoid(X_ext @ self.w)
        return np.column_stack([1 - probabilities, probabilities])
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


def load_and_prepare_data():

    print("=" * 70)
    print("ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    print("=" * 70)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    print(f"Исходная размерность данных: {X.shape}")
    print(f"Количество классов: {len(np.unique(y))}")
    print(f"Баланс классов: {np.bincount(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nОбучающая выборка: {X_train_scaled.shape}")
    print(f"Тестовая выборка: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, data


def compare_optimization_methods(X_train, y_train):

    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ МЕТОДОВ ОПТИМИЗАЦИИ (C=1.0)")
    print("=" * 70)
    
    # Обучаем модель методом Ньютона-Рафсона
    print("\n1. Обучение методом Ньютона-Рафсона...")
    start_nr = time.time()
    model_nr = LogisticRegressionNR(C=1.0, max_iter=50, tol=1e-6, verbose=True)
    model_nr.fit(X_train, y_train)
    time_nr = time.time() - start_nr
    
    print(f"Время обучения: {time_nr:.4f} сек")
    print(f"Количество итераций: {len(model_nr.loss_history)}")
    print(f"Финальная потеря: {model_nr.loss_history[-1]:.6f}")
    
    # Обучаем модель методом IRLS
    print("\n2. Обучение методом IRLS...")
    start_irls = time.time()
    model_irls = LogisticRegressionIRLS(C=1.0, max_iter=50, tol=1e-6, verbose=True)
    model_irls.fit(X_train, y_train)
    time_irls = time.time() - start_irls
    
    print(f"Время обучения: {time_irls:.4f} сек")
    print(f"Количество итераций: {len(model_irls.loss_history)}")
    print(f"Финальная потеря: {model_irls.loss_history[-1]:.6f}")
    
    # Визуализация сходимости
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # График 1: Сравнение лоссов
    axes[0].plot(model_nr.loss_history, 'b-', linewidth=2, label='Ньютон-Рафсон')
    axes[0].plot(model_irls.loss_history, 'r--', linewidth=2, label='IRLS')
    axes[0].set_xlabel('Итерация')
    axes[0].set_ylabel('Логарифмическая потеря')
    axes[0].set_title('Сравнение сходимости')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # График 2: Логарифмическая шкала
    axes[1].semilogy(model_nr.loss_history, 'b-', linewidth=2, label='Ньютон-Рафсон')
    axes[1].semilogy(model_irls.loss_history, 'r--', linewidth=2, label='IRLS')
    axes[1].set_xlabel('Итерация')
    axes[1].set_ylabel('Лог-потеря (лог шкала)')
    axes[1].set_title('Сходимость в логарифмической шкале')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/optimization_comparison_fixed.png')
    plt.show()
    
    return model_nr, model_irls, time_nr, time_irls


def evaluate_model(model, X_test, y_test, model_name):

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_proba': y_proba
    }


def compare_with_sklearn(X_train, X_test, y_train, y_test):

    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ С ЭТАЛОННОЙ РЕАЛИЗАЦИЕЙ (SKLEARN, C=1.0)")
    print("=" * 70)
    
    # Обучаем sklearn модель
    print("\n3. Обучение sklearn модели...")
    start_sklearn = time.time()
    sklearn_model = SklearnLogisticRegression(
        C=1.0,
        penalty='l2',
        max_iter=1000,
        tol=1e-6,
        solver='lbfgs',
        random_state=42
    )
    sklearn_model.fit(X_train, y_train)
    time_sklearn = time.time() - start_sklearn
    print(f"Время обучения: {time_sklearn:.4f} сек")
    
    # Обучаем наши модели
    model_nr = LogisticRegressionNR(C=1.0, max_iter=50, tol=1e-6, verbose=False)
    model_nr.fit(X_train, y_train)
    
    model_irls = LogisticRegressionIRLS(C=1.0, max_iter=50, tol=1e-6, verbose=False)
    model_irls.fit(X_train, y_train)
 
    results = []
    results.append(evaluate_model(model_nr, X_test, y_test, "Custom NR"))
    results.append(evaluate_model(model_irls, X_test, y_test, "Custom IRLS"))
    results.append(evaluate_model(sklearn_model, X_test, y_test, "Sklearn"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("=" * 70)
    print(f"{'Метод':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 70)
    
    for res in results:
        print(f"{res['model']:<15} {res['accuracy']:<10.4f} {res['precision']:<10.4f} "
              f"{res['recall']:<10.4f} {res['f1']:<10.4f} {res['auc']:<10.4f}")
    
    # ROC-кривые
    colors = ['blue', 'red', 'green']
    for idx, res in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        axes[0].plot(fpr, tpr, color=colors[idx], lw=2, 
                    label=f"{res['model']} (AUC = {res['auc']:.3f})")
    
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC-кривые')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Матрица ошибок для Custom NR
    sns.heatmap(results[0]['confusion_matrix'], annot=True, fmt='d', 
                cmap='Blues', ax=axes[1], cbar=False)
    axes[1].set_title('Confusion Matrix\nCustom NR')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    # Матрица ошибок для Sklearn
    sns.heatmap(results[2]['confusion_matrix'], annot=True, fmt='d', 
                cmap='Blues', ax=axes[2], cbar=False)
    axes[2].set_title('Confusion Matrix\nSklearn')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('images/model_comparison_fixed.png')
    plt.show()
    
    return results


def analyze_coefficients(X_train, y_train, data):

    print("\n" + "=" * 70)
    print("АНАЛИЗ КОЭФФИЦИЕНТОВ (C=1.0)")
    print("=" * 70)
    
    # Обучаем модель
    model = LogisticRegressionIRLS(C=1.0, max_iter=100, tol=1e-6)
    model.fit(X_train, y_train)

    coefficients = model.w[1:]
    intercept = model.w[0]

    feature_names = data.feature_names
    sorted_indices = np.argsort(np.abs(coefficients))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # График 1: Коэффициенты
    top_n = 10
    top_indices = sorted_indices[:top_n]
    
    bars = axes[0].barh(range(top_n), coefficients[top_indices])
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels([feature_names[i] for i in top_indices], fontsize=9)
    axes[0].set_xlabel('Значение коэффициента')
    axes[0].set_title(f'Топ-{top_n} наиболее важных признаков')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # График 2: Абсолютные значения коэффициентов
    abs_coefficients = np.abs(coefficients[top_indices])
    axes[1].barh(range(top_n), abs_coefficients)
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels([feature_names[i] for i in top_indices], fontsize=9)
    axes[1].set_xlabel('Абсолютное значение коэффициента')
    axes[1].set_title('Абсолютная важность признаков')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('images/coefficient_analysis_fixed.png')
    plt.show()
    
    # Вывод интерпретации
    print(f"\nIntercept (свободный член): {intercept:.4f}")
    print("\nТоп-5 наиболее важных признаков:")
    for i in range(min(5, len(top_indices))):
        idx = top_indices[i]
        coef = coefficients[idx]
        feature = feature_names[idx]
        
        if coef > 0:
            effect = "увеличивает"
        else:
            effect = "уменьшает"
            
        print(f"  {feature}:")
        print(f"    Коэффициент: {coef:.4f}")
        print(f"    Интерпретация: увеличение признака на 1 ст. отклонение {effect} вероятность положительного класса")
        print()


def main():
    
    X_train, X_test, y_train, y_test, data = load_and_prepare_data()

    model_nr, model_irls, time_nr, time_irls = compare_optimization_methods(X_train, y_train)

    results = compare_with_sklearn(X_train, X_test, y_train, y_test)

    analyze_coefficients(X_train, y_train, data)
    
    return {
        'model_nr': model_nr,
        'model_irls': model_irls,
        'results': results,
        'data': data
    }


if __name__ == "__main__":
    results = main()