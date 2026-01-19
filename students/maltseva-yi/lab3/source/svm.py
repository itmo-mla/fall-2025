import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import scipy.optimize as opt
import time

class MySVM:
    # Реализация SVM через двойственную задачу
    
    def __init__(self, C=1.0, kernel='linear', degree=3):
        self.C = C
        self.kernel_type = kernel
        self.degree = degree
        self.lambdas = None
        self.support_vectors = None
        self.support_labels = None
        self.X_train = None
        
    def _kernel(self, X1, X2):
        # Вычисление ядра между матрицами X1 и X2
        if self.kernel_type == 'linear':
            return X1 @ X2.T
        elif self.kernel_type == 'poly':
            return (X1 @ X2.T + 1) ** self.degree
        elif self.kernel_type == 'rbf':
            # Автоматический подбор gamma
            gamma = 1.0 / (X1.shape[1] * X1.var())
            norm1 = np.sum(X1**2, axis=1).reshape(-1, 1)
            norm2 = np.sum(X2**2, axis=1).reshape(1, -1)
            K = norm1 + norm2 - 2 * X1 @ X2.T
            return np.exp(-gamma * K)
    
    def fit(self, X, y):
        # Обучение модели - решение двойственной задачи
        self.X_train = X.copy()
        y_binary = np.where(y == 0, -1, 1)
        n = X.shape[0]
        
        # Матрица ядра
        K = self._kernel(X, X)
        
        # Целевая функция L(λ)
        def objective(lambdas):
            term1 = np.sum(lambdas)
            term2 = 0.5 * np.sum(np.outer(lambdas * y_binary, lambdas * y_binary) * K)
            return -term1 + term2
        
        # Ограничения: Σ λ_i y_i = 0, 0 ≤ λ_i ≤ C
        constraints = {
            'type': 'eq',
            'fun': lambda l: np.dot(l, y_binary)
        }
        bounds = [(0, self.C) for _ in range(n)]
        
        lambdas0 = np.ones(n) * 0.1
        
        # Оптимизация
        result = opt.minimize(
            objective, lambdas0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-4}
        )
        
        self.lambdas = result.x
        
        # Выделяем опорные векторы (λ > 1e-5)
        sv_idx = self.lambdas > 1e-5
        self.support_vectors = X[sv_idx]
        self.support_labels = y_binary[sv_idx]

        # Смещение w0 через опорные векторы
        if self.kernel_type == 'linear':
            self.w = np.sum((self.lambdas[sv_idx] * self.support_labels).reshape(-1, 1) * 
                self.support_vectors, axis=0)
            # Ищем опорный вектор на границе зазора: 0 < λ < C
            sv_lambdas = self.lambdas[sv_idx]
            margin_sv = (sv_lambdas > 1e-5) & (sv_lambdas < self.C - 1e-5)
            if np.any(margin_sv):
                idx = np.where(margin_sv)[0][0]
                self.w0 = self.support_labels[idx] - np.dot(self.w, self.support_vectors[idx])
            else:
                self.w0 = np.mean(self.support_labels - np.dot(self.support_vectors, self.w))
        else:
            self.w0 = 0
        
        print(f"Обучено. Опорных векторов: {len(self.support_vectors)}")
    
    def predict(self, X):
        if self.kernel_type == 'linear' and hasattr(self, 'w'):
            scores = X @ self.w + self.w0
        else:
            # f(x) = Σ λ_i y_i K(x, x_i) + w0
            K_test = self._kernel(X, self.support_vectors)
            scores = np.dot(K_test, self.support_labels * self.lambdas[self.lambdas > 1e-5]) + self.w0
        
        return np.where(scores >= 0, 1, 0)

def demo_kernels():
    # Сравнение разных ядер
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    kernels = ['linear', 'poly', 'rbf']
    kernel_names = ['Линейное', 'Полином (d=3)', 'RBF']
    
    results = []
    
    for kernel, name in zip(kernels, kernel_names):
        print(f"\nЯдро: {name}")
        
        # Наша реализация
        start = time.time()
        our_svm = MySVM(C=1.0, kernel=kernel, degree=3)
        our_svm.fit(X_train_s, y_train)
        our_time = time.time() - start
        
        y_pred_our = our_svm.predict(X_test_s)
        acc_our = accuracy_score(y_test, y_pred_our)
        
        # Sklearn
        start = time.time()
        skl_svm = SVC(C=1.0, kernel=kernel, degree=3, gamma='scale')
        skl_svm.fit(X_train_s, y_train)
        skl_time = time.time() - start
        
        y_pred_skl = skl_svm.predict(X_test_s)
        acc_skl = accuracy_score(y_test, y_pred_skl)
        
        results.append({
            'kernel': name,
            'our_acc': acc_our,
            'skl_acc': acc_skl,
            'our_time': our_time,
            'skl_time': skl_time,
            'n_sv': len(our_svm.support_vectors)
        })
        
        print(f"  Наш:    acc={acc_our:.3f}, время={our_time:.2f}с, SV={len(our_svm.support_vectors)}")
        print(f"  Sklearn: acc={acc_skl:.3f}, время={skl_time:.2f}с, SV={len(skl_svm.support_vectors_)}")
    
    return results

def regularization_study():
    # Влияние параметра C
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    C_values = [0.01, 0.1, 1, 10, 100]
    train_accs = []
    test_accs = []
    n_svs = []
    
    for C in C_values:
        svm = MySVM(C=C, kernel='linear')
        svm.fit(X_train_s, y_train)
        
        y_pred_train = svm.predict(X_train_s)
        y_pred_test = svm.predict(X_test_s)
        
        train_accs.append(accuracy_score(y_train, y_pred_train))
        test_accs.append(accuracy_score(y_test, y_pred_test))
        n_svs.append(len(svm.support_vectors))
        
        print(f"C={C:6.2f}: train_acc={train_accs[-1]:.3f}, test_acc={test_accs[-1]:.3f}, SV={n_svs[-1]}")
    
    # График
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.semilogx(C_values, train_accs, 'bo-', label='Train')
    ax1.semilogx(C_values, test_accs, 'ro-', label='Test')
    ax1.set_xlabel('Параметр C')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Влияние C на качество')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.semilogx(C_values, n_svs, 'go-')
    ax2.set_xlabel('Параметр C')
    ax2.set_ylabel('Количество SV')
    ax2.set_title('Влияние C на число опорных векторов')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/svm_c_study.png', dpi=100)
    plt.show()
    
    return C_values, train_accs, test_accs, n_svs

def visualize_2d_svm():
    # Визуализация SVM в 2D (через PCA)
    from sklearn.decomposition import PCA
    
    X, y = load_breast_cancer(return_X_y=True)

    X_s = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_s)

    from sklearn.utils import resample
    X_small, y_small = resample(X_2d, y, n_samples=100, random_state=42, stratify=y)
    
    # Обучаем SVM
    svm = MySVM(C=1.0, kernel='linear')
    svm.fit(X_small, y_small)
    
    x_min, x_max = X_small[:, 0].min() - 1, X_small[:, 0].max() + 1
    y_min, y_max = X_small[:, 1].min() - 1, X_small[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                        np.linspace(y_min, y_max, 50))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu_r)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    scatter = plt.scatter(X_small[:, 0], X_small[:, 1], 
                         c=y_small, cmap=plt.cm.RdBu_r,
                         edgecolors='k', s=60)
    
    if svm.support_vectors is not None:
        plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1],
                   s=150, facecolors='none', 
                   edgecolors='yellow', linewidths=2,
                   label='Опорные векторы')
    
    plt.xlabel('Первая компонента (PCA)')
    plt.ylabel('Вторая компонента (PCA)')
    plt.title('SVM: Разделяющая гиперплоскость в 2D')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Класс')
    
    plt.tight_layout()
    plt.savefig('images/svm_2d_visualization.png', dpi=100)
    plt.show()

def main():

    # 1. Загружаем данные
    X, y = load_breast_cancer(return_X_y=True)
    print(f"\nДанные: {X.shape[0]} объектов, {X.shape[1]} признаков")
    print(f"Классы: {np.unique(y)} ({np.bincount(y)})")
    
    # 2. Делим на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Нормализуем
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 3. Базовый пример с линейным ядром
    print("\n1. Базовый пример (линейное ядро):")
    our_svm = MySVM(C=1.0, kernel='linear')
    our_svm.fit(X_train_s, y_train)
    
    y_pred = our_svm.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy на тесте: {acc:.3f}")
    print(f"   Опорных векторов: {len(our_svm.support_vectors)}")

    if hasattr(our_svm, 'w') and our_svm.w is not None:
        norm_w = np.linalg.norm(our_svm.w)
        margin = 2.0 / norm_w
        print(f"   Норма вектора w: ||w|| = {norm_w:.4f}")
        print(f"   Ширина зазора (margin): 2/||w|| = {margin:.4f}")
        print(f"   Смещение w0: {our_svm.w0:.4f}")
    
    # 4. Сравнение с sklearn
    print("\n2. Сравнение с sklearn (линейное ядро):")
    skl_svm = SVC(C=1.0, kernel='linear')
    skl_svm.fit(X_train_s, y_train)
    skl_acc = accuracy_score(y_test, skl_svm.predict(X_test_s))
    print(f"   Наш SVM:    {acc:.3f}")
    print(f"   Sklearn SVM: {skl_acc:.3f}")
    print(f"   Разница:     {abs(acc - skl_acc):.4f}")
    
    # 5. Визуализация в 2D
    print("\n3. Визуализация (2D PCA)...")
    visualize_2d_svm()
    
    # 6. Исследование регуляризации
    print("\n4. Исследование параметра C:")
    regularization_study()
    
    # 7. Сравнение ядер
    print("\n5. Сравнение разных ядер:")
    kernel_results = demo_kernels()
    
    # Сводная таблица
    print("\n" + "=" * 60)
    print("Итоговая таблица:")
    print("-" * 60)
    print(f"{'Ядро':<15} {'Наш acc':<10} {'Sklearn acc':<12} {'Наши SV':<10}")
    print("-" * 60)
    for res in kernel_results:
        print(f"{res['kernel']:<15} {res['our_acc']:<10.3f} {res['skl_acc']:<12.3f} {res['n_sv']:<10}")

if __name__ == "__main__":
    main()
    