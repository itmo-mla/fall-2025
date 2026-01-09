import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class MyPCA:
    # Реализация PCA через сингулярное разложение (SVD)

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.V_ = None
        self.U_ = None
        
    def fit(self, X):
        # Обучение PCA на данных X
  
        # 1. Центрирование данных
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 2. Сингулярное разложение
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # 3. Сохраняем результаты
        self.singular_values_ = S
        self.components_ = Vt
        self.V_ = U
        self.U_ = Vt.T
        
        # 4. Вычисляем объясненную дисперсию (собственные значения)
        n_samples = X.shape[0]
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        total_variance = np.sum(self.explained_variance_)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        # 5. Если задано n_components, обрезаем
        if self.n_components is not None:
            self.components_ = self.components_[:self.n_components]
            self.explained_variance_ = self.explained_variance_[:self.n_components]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]
            
        return self
    
    def transform(self, X):
        # Преобразование данных в новое пространство: G = F * U
        X_centered = X - self.mean_
        
        if self.n_components is not None:
            components = self.components_[:self.n_components]
        else:
            components = self.components_
            
        # G = X_centered * U (в лекции)
        return np.dot(X_centered, components.T)
    
    def inverse_transform(self, X_transformed):
        # Обратное преобразование: F̂ = G * U^T
        if self.n_components is not None:
            components = self.components_[:self.n_components]
        else:
            components = self.components_
            
        # F̂ = X_transformed * U^T + mean
        X_reconstructed = np.dot(X_transformed, components) + self.mean_
        return X_reconstructed
    
    def fit_transform(self, X):
        # Обучение и преобразование
        self.fit(X)
        return self.transform(X)
    
    def get_cumulative_variance(self):
        # Кумулятивная объясненная дисперсия
        return np.cumsum(self.explained_variance_ratio_)
    
    def reconstruction_error(self, X):
        # Вычисление ошибки реконструкции ||F - G*U^T||^2
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        return np.mean((X - X_reconstructed) ** 2)

def pca_for_regression_demo(X, y, n_components_list=[2, 5, 10]):
    # Демонстрация связи PCA с линейной регрессией

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    for n_components in n_components_list:
        # Применяем PCA
        pca = MyPCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        # Обучаем линейную регрессию на преобразованных данных
        lr = LinearRegression()
        lr.fit(X_train_pca, y_train)
        
        # Оценка качества
        train_score = lr.score(X_train_pca, y_train)
        test_score = lr.score(X_test_pca, y_test)
        
        # Ошибка реконструкции
        recon_error = pca.reconstruction_error(X_train_scaled)
        
        results[n_components] = {
            'train_score': train_score,
            'test_score': test_score,
            'recon_error': recon_error,
            'explained_variance': np.sum(pca.explained_variance_ratio_)
        }
    
    return results

def main():

    # 1. Загрузка датасета для линейной регрессии
    print("Загрузка датасета Diabetes для регрессии...")
    data = load_diabetes()
    X = data.data
    y = data.target
    
    print(f"Размерность исходных данных: {X.shape}")
    print(f"Целевая переменная: {y[:5]}...")
    
    # 2. Масштабирование данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Реализация PCA через SVD
    print("\n" + "="*60)
    print("Реализация PCA через сингулярное разложение")
    print("="*60)
    
    pca_manual = MyPCA()
    pca_manual.fit(X_scaled)
    
    print(f"Собственные значения (первые 5): {pca_manual.singular_values_[:5] ** 2 / (X.shape[0] - 1)}")
    print(f"Объясненная дисперсия (первые 5): {pca_manual.explained_variance_ratio_[:5]}")
    
    # 4. Определение эффективной размерности
    print("\n" + "="*60)
    print("Определение эффективной размерности выборки")
    print("="*60)
    
    cumulative_variance = pca_manual.get_cumulative_variance()
    
    # Пороговые значения
    thresholds = [0.8, 0.9, 0.95, 0.99]
    for threshold in thresholds:
        effective_dim = np.argmax(cumulative_variance >= threshold) + 1
        print(f"Для {threshold*100}% дисперсии нужно {effective_dim} компонент(ы)")
    
    # 5. Сравнение с эталонной реализацией sklearn
    print("\n" + "="*60)
    print("Сравнение с эталонной реализацией (sklearn)")
    print("="*60)
    
    n_components = 5
    pca_sklearn = SklearnPCA(n_components=n_components)
    X_sklearn = pca_sklearn.fit_transform(X_scaled)
    
    pca_manual = MyPCA(n_components=n_components)
    X_manual = pca_manual.fit_transform(X_scaled)
    
    # Проверка эквивалентности
    print("Объясненная дисперсия:")
    print(f"  Ручная:    {pca_manual.explained_variance_ratio_}")
    print(f"  Sklearn:   {pca_sklearn.explained_variance_ratio_}")
    
    # Корреляция между компонентами (должна быть ±1)
    corr_matrix = np.corrcoef(X_manual.T, X_sklearn.T)
    print(f"\nМаксимальная корреляция между компонентами: {np.max(np.abs(corr_matrix[:n_components, n_components:])):.6f}")
    
    # 6. Визуализация
    print("\n" + "="*60)
    print("Визуализация результатов")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # График 1: Объясненная дисперсия
    axes[0, 0].bar(range(1, len(pca_manual.explained_variance_ratio_) + 1), 
                   pca_manual.explained_variance_ratio_)
    axes[0, 0].set_xlabel('Главная компонента')
    axes[0, 0].set_ylabel('Доля объясненной дисперсии')
    axes[0, 0].set_title('Объясненная дисперсия по компонентам')
    axes[0, 0].grid(True, alpha=0.3)
    
    # График 2: Кумулятивная дисперсия
    axes[0, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-')
    for threshold in thresholds:
        axes[0, 1].axhline(y=threshold, color='r', linestyle='--', alpha=0.5, 
                          label=f'{threshold*100}%' if threshold == 0.95 else '')
    axes[0, 1].set_xlabel('Количество компонент')
    axes[0, 1].set_ylabel('Кумулятивная объясненная дисперсия')
    axes[0, 1].set_title('Кумулятивная объясненная дисперсия')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # График 3: Ошибка реконструкции в зависимости от числа компонент
    recon_errors = []
    n_comps_range = range(1, min(11, X.shape[1]))
    for n_comp in n_comps_range:
        pca_temp = MyPCA(n_components=n_comp)
        pca_temp.fit(X_scaled)
        recon_errors.append(pca_temp.reconstruction_error(X_scaled))
    
    axes[1, 0].plot(n_comps_range, recon_errors, 'g-')
    axes[1, 0].set_xlabel('Количество компонент')
    axes[1, 0].set_ylabel('Средняя квадратичная ошибка')
    axes[1, 0].set_title('Ошибка реконструкции ||F - G*U^T||²')
    axes[1, 0].grid(True, alpha=0.3)
    
    # График 4: Связь с регрессией (если целевая переменная доступна)
    axes[1, 1].scatter(X_manual[:, 0], y, alpha=0.5)
    axes[1, 1].set_xlabel('Первая главная компонента')
    axes[1, 1].set_ylabel('Целевая переменная')
    axes[1, 1].set_title('Связь первой компоненты с целевой переменной')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/pca_results.png')
    plt.show()
    
    # 7. Демонстрация связи с линейной регрессией
    print("\n" + "="*60)
    print("Демонстрация связи PCA с линейной регрессией")
    print("="*60)
    
    regression_results = pca_for_regression_demo(X, y, n_components_list=[2, 5, 8, 10])
    
    for n_comp, res in regression_results.items():
        print(f"\n{n_comp} компонент(ы):")
        print(f"  Объясненная дисперсия: {res['explained_variance']:.3f}")
        print(f"  R² на обучении: {res['train_score']:.3f}")
        print(f"  R² на тесте: {res['test_score']:.3f}")
        print(f"  Ошибка реконструкции: {res['recon_error']:.6f}")
    
    return {
        'pca_manual': pca_manual,
        'pca_sklearn': pca_sklearn,
        'X': X_scaled,
        'y': y,
        'cumulative_variance': cumulative_variance,
        'regression_results': regression_results
    }

if __name__ == "__main__":
    results = main()