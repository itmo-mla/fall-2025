import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

california = fetch_california_housing()
X_raw = california.data
y = california.target
feature_names = california.feature_names

print(f"Размерность данных: {X_raw.shape}")
print(f"Количество образцов: {X_raw.shape[0]}")
print(f"Количество признаков: {X_raw.shape[1]}")
print(f"Целевая переменная: медианная стоимость дома (в $100,000)")
print("\nНазвания признаков:")
for i, name in enumerate(feature_names):
    print(f"  {i + 1}. {name}")

df_corr = pd.DataFrame(X_raw, columns=feature_names)
df_corr['Price'] = y

print("\n" + "=" * 70)
print("Корреляция признаков с целевой переменной")
print("=" * 70)
correlations = df_corr.corr()['Price'].sort_values(ascending=False)
for feature, corr in correlations.items():
    print(f"{feature:20} : {corr:.4f}")

print("\n" + "=" * 70)
print("Стандартизация данных для PCA")
print("=" * 70)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_raw)
print("Данные стандартизированы: среднее = 0, стандартное отклонение = 1")


# F = VDU^T - сингулярное разложение матрицы


class MyPCA:
    """
    Реализация PCA через сингулярное разложение (SVD)
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None  # строки матрицы U^T
        self.explained_variance_ = None  # λ_j (собственные значения F^TF)
        self.explained_variance_ratio_ = None  # λ_j/Σλ_i
        self.mean_ = None  # центрирование данных
        self.singular_values_ = None  # √λ_j (сингулярные значения)
        self.n_features_in_ = None

    def fit(self, X):
        """
        Обучение PCA на данных X.

        Параметры:
            X : array-like, shape (n_samples, n_features)
                Матрица признаков (уже центрированная или нет — центрирование делается внутри).
        """
        self.n_features_in_ = X.shape[1]
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_  # Центрирование обязательно для PCA

        # Вычисляем SVD - X_centered = U @ diag(S) @ Vt
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Определяем количество компонент
        if self.n_components is None:
            self.n_components = min(X.shape)
        else:
            self.n_components = min(self.n_components, X.shape[1])

        # Собственные значения матрицы ковариации - λ_j = s_j^2 / (n - 1)
        # Это дисперсии вдоль главных компонент
        full_explained_variance = (S ** 2) / (X.shape[0] - 1)

        # Общая дисперсия — сумма по ВСЕМ признакам (всем компонентам)
        total_variance = np.sum(full_explained_variance)

        # Доли объяснённой дисперсии для всех компонент
        full_explained_variance_ratio = full_explained_variance / total_variance

        # Теперь берём только первые n_components
        self.components_ = Vt[:self.n_components]  # строки U^T (веса признаков)
        self.singular_values_ = S[:self.n_components]
        self.explained_variance_ = full_explained_variance[:self.n_components]
        self.explained_variance_ratio_ = full_explained_variance_ratio[:self.n_components]

        return self

    def transform(self, X):
        if self.mean_ is None:
            raise ValueError("Модель не обучена")
        X_centered = X - self.mean_
        # G = F U (новые признаки)
        # Здесь преобразование в пространство главных компонент
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        if self.mean_ is None or self.components_ is None:
            raise ValueError("Модель не обучена")
        # F = G U^T (восстановление исходных признаков)
        return np.dot(X_transformed, self.components_) + self.mean_


def determine_effective_dimension(X_scaled, thresholds=[0.80, 0.90, 0.95, 0.99]):
    """
    Определение эффективной размерности выборки для различных порогов
    E_m = (λ_{m+1} + ... + λ_n)/(λ_1 + ... + λ_n) ≤ ε
    """
    pca = MyPCA()
    pca.fit(X_scaled)
    # Кумулятивная сумма объясненной дисперсии
    # 1 - E_m = (λ_1 + ... + λ_m)/(Σλ_i) - доля сохраненной дисперсии
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    results = {}
    for threshold in thresholds:
        # Находим первую компоненту, где кумулятивная дисперсия ≥ порога
        above_threshold = np.where(cumulative_variance >= threshold)[0]
        if len(above_threshold) > 0:
            effective_dim = above_threshold[0] + 1
        else:
            effective_dim = len(cumulative_variance)

        results[threshold] = {
            'dimension': effective_dim,
            'variance': cumulative_variance[effective_dim - 1] if effective_dim > 0 else 0
        }

    return results, cumulative_variance, pca


def plot_explained_variance(X_scaled, feature_names):
    """
    Визуализация объясненной дисперсии
    """
    thresholds = [0.80, 0.90, 0.95, 0.99]
    results, cumulative_var, pca = determine_effective_dimension(X_scaled, thresholds)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Scree plot (график собственных значений) анализ круглого склона для выбора m
    explained_var_ratio = pca.explained_variance_ratio_
    n_features = X_scaled.shape[1]
    bars = ax1.bar(range(1, n_features + 1), explained_var_ratio)

    # Добавляем числовые значения на столбцы
    for i, (bar, val) in enumerate(zip(bars, explained_var_ratio)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Выделяем пороговые компоненты
    colors = ['green', 'orange', 'blue', 'purple']
    for i, (threshold, info) in enumerate(results.items()):
        if info['dimension'] <= len(bars):
            bars[info['dimension'] - 1].set_edgecolor(colors[i])
            bars[info['dimension'] - 1].set_linewidth(3)
            bars[info['dimension'] - 1].set_facecolor('lightgray')

    ax1.set_xlabel('Номер главной компоненты', fontsize=12)
    ax1.set_ylabel('Доля объясненной дисперсии', fontsize=12)
    ax1.set_title('Scree Plot', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(explained_var_ratio) * 1.2])

    # График кумулятивной дисперсии
    ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-', linewidth=2, markersize=8)

    # Добавляем числовые метки
    for i, val in enumerate(cumulative_var):
        ax2.annotate(f'{val:.3f}', xy=(i + 1, val), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)

    for i, (threshold, info) in enumerate(results.items()):
        ax2.axhline(y=threshold, color=colors[i], linestyle='--', alpha=0.7, linewidth=2, label=f'{threshold * 100:.0f}% порог')
        ax2.axvline(x=info['dimension'], color=colors[i], linestyle=':', alpha=0.5, linewidth=2)
        ax2.annotate(f'{threshold * 100:.0f}%', xy=(info['dimension'], threshold), xytext=(5, 5),
                     textcoords='offset points', fontsize=10, color=colors[i], fontweight='bold')

    ax2.set_xlabel('Количество главных компонент', fontsize=12)
    ax2.set_ylabel('Кумулятивная дисперсия', fontsize=12)
    ax2.set_title('Кумулятивная объясненная дисперсия', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    ax2.set_xlim([0.5, len(cumulative_var) + 0.5])

    # Таблица эффективных размерностей
    ax3.axis('tight')
    ax3.axis('off')

    table_data = []
    for threshold, info in results.items():
        reduction = (1 - info['dimension'] / X_scaled.shape[1]) * 100
        table_data.append([
            f"{threshold * 100:.0f}%",
            info['dimension'],
            f"{info['variance']:.4f}",
            f"{reduction:.1f}%"
        ])

    table = ax3.table(cellText=table_data, colLabels=['Порог', 'Компонент', 'Дисперсия', 'Сокращение'], loc='center', cellLoc='center', colColours=['#f0f0f0'] * 4)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0)

    # Выделяем ячейки
    for i in range(len(table_data)):
        table[(i + 1, 1)].set_facecolor('#e6f3ff')
        table[(i + 1, 2)].set_facecolor('#e6ffe6')

    ax3.set_title('Эффективные размерности', fontsize=14, fontweight='bold', pad=20)

    plt.suptitle('Анализ PCA на стандартизированных данных датасета', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_filename = f"{PLOTS_DIR}/1_plot_explained_variance.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print("\n" + "-" * 70)
    print("Анализ эффективной размерности")
    print("-" * 70)
    print(f"Общее количество признаков: {X_scaled.shape[1]}")
    print("\nДля сохранения заданной доли дисперсии требуется компонент:")

    pca_full = MyPCA()
    pca_full.fit(X_scaled)
    cumulative_variance_correct = np.cumsum(pca_full.explained_variance_ratio_)

    for threshold in thresholds:
        above_threshold = np.where(cumulative_variance_correct >= threshold)[0]
        if len(above_threshold) > 0:
            effective_dim = above_threshold[0] + 1
            variance_retained = cumulative_variance_correct[effective_dim - 1]
            reduction = (1 - effective_dim / X_scaled.shape[1]) * 100
            print(f"    {threshold * 100:.0f}% дисперсии: {effective_dim} компонент "
                  f"(фактически {variance_retained:.3f}, сокращение {reduction:.1f}%)")

    # Вывод собственных значений и их вклада
    # λ_1 ≥ λ_2 ≥ ... ≥ λ_n ≥ 0
    print("\n" + "=" * 70)
    print("Детальный анализ главных компонент")
    print("=" * 70)
    print(f"{'Компонента':<12} {'Собств. знач.':<15} {'Доля':<10} {'Накопленная':<10}")
    print("-" * 60)

    # Собственные значения: λ_j = s_j^2 / (n-1)
    eigenvals = pca_full.singular_values_ ** 2 / (X_scaled.shape[0] - 1)

    for i in range(len(pca_full.explained_variance_ratio_)):
        print(f"PC{i + 1:<11} {eigenvals[i]:<15.2f} "
              f"{pca_full.explained_variance_ratio_[i]:<10.4f} "
              f"{cumulative_variance_correct[i]:<10.4f}")

    return pca_full, results


print("\nЗапуск PCA на стандартизированных данных")
pca_custom, eff_dim_results = plot_explained_variance(X_scaled, feature_names)


def compare_with_sklearn_california(X_scaled, n_components=None):
    """
    Сравнение реализации MyPCA с sklearn PCA
    """
    if n_components is None:
        # Используем порог 95% для определения количества компонент
        results, _, _ = determine_effective_dimension(X_scaled, [0.95])
        n_components = results[0.95]['dimension']
        print(f"Автоматически выбрано n_components={n_components} (95% дисперсии)")

    print("\n" + "=" * 70)
    print(f"Сравнение с реализацией sklearn (n_components={n_components})")
    print("=" * 70)

    # MyPCA
    my_pca = MyPCA(n_components=n_components)
    X_custom = my_pca.fit_transform(X_scaled)

    # sklearn PCA
    sklearn_pca = sklearnPCA(n_components=n_components)
    X_sklearn = sklearn_pca.fit_transform(X_scaled)

    print("\n1. Сравнение объясненной дисперсии:")
    print("-" * 60)
    print(f"{'Компонента':<12} {'My PCA':<15} {'sklearn PCA':<15} {'Разница':<12}")
    print("-" * 60)

    max_abs_diff = 0
    for i in range(n_components):
        my_var = my_pca.explained_variance_ratio_[i]
        sklearn_var = sklearn_pca.explained_variance_ratio_[i]
        abs_diff = abs(my_var - sklearn_var)
        max_abs_diff = max(max_abs_diff, abs_diff)

        print(f"PC{i + 1:<11} {my_var:.8f}{'':<6} {sklearn_var:.8f}{'':<6} {abs_diff:.2e}")

    print(f"\nМаксимальная абсолютная разница: {max_abs_diff:.2e}")

    print("\n2. Сравнение компонент (с учетом знака):")
    print("-" * 60)

    component_correlations = []
    X_my_adjusted = X_custom.copy()

    for i in range(n_components):
        # Вычисляем корреляцию между компонентами
        corr_matrix = np.corrcoef(X_custom[:, i], X_sklearn[:, i])
        correlation = corr_matrix[0, 1]
        component_correlations.append(correlation)

        # Направление собственных векторов не уникально
        # Может отличаться знаком (v_j и -v_j оба собственные векторы)
        if correlation < -0.99:
            X_my_adjusted[:, i] = -X_my_adjusted[:, i]
            correlation = -correlation
            sign_info = " (знак изменен)"
        elif abs(correlation) < 0.99:
            sign_info = " (НИЗКАЯ корреляция!)"
        else:
            sign_info = ""

        print(f"PC{i + 1}: Корреляция = {correlation:.8f}{sign_info}")

    avg_correlation = np.mean(np.abs(component_correlations))
    print(f"\nСредняя абсолютная корреляция: {avg_correlation:.8f}")

    print("\n3. Сравнение преобразованных данных:")
    print("-" * 60)

    mae = np.mean(np.abs(X_my_adjusted - X_sklearn))
    rmse = np.sqrt(np.mean((X_my_adjusted - X_sklearn) ** 2))
    max_diff = np.max(np.abs(X_my_adjusted - X_sklearn))

    print(f"MAE: {mae:.2e}")
    print(f"RMSE: {rmse:.2e}")
    print(f"Максимальная разность: {max_diff:.2e}")

    print("\n4. Проверка обратного преобразования:")
    print("-" * 60)

    X_reconstructed_custom = my_pca.inverse_transform(X_custom)
    X_reconstructed_sklearn = sklearn_pca.inverse_transform(X_sklearn)

    mse_custom = np.mean((X_scaled - X_reconstructed_custom) ** 2)
    mse_sklearn = np.mean((X_scaled - X_reconstructed_sklearn) ** 2)

    print(f"Среднеквадратичная ошибка восстановления:")
    print(f"    My PCA:      {mse_custom:.6e}")
    print(f"    sklearn PCA: {mse_sklearn:.6e}")

    if mse_sklearn > 0:
        rel_diff = abs(mse_custom - mse_sklearn) / mse_sklearn * 100
        print(f"    Относительная разница: {rel_diff:.6f}%")

    print("\n5. Проверка ортогональности компонент")
    print("-" * 60)

    # U^T U = I_m (ортогональность компонент)
    # Проверяем это свойство
    my_components = my_pca.components_
    my_orthogonality = np.abs(np.dot(my_components, my_components.T) - np.eye(n_components))

    sklearn_components = sklearn_pca.components_
    sklearn_orthogonality = np.abs(np.dot(sklearn_components, sklearn_components.T) - np.eye(n_components))

    print(f"Макс. отклонение от ортогональности:")
    print(f"  My PCA:      {np.max(my_orthogonality):.2e}")
    print(f"  sklearn PCA: {np.max(sklearn_orthogonality):.2e}")

    n_plots = min(3, n_components)
    if n_plots > 0:
        fig, axes = plt.subplots(2, n_plots, figsize=(5 * n_plots, 12))
        if n_plots == 1:
            axes = axes.reshape(2, 1)

        for i in range(n_plots):
            # Диаграмма рассеяния для сравнения компонент
            axes[0, i].scatter(X_my_adjusted[:, i], X_sklearn[:, i], alpha=0.5, s=20, color='blue', edgecolor='black',
                               linewidth=0.5)
            x_min = min(X_my_adjusted[:, i].min(), X_sklearn[:, i].min())
            x_max = max(X_my_adjusted[:, i].max(), X_sklearn[:, i].max())
            axes[0, i].plot([x_min, x_max], [x_min, x_max], 'r--', alpha=0.8, linewidth=2, label='Идеальное соответствие')
            axes[0, i].set_xlabel('My PCA', fontsize=12)
            axes[0, i].set_ylabel('sklearn PCA', fontsize=12)
            axes[0, i].set_title(f'PC{i + 1} (корр.={component_correlations[i]:.8f})', fontsize=14, fontweight='bold', pad=10)
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].legend(loc='best')
            axes[0, i].text(0.05, 0.95, f'R = {component_correlations[i]:.8f}', transform=axes[0, i].transAxes, fontsize=11,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # Гистограмма разностей
            diff = X_my_adjusted[:, i] - X_sklearn[:, i]
            axes[1, i].hist(diff, bins=50, alpha=0.7, edgecolor='black', color='green')
            axes[1, i].axvline(x=0, color='r', linestyle='--', linewidth=2)
            axes[1, i].axvline(x=np.mean(diff), color='b', linestyle='-', linewidth=2, alpha=0.7, label=f'Среднее: {np.mean(diff):.2e}')

            # Статистика разностей
            stats_text = f'Среднее: {np.mean(diff):.2e}\nСтд: {np.std(diff):.2e}\nМакс: {np.max(np.abs(diff)):.2e}'
            axes[1, i].text(0.95, 0.95, stats_text, transform=axes[1, i].transAxes, fontsize=10,
                            verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round',
                                                                                            facecolor='lightblue', alpha=0.8))
            axes[1, i].set_xlabel('Разность my - sklearn', fontsize=12)
            axes[1, i].set_ylabel('Частота', fontsize=12)
            axes[1, i].set_title(f'Распределение разностей PC{i + 1}', fontsize=14, fontweight='bold', pad=10)
            axes[1, i].legend(loc='best')
            axes[1, i].grid(True, alpha=0.3)

        plt.suptitle(f'Сравнение My PCA и sklearn PCA (n_components={n_components})', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plot_filename = f"{PLOTS_DIR}/2_plot_comparison_pca.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    return my_pca, sklearn_pca, X_my_adjusted, X_sklearn


my_pca, sklearn_pca, X_my_adj, X_sklearn = compare_with_sklearn_california(X_scaled, n_components=6)


def pca_regression_analysis(X_raw, y, feature_names):
    """
    Анализ влияния PCA на качество линейной регрессии
    """
    print("1. Разделение данных: 70% train, 30% test")
    print("2. Стандартизация признаков для PCA")
    print("3. Целевая переменная НЕ стандартизируется (для интерпретации R²)")
    print("4. Тестируем от 1 до 8 компонент PCA")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split( X_raw, y, test_size=0.3, random_state=42, shuffle=True)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)

    print("\nБАЗОВАЯ МОДЕЛЬ: Линейная регрессия без PCA")

    model_base = LinearRegression()
    model_base.fit(X_train_scaled, y_train)

    train_score_base = model_base.score(X_train_scaled, y_train)
    test_score_base = model_base.score(X_test_scaled, y_test)

    print(f"R² на обучающей выборке: {train_score_base:.6f}")
    print(f"R² на тестовой выборке:  {test_score_base:.6f}")
    print(f"Разность (переобучение):  {train_score_base - test_score_base:.6f}")

    # Предсказания для базовой модели
    y_pred_base_train = model_base.predict(X_train_scaled)
    y_pred_base_test = model_base.predict(X_test_scaled)
    mse_base_test = np.mean((y_test - y_pred_base_test) ** 2)
    print(f"MSE на тестовой выборке:   {mse_base_test:.6f}")

    print("\n\n" + "=" * 70)
    print("ЭКСПЕРИМЕНТ: Линейная регрессия с разным количеством компонент PCA")
    print("=" * 70)

    n_features = X_train_scaled.shape[1]
    n_components_list = list(range(1, n_features + 1))

    results = []
    y_predictions = []

    print(f"\n{'n_comp':<8} {'Train R²':<12} {'Test R²':<12} {'Diff':<12} {'Var Expl':<12} {'Test MSE':<12}")
    print("-" * 80)

    for n in n_components_list:
        pca = MyPCA(n_components=n)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        explained_var = np.sum(pca.explained_variance_ratio_)

        model = LinearRegression()
        model.fit(X_train_pca, y_train)

        train_score = model.score(X_train_pca, y_train)
        test_score = model.score(X_test_pca, y_test)
        score_diff = train_score - test_score

        y_pred_test = model.predict(X_test_pca)
        mse_test = np.mean((y_test - y_pred_test) ** 2)

        y_predictions.append(y_pred_test)
        results.append({
            'n_components': n,
            'train_score': train_score,
            'test_score': test_score,
            'score_diff': score_diff,
            'explained_variance': explained_var,
            'mse_test': mse_test,
            'model': model,
            'pca': pca
        })

        print(f"{n:<8} {train_score:.6f}    {test_score:.6f}    {score_diff:.6f}    "
              f"{explained_var:.6f}    {mse_test:.6f}")

    # Находим оптимальное количество компонент
    test_scores = [r['test_score'] for r in results]
    optimal_idx = np.argmax(test_scores)
    optimal_result = results[optimal_idx]

    print("\n" + "=" * 70)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 70)

    print(f"\nОптимальное количество компонент для регрессии: {optimal_result['n_components']}")
    print(f"Объясненная дисперсия: {optimal_result['explained_variance']:.4f}")
    print(f"Качество модели (Test R²): {optimal_result['test_score']:.6f}")
    print(f"Качество модели (Test MSE): {optimal_result['mse_test']:.6f}")
    print(f"Без PCA (Test R²): {test_score_base:.6f}")
    print(f"Без PCA (Test MSE): {mse_base_test:.6f}")

    improvement_r2 = optimal_result['test_score'] - test_score_base
    improvement_mse = mse_base_test - optimal_result['mse_test']

    # Проблемы при использовании PCA для регрессии
    # 1. Потеря информации, важной для предсказания
    # 2. PCA оптимизирует дисперсию, а не предсказательную силу

    print("\n" + "=" * 70)

    if optimal_result['test_score'] < test_score_base:
        print("\nPCA НЕ РЕКОМЕНДУЕТСЯ для этого датасета!")
        print("Причины:")
        print("1. PCA может терять информацию, важную для предсказания")
        print("2. PCA оптимизирует сохранение дисперсии, а не качество регрессии")
        print("3. Для борьбы с переобучением лучше использовать регуляризацию")
    else:
        print("\nPCA может быть полезен для этого датасета")
        reduction = (1 - optimal_result['n_components'] / n_features) * 100
        print(f"Сокращение размерности: {reduction:.1f}%")

    fig = plt.figure(figsize=(18, 5))

    # График 1: Качество модели
    ax1 = plt.subplot(1, 3, 1)
    n_components = [r['n_components'] for r in results]
    train_scores = [r['train_score'] for r in results]
    test_scores = [r['test_score'] for r in results]

    ax1.plot(n_components, train_scores, 'b-', label='Обучающая', linewidth=3, marker='o', markersize=8)
    ax1.plot(n_components, test_scores, 'r-', label='Тестовая', linewidth=3, marker='s', markersize=8)
    ax1.axhline(y=test_score_base, color='g', linestyle='--', label=f'Без PCA (test={test_score_base:.3f})', linewidth=2)
    ax1.axvline(x=optimal_result['n_components'], color='k', linestyle=':', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Количество компонент PCA', fontsize=12)
    ax1.set_ylabel('R² score', fontsize=12)
    ax1.set_title('Качество линейной регрессии', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.5, max(n_components) + 0.5])
    ax1.set_ylim([min(min(train_scores), min(test_scores)) * 0.95, max(max(train_scores), max(test_scores)) * 1.05])

    # График 2: MSE
    ax2 = plt.subplot(1, 3, 2)
    mse_scores = [r['mse_test'] for r in results]
    ax2.plot(n_components, mse_scores, 'm-', linewidth=3, marker='d', markersize=8, label='С PCA')
    ax2.axhline(y=mse_base_test, color='g', linestyle='--', label=f'Без PCA (MSE={mse_base_test:.3f})', linewidth=2)
    ax2.axvline(x=optimal_result['n_components'], color='k', linestyle=':', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Количество компонент PCA', fontsize=12)
    ax2.set_ylabel('MSE (меньше - лучше)', fontsize=12)
    ax2.set_title('Среднеквадратичная ошибка', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.5, max(n_components) + 0.5])

    # График 3: Объясненная дисперсия и переобучение
    ax3 = plt.subplot(1, 3, 3)
    ax3_twin = ax3.twinx()
    explained_vars = [r['explained_variance'] for r in results]
    score_diffs = [r['score_diff'] for r in results]
    line1 = ax3.plot(n_components, explained_vars, 'g-', label='Объясненная дисперсия', linewidth=3, marker='^', markersize=8)
    ax3.set_xlabel('Количество компонент PCA', fontsize=12)
    ax3.set_ylabel('Объясненная дисперсия', fontsize=12, color='green')
    ax3.tick_params(axis='y', labelcolor='green')

    line2 = ax3_twin.plot(n_components, score_diffs, 'm-', label='Переобучение (train-test)', linewidth=3, marker='v', markersize=8)
    ax3_twin.set_ylabel('Разность R² (train-test)', fontsize=12, color='purple')
    ax3_twin.tick_params(axis='y', labelcolor='purple')
    ax3.axhline(y=0.90, color='g', linestyle='--', alpha=0.5, linewidth=1, label='90% дисперсии')
    ax3.axhline(y=0.95, color='g', linestyle=':', alpha=0.5, linewidth=1, label='95% дисперсии')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='lower right')
    ax3.set_title('Объясненная дисперсия и переобучение', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.suptitle('Анализ PCA для линейной регрессии на датасете', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plot_filename = f"{PLOTS_DIR}/3_plot_pca_regression.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    return results, optimal_result


print("\n" + "=" * 70)
print("ЗАПУСК АНАЛИЗА PCA ДЛЯ РЕГРЕССИИ")
print("=" * 70)
regression_results, optimal_reg_result = pca_regression_analysis(X_raw, y, feature_names)

# Обучаем PCA на всех данных для анализа
pca_all = MyPCA(n_components=8)
X_all_pca = pca_all.fit_transform(X_scaled)

# Анализируем веса в пространстве PCA
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print("\nВклады компонент в регрессионную модель (стандартизированные коэффициенты):")
print("-" * 80)
print(f"{'Компонента':<12} {'Объясн. дисперсия':<18} {'Коэффициент':<15} {'Вклад (%)':<12}")
print("-" * 80)

for n in range(1, 9):
    pca_temp = MyPCA(n_components=n)
    X_train_pca = pca_temp.fit_transform(X_train)
    X_test_pca = pca_temp.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_pca, y_train)

    # Анализируем коэффициенты
    coefs = model.coef_
    explained_var = np.sum(pca_temp.explained_variance_ratio_)

    # Вклад каждой компоненты (абсолютные значения коэффициентов)
    if n == 1:
        print(f"PC1{' ' * 9} {pca_temp.explained_variance_ratio_[0]:<18.4f} {coefs[0]:<15.4f} {'100.0':<12}")
    else:
        total_impact = np.sum(np.abs(coefs))
        for i in range(n):
            impact_pct = (np.abs(coefs[i]) / total_impact * 100) if total_impact > 0 else 0
            if i == n - 1:
                print(
                    f"PC{i + 1}{' ' * 9} {pca_temp.explained_variance_ratio_[i]:<18.4f} {coefs[i]:<15.4f} {impact_pct:<11.1f}%")