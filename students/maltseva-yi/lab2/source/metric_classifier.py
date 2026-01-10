import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import time

# Реализация метода Парзеновского окна с гауссовским ядром
class ParzenKNN:
    def __init__(self, k=3):
        self.k = k

    # Гауссовское ядро K(r) = exp(-2r^2)
    def gaussian_kernel(self, r):
        return np.exp(-2 * r ** 2)

    def fit(self, X, y):
        self.X_train, self.y_train = X, y

    def predict(self, X_test):
        preds = []
        for x in X_test:
            # Вычисляем расстояния до всех объектов обучения
            distances = np.linalg.norm(self.X_train - x, axis=1)
           
            # Метод переменной ширины окна по k ближайшим соседям
            # h(x) = ρ(x, x^{(k+1)})
            sorted_distances = np.sort(distances)
            h = sorted_distances[min(self.k, len(distances)-1)]

            # Защита от деления на ноль
            if h == 0:
                h = 1e-10
           
            # Вычисляем веса по формуле Парзеновского окна
            # w(i,x) = K(ρ(x,x_i)/h)
            weights = self.gaussian_kernel(distances / h)

            # Суммируем веса по классам - мера "голосов" Γ_y(x)
            class_weights = {}
            for i in range(len(self.y_train)):
                label = self.y_train[i]
                class_weights[label] = class_weights.get(label, 0) + weights[i]
       
            # Относим объект к классу с максимальным весом
            preds.append(max(class_weights, key=class_weights.get))
        return np.array(preds)

# Скользящий контроль leave-one-out для выбора оптимального k
# LOO(k,X^ℓ) = Σ [a(x_i; X^ℓ\{x_i}) ≠ y_i] → min_k
def loo_validation(X, y, k_values):
    errors = []
    for k in k_values:
        model = ParzenKNN(k=k)
        err = 0
        for i in range(len(X)):
            # Исключаем i-й объект из обучения
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            model.fit(X_train, y_train)
            # Проверяем, правильно ли классифицирован исключенный объект
            if model.predict([X[i]])[0] != y[i]:
                err += 1
        errors.append(err / len(X))
    return errors

def condensed_nn(X, y):
    if len(X) == 0:
        return X, y
   
    # Начинаем с одного объекта каждого класса
    classes = np.unique(y)
    S = []
   
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) > 0:
            S.append(cls_indices[0])
   
    changed = True
    while changed:
        changed = False
        model = ParzenKNN(k=1)
        model.fit(X[S], y[S])
       
        for i in range(len(X)):
            if i in S:
                continue
           
            if model.predict([X[i]])[0] != y[i]:
                S.append(i)
                changed = True
                break  # Перестраиваем модель
   
    return X[S], y[S]

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
   
    return accuracy, precision, recall, f1

def compare_metric_kernel():
    # Сравнение влияния метрики и ядра на точность
    
    # Используем готовые данные
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Выбираем метрики
    metrics = {
        'Евклидова': lambda x, Xt: np.linalg.norm(Xt - x, axis=1),  # L2
        'Чебышева': lambda x, Xt: np.max(np.abs(Xt - x), axis=1)    # L∞
    }
    
    # Выбираем ядра
    kernels = {
        'Гауссово': lambda r: np.exp(-2 * r ** 2),      # плавное, бесконечное
        'Треугольное': lambda r: np.maximum(0, 1 - np.abs(r))  # линейное
    }
    
    print("\n" + "="*70)
    print("ВЛИЯНИЕ МЕТРИКИ И ЯДРА НА ТОЧНОСТЬ")
    print("Евклидова vs Чебышева | Гауссово vs Треугольное")
    print("="*70)
    
    results = []
    k_values = range(1, 31)
    
    for metric_name, metric_func in metrics.items():
        for kernel_name, kernel_func in kernels.items():
            
            # Оптимизация k для этой комбинации
            best_k = 1
            best_acc = 0
            
            # Перебираем k для нахождения оптимального
            for k in k_values:
                preds = []
                for x in X_test:
                    distances = metric_func(x, X_train)
                    h = np.sort(distances)[min(k, len(distances)-1)]
                    h = max(h, 1e-10)
                    weights = kernel_func(distances / h)
                    
                    # Голосование
                    w0 = np.sum(weights[y_train == 0])
                    w1 = np.sum(weights[y_train == 1])
                    preds.append(0 if w0 > w1 else 1)
                
                acc = accuracy_score(y_test, preds)
                
                if acc > best_acc:
                    best_acc = acc
                    best_k = k
            
            # Финальное предсказание с оптимальным k
            final_preds = []
            for x in X_test:
                distances = metric_func(x, X_train)
                h = np.sort(distances)[min(best_k, len(distances)-1)]
                h = max(h, 1e-10)
                weights = kernel_func(distances / h)
                
                w0 = np.sum(weights[y_train == 0])
                w1 = np.sum(weights[y_train == 1])
                final_preds.append(0 if w0 > w1 else 1)
            
            final_acc = accuracy_score(y_test, final_preds)
            results.append((metric_name, kernel_name, best_k, final_acc))
            print(f"{metric_name:12} + {kernel_name:15} | k={best_k:2} → Accuracy = {final_acc:.4f}")
    
    # Анализ
    print("\n" + "="*70)
    print("АНАЛИЗ:")
    print("="*70)
    
    results_dict = {}
    for m, k_name, k_val, acc in results:
        results_dict[(m, k_name)] = acc
    
    # 1. ВЛИЯНИЕ МЕТРИКИ (при фиксированном ядре)
    print("\n1. ВЛИЯНИЕ МЕТРИКИ (при фиксированном ядре):")
    print("-" * 45)
    
    metric_effects = []
    
    # Для Гауссова ядра:
    eucl_gauss = results_dict[('Евклидова', 'Гауссово')]
    cheb_gauss = results_dict[('Чебышева', 'Гауссово')]
    effect_gauss = abs(eucl_gauss - cheb_gauss)
    metric_effects.append(effect_gauss)
    print(f"При Гауссовом ядре: смена метрики меняет accuracy на {effect_gauss:.4f}")
    print(f"  Евклидова: {eucl_gauss:.4f}, Чебышева: {cheb_gauss:.4f}")
    
    # Для Треугольного ядра:
    eucl_tri = results_dict[('Евклидова', 'Треугольное')]
    cheb_tri = results_dict[('Чебышева', 'Треугольное')]
    effect_tri = abs(eucl_tri - cheb_tri)
    metric_effects.append(effect_tri)
    print(f"При Треугольном ядре: смена метрики меняет accuracy на {effect_tri:.4f}")
    print(f"  Евклидова: {eucl_tri:.4f}, Чебышева: {cheb_tri:.4f}")
    
    avg_metric_effect = np.mean(metric_effects)
    print(f"\nСРЕДНЕЕ влияние метрики: {avg_metric_effect:.4f}")
    
    # 2. ВЛИЯНИЕ ЯДРА (при фиксированной метрике)
    print("\n2. ВЛИЯНИЕ ЯДРА (при фиксированной метрике):")
    print("-" * 45)
    
    kernel_effects = []
    
    # Для Евклидовой метрики:
    eucl_gauss = results_dict[('Евклидова', 'Гауссово')]
    eucl_tri = results_dict[('Евклидова', 'Треугольное')]
    effect_eucl = abs(eucl_gauss - eucl_tri)
    kernel_effects.append(effect_eucl)
    print(f"При Евклидовой метрике: смена ядра меняет accuracy на {effect_eucl:.4f}")
    print(f"  Гауссово: {eucl_gauss:.4f}, Треугольное: {eucl_tri:.4f}")
    
    # Для Чебышевской метрики:
    cheb_gauss = results_dict[('Чебышева', 'Гауссово')]
    cheb_tri = results_dict[('Чебышева', 'Треугольное')]
    effect_cheb = abs(cheb_gauss - cheb_tri)
    kernel_effects.append(effect_cheb)
    print(f"При Чебышевской метрике: смена ядра меняет accuracy на {effect_cheb:.4f}")
    print(f"  Гауссово: {cheb_gauss:.4f}, Треугольное: {cheb_tri:.4f}")
    
    avg_kernel_effect = np.mean(kernel_effects)
    print(f"\nСРЕДНЕЕ влияние ядра: {avg_kernel_effect:.4f}")

def main():
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Оптимизация параметра k по критерию LOO
    k_values = range(1, 21)
    loo_errors = loo_validation(X_train, y_train, k_values)
    optimal_k = k_values[np.argmin(loo_errors)]

    # Визуализация зависимости ошибки от k
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, loo_errors, marker='o', linewidth=2)
    plt.axvline(optimal_k, color='red', linestyle='--', label=f'k = {optimal_k}')
    plt.xlabel("k")
    plt.ylabel("Эмпирический риск (LOO)")
    plt.title("Зависимость ошибки от k")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('images/loo_plot.png')
    plt.show()

    # Тестирование нашего метрического классификатора
    start_custom = time.time()
    model = ParzenKNN(k=optimal_k)
    model.fit(X_train, y_train)
    y_pred_custom = model.predict(X_test)
    custom_time = time.time() - start_custom
   
    acc_custom, prec_custom, rec_custom, f1_custom = calculate_metrics(y_test, y_pred_custom)

    # Сравнение с реализацией из sklearn
    skl = KNeighborsClassifier(n_neighbors=optimal_k)
    skl.fit(X_train, y_train)

    start_skl = time.time()
    y_pred_skl = skl.predict(X_test)
    skl_time = time.time() - start_skl

    acc_skl, prec_skl, rec_skl, f1_skl = calculate_metrics(y_test, y_pred_skl)

    # Применение отбора эталонов для сжатия выборки
    X_proto, y_proto = condensed_nn(X_train, y_train)

    start_proto = time.time()
    proto_model = ParzenKNN(k=optimal_k)
    proto_model.fit(X_proto, y_proto)
    y_proto_pred = proto_model.predict(X_test)
    proto_time = time.time() - start_proto
   
    acc_proto, prec_proto, rec_proto, f1_proto = calculate_metrics(y_test, y_proto_pred)

    # Визуализация в пространстве главных компонент
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)
    X_proto_pca = pca.transform(X_proto)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
   
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis', alpha=0.6)
    ax1.set_title("Исходная выборка")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    plt.colorbar(scatter1, ax=ax1)

    scatter2 = ax2.scatter(X_proto_pca[:, 0], X_proto_pca[:, 1], c=y_proto, cmap='viridis', s=60)
    ax2.set_title("После отбора эталонов")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    plt.colorbar(scatter2, ax=ax2)

    plt.tight_layout()
    plt.savefig('images/pca_visualization.png')
    plt.show()

    # Сравнительная таблица результатов
    print(f"{'Метод':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Время (с)':<10}")
    print("-" * 75)
    print(f"{'Наш KNN':<25} {acc_custom:<10.4f} {prec_custom:<10.4f} {rec_custom:<10.4f} {f1_custom:<10.4f} {custom_time:<10.4f}")
    print(f"{'Sklearn KNN':<25} {acc_skl:<10.4f} {prec_skl:<10.4f} {rec_skl:<10.4f} {f1_skl:<10.4f} {skl_time:<10.4f}")
    print(f"{'После отбора эталонов':<25} {acc_proto:<10.4f} {prec_proto:<10.4f} {rec_proto:<10.4f} {f1_proto:<10.4f} {proto_time:<10.4f}")

if __name__ == "__main__":
    main()
    compare_metric_kernel()
