import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

from data_preprocessing import X_train_array, y_train_array
from linear_classifier import add_bias_term, initialize_weights
from margins import calculate_all_margins, margins_plot
from stochastic_gradient_descent import stochastic_gradient_descent, sgd_plot

def learning_rates_fig(results):
    plt.figure(figsize=(12, 6))
    for lr, (error, history) in results.items():
        plt.plot(history, label=f'lr={lr}, error={error:.3f}')
    plt.xlabel('Эпоха')
    plt.ylabel('Средняя потеря')
    plt.title('Сравнение разных learning rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    import os
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'learning_rate_comparison.png'))
    plt.close()

def optimization_methods_fig(results):
    plt.figure(figsize=(12, 6))
    for name, (error, history, *_rest) in results.items():
        plt.plot(history, label=f'{name}, test_error={error:.3f}')
    plt.xlabel('Эпоха')
    plt.ylabel('Средняя потеря')
    plt.title('Сравнение методов оптимизации')
    plt.legend()
    plt.grid(True, alpha=0.3)
    import os
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'optimization_methods_comparison.png'))
    plt.close()

def learning_rates_compare():
    X_train_with_bias = add_bias_term(X_train_array)
    w_init = initialize_weights(X_train_with_bias.shape[1])

    learning_rates = [0.001, 0.005, 0.01]
    results = {}

    for lr in learning_rates:
        print(f"\nОбучение с learning_rate = {lr}")
        w_trained, loss_history = stochastic_gradient_descent(
            X_train_with_bias, 
            y_train_array,
            w_init,
            learning_rate=lr,
            n_epochs=100,
            batch_size=32  # Увеличиваем batch_size для стабильности
        )
        
        # Оцениваем качество
        margins_trained = calculate_all_margins(w_trained, X_train_with_bias, y_train_array)
        error_rate = np.sum(margins_trained < 0) / len(margins_trained)
        results[lr] = (error_rate, loss_history)
        margins_plot(margins_trained)
        sgd_plot(loss_history, '')
    
    learning_rates_fig(results)

def multi_start_training(X, y, n_starts=5, n_epochs=50, batch_size=32):
    """
    Мультистарт: несколько запусков с разной инициализацией
    """
    best_w = None
    best_loss = float('inf')
    best_error = 1.0
    
    for start in range(n_starts):
        print(f"Запуск {start + 1}/{n_starts}")
        
        # Случайная инициализация
        w = initialize_weights(X.shape[1])
        
        # Обучение
        w_trained, loss_history = stochastic_gradient_descent(
            X, y, w, learning_rate=0.005, n_epochs=n_epochs, batch_size=batch_size
        )
        
        # Оценка качества
        final_loss = loss_history[-1]
        margins = calculate_all_margins(w_trained, X, y)
        error_rate = np.sum(margins < 0) / len(margins)
        
        if error_rate < best_error:
            best_error = error_rate
            best_w = w_trained
            best_loss = final_loss
    
    print(f"Лучшая ошибка из {n_starts} запусков: {best_error:.4f}")
    return best_w, best_error

def print_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Вычисляет и выводит confusion matrix.
    Предполагается, что метки классов: -1 и 1
    """
    # Вычисляем компоненты confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))   # True Positive
    tn = np.sum((y_true == -1) & (y_pred == -1)) # True Negative
    fp = np.sum((y_true == -1) & (y_pred == 1))  # False Positive
    fn = np.sum((y_true == 1) & (y_pred == -1))  # False Negative
    
    print(f"\n{'-'*10} {title} {'-'*10}")
    print("                      Actual")
    print("                  Positive  Negative")
    print(f"Predict Positive     {tp:4d}      {fp:4d}")
    print(f"        Negative     {fn:4d}      {tn:4d}")
    print()
    
    # Дополнительные метрики
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Точность (Accuracy):  {accuracy:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall:               {recall:.4f}")
    print(f"F1-score:             {f1:.4f}")
    
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1
    }


def calculate_metrics(y_true, y_pred):
    """
    Вычисляет метрики классификации: Accuracy, Precision, Recall, F1
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1


def generate_comparison_table(results, X_train, y_train, X_test, y_test):
    """
    Генерирует таблицу сравнения всех методов с метриками на train и test
    """
    print("\n" + "-"*20 + " СРАВНИТЕЛЬНАЯ ТАБЛИЦА ВСЕХ МЕТОДОВ " + "-"*20)
    print(f"{'Метод':<30} {'Val Err':>10} {'Test Acc':>10} {'Test Prec':>10} {'Test Rec':>10} {'Test F1':>10}")
    print("-"*80)
    
    table_data = []
    
    for method_name, method_results in results.items():
        w_trained = method_results.get('weights')
        val_error = method_results.get('val_error', 0.0)
        
        if w_trained is not None:
            # Предсказания на тесте
            # Проверяем нужен ли bias
            if X_test.shape[1] + 1 == w_trained.shape[0]:
                X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
            else:
                X_test_bias = X_test
            
            y_pred = np.sign(np.dot(X_test_bias, w_trained))
            test_acc, test_prec, test_rec, test_f1 = calculate_metrics(y_test, y_pred)
            
            print(f"{method_name:<30} {val_error:>10.4f} {test_acc:>10.4f} {test_prec:>10.4f} {test_rec:>10.4f} {test_f1:>10.4f}")
            
            table_data.append({
                'method': method_name,
                'val_error': val_error,
                'test_acc': test_acc,
                'test_prec': test_prec,
                'test_rec': test_rec,
                'test_f1': test_f1
            })
    
    print("-"*80)
    return table_data


def compare_with_baseline(X_train, y_train, X_test, y_test, our_model_weights):
    """
    Сравнение нашей модели с эталонной LogisticRegression из sklearn
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, recall_score, f1_score

    # Наша модель: поддерживаем веса с bias (w0) и матрицы без bias.
    # Если X имеет на 1 признак меньше, чем размерность весов — добавляем столбец единиц.
    if X_test.shape[1] + 1 == our_model_weights.shape[0]:
        X_test_ours = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
        X_train_ours = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    else:
        X_test_ours = X_test
        X_train_ours = X_train

    our_predictions = np.sign(np.dot(X_test_ours, our_model_weights))
    our_accuracy = accuracy_score(y_test, our_predictions)
    our_precision = precision_score(y_test, our_predictions)
    our_recall = recall_score(y_test, our_predictions)
    our_f1 = f1_score(y_test, our_predictions)

    # Эталонная модель (LogisticRegression)
    baseline_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=1.0  # коэффициент регуляризации (обратный к alpha в SGD)
    )

    baseline_model.fit(X_train, y_train)
    baseline_predictions = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)
    baseline_precision = precision_score(y_test, baseline_predictions)
    baseline_recall = recall_score(y_test, baseline_predictions)
    baseline_f1 = f1_score(y_test, baseline_predictions)

    print("\n" + "-" * 5 + " СРАВНЕНИЕ С ЭТАЛОННОЙ МОДЕЛЬЮ (LogisticRegression) " + "-" * 5)
    print(f"{'Метрика':<12}{'Наша':>12}{'Baseline':>12}{'Δ (ours-base)':>14}")
    print("-" * 60)
    print(f"{'Accuracy':<12}{our_accuracy:>12.4f}{baseline_accuracy:>12.4f}{(our_accuracy-baseline_accuracy):>14.4f}")
    print(f"{'Precision':<12}{our_precision:>12.4f}{baseline_precision:>12.4f}{(our_precision-baseline_precision):>14.4f}")
    print(f"{'Recall':<12}{our_recall:>12.4f}{baseline_recall:>12.4f}{(our_recall-baseline_recall):>14.4f}")
    print(f"{'F1-score':<12}{our_f1:>12.4f}{baseline_f1:>12.4f}{(our_f1-baseline_f1):>14.4f}")
    
    # Выводим confusion matrix для нашей модели
    print_confusion_matrix(y_test, our_predictions, "Confusion Matrix (Наша модель)")
    
    # Выводим confusion matrix для baseline
    print_confusion_matrix(y_test, baseline_predictions, "Confusion Matrix (Baseline)")

    return our_accuracy, baseline_accuracy
