import matplotlib.pyplot as plt
import numpy as np


def calculate_margin(w, x_i, y_i):
    # Вычисляет отступ для одного объекта. Возвращает отступ (правильная метка * предсказание)
    prediction = np.dot(w, x_i)
    margin = y_i * prediction
    return margin

def calculate_all_margins(w, X, y):
    # Вычисляет отступы для всех объектов выборки. Возвращает массив отступов
    margins = []
    for i in range(len(y)):
        margin = calculate_margin(w, X[i], y[i])
        margins.append(margin)
    return np.array(margins)

def margins_plot(margins):
    # Визуализируем отступы
    plt.figure(figsize=(10, 6))
    plt.hist(margins, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Граница решения')
    plt.xlabel('Отступ')
    plt.ylabel('Количество объектов')
    plt.title('Гистограмма отступов для случайно инициализированных весов')
    plt.legend()
    plt.grid(True, alpha=0.3)

    import os
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'margins_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_margins_by_class(margins, y, title="Анализ отступов по классам"):
    """
    Подробный анализ отступов с разделением по классам.
    Выводит статистику отдельно для каждого класса.
    """
    print(f"\n{'-'*10} {title} {'-'*10}")
    
    # Общая статистика
    print(f"\nОбщая статистика:")
    print(f"  Минимальный отступ: {np.min(margins):.4f}")
    print(f"  Максимальный отступ: {np.max(margins):.4f}")
    print(f"  Средний отступ: {np.mean(margins):.4f}")
    print(f"  Медианный отступ: {np.median(margins):.4f}")
    print(f"  Стандартное отклонение: {np.std(margins):.4f}")
    
    error_rate = np.sum(margins < 0) / len(margins)
    border_rate = np.sum((margins >= 0) & (margins < 1)) / len(margins)
    confident_rate = np.sum(margins >= 1) / len(margins)
    
    print(f"  Доля ошибок (M < 0): {error_rate:.1%}")
    print(f"  Доля пограничных (0 ≤ M < 1): {border_rate:.1%}")
    print(f"  Доля уверенных (M ≥ 1): {confident_rate:.1%}")
    
    # Анализ по классам
    print(f"\nАнализ по классам:")
    unique_labels = np.unique(y)
    
    for label in unique_labels:
        class_margins = margins[y == label]
        class_error_rate = np.sum(class_margins < 0) / len(class_margins)
        class_confident_rate = np.sum(class_margins >= 1) / len(class_margins)
        
        print(f"\n  Класс {label}:")
        print(f"    Количество объектов: {len(class_margins)}")
        print(f"    Средний отступ: {np.mean(class_margins):.4f}")
        print(f"    Минимальный отступ: {np.min(class_margins):.4f}")
        print(f"    Максимальный отступ: {np.max(class_margins):.4f}")
        print(f"    Доля отрицательных отступов (ошибки): {class_error_rate:.1%}")
        print(f"    Доля уверенных (M ≥ 1): {class_confident_rate:.1%}")
    
    return {
        'mean_margin': np.mean(margins),
        'error_rate': error_rate,
        'border_rate': border_rate,
        'confident_rate': confident_rate
    }


def plot_margin_ranking(margins, title="Распределение отступов по рангу"):
    """
    Визуализирует отступы отсортированные по возрастанию.
    Цветовая заливка разделяет объекты на категории:
    - Красный: M < 0 (ошибки классификации)
    - Желтый: 0 <= M < 1 (пограничные правильные предсказания)
    - Зеленый: M >= 1 (уверенные правильные предсказания)
    """
    # Сортируем отступы по возрастанию
    sorted_margins = np.sort(margins)
    n_objects = len(sorted_margins)
    x_range = np.arange(n_objects)

    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    ax.plot(x_range, sorted_margins, 'b-', linewidth=2, alpha=0.8, label='Отступы')

    # Красная зона: M < 0 (ошибки)
    noisy = sorted_margins < 0
    if np.any(noisy):
        ax.fill_between(x_range[noisy], sorted_margins[noisy], 0, 
                        color='red', alpha=0.3, label='Ошибки (M < 0)')
    
    # Желтая зона: 0 <= M < 1 (пограничные)
    borderline = (sorted_margins >= 0) & (sorted_margins < 1)
    if np.any(borderline):
        ax.fill_between(x_range[borderline], sorted_margins[borderline], 0, 
                        color='yellow', alpha=0.3, label='Пограничные (0 ≤ M < 1)')
    
    # Зеленая зона: M >= 1 (надежные)
    reliable = sorted_margins >= 1
    if np.any(reliable):
        ax.fill_between(x_range[reliable], sorted_margins[reliable], 0, 
                        color='green', alpha=0.3, label='Надёжные (M ≥ 1)')

    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(y=1, color='orange', linestyle='--', linewidth=2, alpha=0.7)

    ax.set_xlabel('Ранг объекта (от худшего к лучшему отступу)', fontsize=11)
    ax.set_ylabel('Отступ', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    error_rate = np.sum(sorted_margins < 0) / n_objects
    border_rate = np.sum((sorted_margins >= 0) & (sorted_margins < 1)) / n_objects
    confident_rate = np.sum(sorted_margins >= 1) / n_objects

    stats_text = (
        f"Доля ошибок (M < 0): {error_rate:.2%}\n"
        f"Доля пограничных (0 ≤ M < 1): {border_rate:.2%}\n"
        f"Доля уверенных (M ≥ 1): {confident_rate:.2%}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.35'),
    )

    plt.tight_layout()

    import os
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'margin_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_margins_detailed(margins, labels=None, title="Детальный анализ отступов"):
    """
    Подробный анализ отступов с разделением по классам
    """
    print(f"\n{'-'*10} {title} {'-'*10}")

    print(f"Общая статистика:")
    print(f"  Минимальный отступ: {np.min(margins):.4f}")
    print(f"  Максимальный отступ: {np.max(margins):.4f}")
    print(f"  Средний отступ: {np.mean(margins):.4f}")
    print(f"  Медианный отступ: {np.median(margins):.4f}")
    print(f"  Стандартное отклонение: {np.std(margins):.4f}")

    error_rate = np.sum(margins < 0) / len(margins)
    border_rate = np.sum((margins >= 0) & (margins < 1)) / len(margins)
    confident_rate = np.sum(margins >= 1) / len(margins)

    print(f"  Доля ошибок (M < 0): {error_rate:.1%}")
    print(f"  Доля пограничных (0 ≤ M < 1): {border_rate:.1%}")
    print(f"  Доля уверенных (M ≥ 1): {confident_rate:.1%}")

    if labels is not None:
        print(f"\nАнализ по классам:")
        unique_labels = np.unique(labels)

        for label in unique_labels:
            class_margins = margins[labels == label]
            class_error_rate = np.sum(class_margins < 0) / len(class_margins)
            class_confident_rate = np.sum(class_margins >= 1) / len(class_margins)

            print(f"  Класс {label}:")
            print(f"    Средний отступ: {np.mean(class_margins):.4f}")
            print(f"    Доля ошибок: {class_error_rate:.1%}")
            print(f"    Доля уверенных: {class_confident_rate:.1%}")

    return {
        'mean_margin': np.mean(margins),
        'error_rate': error_rate,
        'border_rate': border_rate,
        'confident_rate': confident_rate
    }

def plot_margin_evolution(margin_history, title="Эволюция отступов во время обучения"):
    """
    Визуализация изменения статистик отступов по эпохам
    """
    if not margin_history:
        print("Нет данных об эволюции отступов")
        return

    epochs = [stats['epoch'] for stats in margin_history]
    mean_margins = [stats['mean_margin'] for stats in margin_history]
    error_rates = [stats['error_rate'] for stats in margin_history]
    min_margins = [stats['min_margin'] for stats in margin_history]
    max_margins = [stats['max_margin'] for stats in margin_history]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Средний отступ
    ax1.plot(epochs, mean_margins, 'b-', linewidth=2, label='Средний отступ')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Средний отступ')
    ax1.set_title('Средний отступ')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Доля ошибок
    ax2.plot(epochs, error_rates, 'r-', linewidth=2, label='Доля ошибок')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Доля ошибок')
    ax2.set_title('Доля ошибок (M < 0)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Минимальный и максимальный отступы
    ax3.plot(epochs, min_margins, 'g--', linewidth=2, label='Мин. отступ')
    ax3.plot(epochs, max_margins, 'm--', linewidth=2, label='Макс. отступ')
    ax3.set_xlabel('Эпоха')
    ax3.set_ylabel('Отступ')
    ax3.set_title('Минимальный и максимальный отступы')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Гистограмма отступов для последней эпохи (если есть данные)
    if 'all_margins' in margin_history[-1]:
        ax4.hist(margin_history[-1]['all_margins'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Граница решения')
        ax4.set_xlabel('Отступ')
        ax4.set_ylabel('Количество объектов')
        ax4.set_title(f'Распределение отступов (эпоха {epochs[-1]})')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Нет данных для гистограммы', ha='center', va='center', transform=ax4.transAxes)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.close()

def track_margin_statistics(margins, epoch):
    """
    Отслеживание статистики отступов для одной эпохи
    """
    stats = {
        'epoch': epoch,
        'mean_margin': np.mean(margins),
        'error_rate': np.mean(margins < 0),
        'min_margin': np.min(margins),
        'max_margin': np.max(margins),
        'all_margins': margins.copy()
    }
    return stats