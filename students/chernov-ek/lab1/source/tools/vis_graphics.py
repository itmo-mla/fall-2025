import numpy as np
import matplotlib.pylab as plt


def visualize_losses_metrics(train_losses, val_losses, metrics, figsize=(10, 4)):
    # Строим графики
    plt.figure(figsize=figsize)

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.title('Изменение функции потерь')
    plt.legend()
    plt.grid(True)

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(metrics, label='Training accuracy')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.title('Изменение точности')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def visualize_ranking_margins(margins_sorted, p1: int = 10, p2: int = 50):
    i = np.arange(len(margins_sorted))

    # Пороги по процентилям
    low_thr = np.percentile(margins_sorted, p1)    # нижние 10% — шумовые
    high_thr = np.percentile(margins_sorted, p2)   # верхние 50% — надёжные

    # Маски
    noise_mask = margins_sorted < low_thr
    border_mask = (margins_sorted >= low_thr) & (margins_sorted <= high_thr)
    reliable_mask = margins_sorted > high_thr

    # Построение
    plt.figure(figsize=(10, 4))
    plt.plot(i, margins_sorted, color='blue', linewidth=1.5)

    # Цветные зоны
    plt.fill_between(i, 0, margins_sorted, where=noise_mask, color='red', alpha=0.6)
    plt.fill_between(i, 0, margins_sorted, where=border_mask, color='khaki', alpha=0.6)
    plt.fill_between(i, 0, margins_sorted, where=reliable_mask, color='limegreen', alpha=0.6)

    # Подписи
    plt.text(len(margins_sorted)*0.05, np.min(margins_sorted) + 0.05, 'шумовые', fontsize=12)
    plt.text(len(margins_sorted)*0.4, 0.05, 'пограничные', fontsize=12)
    plt.text(len(margins_sorted)*0.75, np.max(margins_sorted) - 0.1, 'надёжные', fontsize=12)

    # Вывод
    plt.xlabel('i', fontsize=12)
    plt.ylabel('Margin', fontsize=12)
    plt.xlim(0, len(margins_sorted))
    plt.ylim(min(margins_sorted) - 0.1, max(margins_sorted) + 0.1)
    plt.tight_layout()
    plt.show()
