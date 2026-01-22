import matplotlib.pyplot as plt
import numpy as np


def calculate_margins(X, y, w):
    return y * np.dot(X, w)


def plot_margins(X, y, w, title="Margin Distribution"):
    margins = calculate_margins(X, y, w)
    sorted_margins = np.sort(margins)

    plt.figure(figsize=(12, 6))
    plt.plot(sorted_margins, color="blue", linewidth=2, label="Sorted Margins")

    threshold1 = 0
    threshold2 = 1.0

    plt.fill_between(
        range(len(sorted_margins)),
        sorted_margins,
        0,
        where=(sorted_margins < threshold1),
        color="red",
        alpha=0.3,
        label="Шумовые (M < 0)",
    )

    plt.fill_between(
        range(len(sorted_margins)),
        sorted_margins,
        0,
        where=((sorted_margins >= threshold1) & (sorted_margins < threshold2)),
        color="yellow",
        alpha=0.3,
        label="Пограничные (0 <= M < 1)",
    )

    plt.fill_between(
        range(len(sorted_margins)),
        sorted_margins,
        0,
        where=(sorted_margins >= threshold2),
        color="green",
        alpha=0.3,
        label="Надёжные (M >= 1)",
    )

    plt.axhline(y=threshold1, color="gray", linestyle="--", alpha=0.5)
    plt.axhline(y=threshold2, color="gray", linestyle=":", alpha=0.5)

    plt.title(title)
    plt.xlabel("Object Index (Sorted)")
    plt.ylabel("Margin Value")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_loss(model, title):
    plt.figure(figsize=(10, 5))
    plt.plot(model.loss_history, label="Train Loss", linewidth=2)
    if model.val_loss_history:
        plt.plot(model.val_loss_history, label="Test Loss", linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_comparision(model1, model2, name1, name2):
    plt.figure(figsize=(10, 5))
    plt.plot(model1.loss_history, label=name1, linewidth=2)
    plt.plot(model2.loss_history, label=name2, linewidth=2, linestyle="--")
    plt.title("Сравнение стратегий предъявления")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_multistart(history, best_model):
    plt.figure(figsize=(10, 6))
    for lh in history:
        plt.plot(lh, alpha=0.3, color="gray")
    plt.plot(best_model.loss_history, color="red", linewidth=2, label="Best Run")
    plt.title("Мультистарт (10 запусков)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
