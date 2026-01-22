import matplotlib.pyplot as plt


def plot_differences(history_1, history_2):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history_1, "b-", linewidth=2, label="Newton-Raphson")
    axes[0].set_xlabel("Итерация", fontsize=12)
    axes[0].set_ylabel("||β_new - β_old||", fontsize=12)
    axes[0].set_title("Сходимость Newton-Raphson", fontsize=14, fontweight="bold")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_2, "orange", linewidth=2, label="IRLS")
    axes[1].set_xlabel("Итерация", fontsize=12)
    axes[1].set_ylabel("||β_new - β_old||", fontsize=12)
    axes[1].set_title("Сходимость IRLS", fontsize=14, fontweight="bold")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
