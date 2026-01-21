import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_training(loss_history, q_history, outpath: str, title: str):
    loss_history = np.asarray(loss_history, dtype=float)
    q_history = np.asarray(q_history, dtype=float)

    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label="Loss", linewidth=1.5)
    plt.plot(q_history, label="Q (recursive)", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_margin_ranking(margins: np.ndarray, outpath: str, title: str, thr: float = 0.3):
    m = np.sort(np.asarray(margins, dtype=float).reshape(-1))
    x = np.arange(len(m))

    plt.figure(figsize=(5, 3))
    plt.plot(m, c='k', linewidth=3)
    plt.axhline(y=0, c='k', linewidth=0.5)

    plt.gca().fill_between(x, m, where=(m >= thr), color='#00ff00')
    plt.gca().fill_between(x, m, where=(m <= -thr), color='#ff0000')
    plt.gca().fill_between(x, m, where=np.bitwise_and(m >= -thr, m <= thr), color='#ffff00')

    plt.ylabel("Margin")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_margins(margins: np.ndarray, outpath: str, title: str, unsure_thr: float = 0.3):

    plot_margin_ranking(margins=margins, outpath=outpath, title=title, thr=unsure_thr)
