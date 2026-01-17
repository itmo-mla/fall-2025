import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from utils import *


def convergence_plot(res_newton, res_irls, image_path: str = "convergence.png", dpi: int = 200):
    plt.figure()
    plt.plot(res_newton.history["iter"], res_newton.history["nll"], marker="o", label="Newton")
    plt.plot(res_irls.history["iter"], res_irls.history["nll"], marker="s", label="IRLS")
    plt.xlabel("Iteration")
    plt.ylabel("Negative log-likelihood")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_path, dpi=dpi)
    plt.close()

def roc_plot(res_newton, res_irls, w_ref, X_test, y_test, image_path: str = "roc.png", dpi: int = 200):
    plt.figure()
    for name, w in [("Newton", res_newton.w), ("IRLS", res_irls.w), ("sklearn", w_ref)]:
        p_test = sigmoid(X_test @ w)
        fpr, tpr, _ = roc_curve(y_test, p_test)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_path, dpi=dpi)
    plt.close()