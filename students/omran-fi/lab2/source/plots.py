import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def plot_empirical_risk(k_values, risks, title, out_dir="out", filename="loo_risk.png"):
    ensure_out_dir(out_dir)

    plt.figure(figsize=(9, 5))
    plt.plot(k_values, risks, marker="o", label="LOO empirical risk")
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Empirical risk (LOO error rate)")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_prototypes_pca(X, y, proto_idx, title, out_dir="out", filename="prototypes_pca.png"):
    ensure_out_dir(out_dir)

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(9, 6))
    for cls in sorted(set(y)):
        mask = (y == cls)
        plt.scatter(X2[mask, 0], X2[mask, 1], alpha=0.4, label=f"class {cls}")

    plt.scatter(
        X2[proto_idx, 0],
        X2[proto_idx, 1],
        s=120,
        alpha=0.9,
        marker="X",
        label=f"prototypes ({len(proto_idx)})",
    )

    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(cm, class_names, title, out_dir="out", filename="confusion_matrix.png"):
    """
    cm: 2D numpy array (confusion matrix)
    class_names: list of class labels (strings)
    """
    ensure_out_dir(out_dir)

    cm = np.asarray(cm)
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)


    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()

    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_pairplot(df, features, target, out_dir="out", filename="pairplot.png"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    os.makedirs(out_dir, exist_ok=True)

    sns.pairplot(df, vars=features, hue=target)
    plt.suptitle("Pairplot of selected features", y=1.02)

    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
