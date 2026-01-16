import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from model import LinearClassifier


SPECIES_TO_ID = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2,
}


def load_iris_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    # cleanup potential empty last row
    df = df.dropna(how="any")
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy(dtype=float)
    y = df["species"].map(SPECIES_TO_ID).to_numpy(dtype=int)
    return X, y


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.asarray(y).astype(int)
    Y = np.zeros((y.shape[0], n_classes), dtype=float)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    test_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train : n_train + n_val])
        test_idx.append(idx[n_train + n_val :])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])


@dataclass
class StandardScaler:
    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "StandardScaler":
        mean_ = X.mean(axis=0)
        std_ = X.std(axis=0)
        std_ = np.where(std_ == 0.0, 1.0, std_)
        return cls(mean_=mean_, std_=std_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def grad_check_mse(clf: LinearClassifier, X: np.ndarray, Y: np.ndarray, l2: float = 0.0, eps: float = 1e-6) -> float:
    """
    Numerical gradient check (finite differences). Returns max absolute error.
    """
    W0 = clf.W.copy()
    g_analytical = clf.grad_mse(X, Y, l2_lambda=l2)

    g_num = np.zeros_like(W0)
    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            Wp = W0.copy()
            Wm = W0.copy()
            Wp[i, j] += eps
            Wm[i, j] -= eps
            clf.W = Wp
            lp = clf.loss_mse(X, Y, l2_lambda=l2)
            clf.W = Wm
            lm = clf.loss_mse(X, Y, l2_lambda=l2)
            g_num[i, j] = (lp - lm) / (2 * eps)
    clf.W = W0
    return float(np.max(np.abs(g_num - g_analytical)))


def maybe_plot_margin_hist(margins: np.ndarray, out_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plt.figure(figsize=(7, 4))
    plt.hist(margins, bins=25, alpha=0.85)
    plt.title(title)
    plt.xlabel("margin = s_true - max_other")
    plt.ylabel("count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def run():
    parser = argparse.ArgumentParser(description="Lab-01: Linear classification on Iris (numpy-only model)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--multistart", type=int, default=15, help="number of random initializations (best on val)")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    csv_path = here / "iris.csv"
    df = pd.read_csv(csv_path).dropna(how="any")
    X, y = load_iris_csv(csv_path)
    n_classes = len(np.unique(y))

    # --- quick dataset analysis with pandas ---
    print("\nPandas EDA:")
    print(df.describe(include="all"))
    print("\nClass distribution:\n", df["species"].value_counts())

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = stratified_split(X, y, seed=args.seed)
    scaler = StandardScaler.fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    Y_tr = one_hot(y_tr, n_classes)
    Y_va = one_hot(y_va, n_classes)
    Y_te = one_hot(y_te, n_classes)

    print("Dataset:", csv_path)
    print("Shapes:", {"train": X_tr.shape, "val": X_va.shape, "test": X_te.shape})

    # --- 2) margin computation + visualization (before training) ---
    clf0 = LinearClassifier(n_features=X_tr.shape[1], n_classes=n_classes, seed=args.seed)
    m0 = clf0.margin(X_tr, y_tr)
    print("\nInitial margin stats (train):", {"min": float(m0.min()), "mean": float(m0.mean()), "max": float(m0.max())})
    maybe_plot_margin_hist(m0, here.parent / "margin_hist_before.png", "Train margin distribution (before training)")

    # --- 3) gradient computation (sanity check) ---
    err = grad_check_mse(clf0, X_tr[:10], Y_tr[:10], l2=args.l2)
    print("Gradient check (max abs err, first 10 train samples):", err)

    # --- 9.1) correlation init + SGD(momentum) + L2 ---
    clf_corr = LinearClassifier(n_features=X_tr.shape[1], n_classes=n_classes, seed=args.seed)
    clf_corr.init_correlation(X_tr, y_tr)
    hist_corr = clf_corr.fit_sgd_momentum(
        X_tr,
        Y_tr,
        y_tr,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        l2_lambda=args.l2,
        order="random",
        seed=args.seed,
    )
    tr_acc = accuracy(y_tr, clf_corr.predict(X_tr))
    va_acc = accuracy(y_va, clf_corr.predict(X_va))
    te_acc = accuracy(y_te, clf_corr.predict(X_te))
    print("\n[SGD+momentum+L2 | correlation init]")
    print("acc:", {"train": tr_acc, "val": va_acc, "test": te_acc})
    models = {"corr_init_sgd": clf_corr}
    model_scores = {"corr_init_sgd": {"val_acc": va_acc, "test_acc": te_acc}}

    # --- 9.2) random init + multistart (pick best on val) ---
    best = None
    best_va = -1.0
    for k in range(int(args.multistart)):
        clf_k = LinearClassifier(n_features=X_tr.shape[1], n_classes=n_classes, seed=args.seed + 1000 + k)
        clf_k.fit_sgd_momentum(
            X_tr,
            Y_tr,
            y_tr,
            epochs=args.epochs,
            lr=args.lr,
            momentum=args.momentum,
            l2_lambda=args.l2,
            order="random",
            seed=args.seed + 2000 + k,
        )
        va_k = accuracy(y_va, clf_k.predict(X_va))
        if va_k > best_va:
            best_va = va_k
            best = clf_k.W.copy()
    clf_ms = LinearClassifier(n_features=X_tr.shape[1], n_classes=n_classes, seed=args.seed)
    clf_ms.W = best
    print("\n[SGD+momentum+L2 | random init | multistart]")
    ms_te = accuracy(y_te, clf_ms.predict(X_te))
    print("best val acc:", best_va, "test acc:", ms_te)
    models["multistart_sgd"] = clf_ms
    model_scores["multistart_sgd"] = {"val_acc": float(best_va), "test_acc": float(ms_te)}

    # --- 8 + 9.3) presentation by |margin| (hard samples first) ---
    clf_margin = LinearClassifier(n_features=X_tr.shape[1], n_classes=n_classes, seed=args.seed)
    hist_margin = clf_margin.fit_sgd_momentum(
        X_tr,
        Y_tr,
        y_tr,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        l2_lambda=args.l2,
        order="margin_abs",
        seed=args.seed,
    )
    print("\n[SGD+momentum+L2 | order by |margin|]")
    margin_tr = accuracy(y_tr, clf_margin.predict(X_tr))
    margin_va = accuracy(y_va, clf_margin.predict(X_va))
    margin_te = accuracy(y_te, clf_margin.predict(X_te))
    print("acc:", {"train": margin_tr, "val": margin_va, "test": margin_te})
    models["margin_order_sgd"] = clf_margin
    model_scores["margin_order_sgd"] = {"val_acc": float(margin_va), "test_acc": float(margin_te)}

    # --- 7) steepest gradient descent (full-batch) ---
    clf_sd = LinearClassifier(n_features=X_tr.shape[1], n_classes=n_classes, seed=args.seed)
    sd_hist = clf_sd.fit_steepest_descent(X_tr, Y_tr, max_iters=250, lr0=1.0, l2_lambda=args.l2)
    sd_va = accuracy(y_va, clf_sd.predict(X_va))
    sd_te = accuracy(y_te, clf_sd.predict(X_te))
    print("\n[Steepest descent + Armijo line search | full-batch]")
    print("final train loss:", sd_hist["loss"][-1], "val acc:", sd_va, "test acc:", sd_te)
    models["steepest_descent"] = clf_sd
    model_scores["steepest_descent"] = {"val_acc": float(sd_va), "test_acc": float(sd_te)}

    # --- choose best model by validation accuracy ---
    best_name = max(model_scores.keys(), key=lambda k: model_scores[k]["val_acc"])
    best_model = models[best_name]
    print("\nBest model by val acc:", best_name, model_scores[best_name])

    # --- margin after training (best model) ---
    m_after = best_model.margin(X_tr, y_tr)
    print("\nFinal margin stats (train, best):", {"min": float(m_after.min()), "mean": float(m_after.mean()), "max": float(m_after.max())})
    maybe_plot_margin_hist(m_after, here.parent / "margin_hist_after.png", f"Train margin distribution (after training, best={best_name})")

    # --- evaluation artifacts ---
    y_pred = best_model.predict(X_te)
    cm = confusion_matrix(y_te, y_pred, n_classes=n_classes)
    print("\nConfusion matrix (best, test):\n", cm)

    # --- optional baseline (sklearn) ---
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        LogisticRegression = None

    if LogisticRegression is not None:
        lr = LogisticRegression(max_iter=2000, multi_class="auto")
        lr.fit(X_tr, y_tr)
        print("\n[Baseline: sklearn LogisticRegression]")
        print("test acc:", accuracy(y_te, lr.predict(X_te)))


if __name__ == "__main__":
    run()
