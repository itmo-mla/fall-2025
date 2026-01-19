import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from model import predict, solve_svm_dual


SPECIES_TO_ID = {"setosa": 0, "versicolor": 1, "virginica": 2}


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


def stratified_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.25, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_ratio))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def maybe_plot_2d_decision(
    X: np.ndarray,
    y: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sv,
    out_path: Path,
    title: str,
    kernel: str,
    gamma: float,
    degree: int,
    coef0: float,
):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = predict(X_train, y_train, sv, grid, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, zz, levels=[-2, 0, 2], alpha=0.2)
    colors = {1.0: "tab:blue", -1.0: "tab:orange"}
    for cls in [-1.0, 1.0]:
        mask = (y == cls)
        plt.scatter(X[mask, 0], X[mask, 1], s=35, c=colors[cls], label=f"class {int(cls)}", alpha=0.8)

    # highlight support vectors (train only)
    if hasattr(sv, "support_idx"):
        sv_pts = X_train[sv.support_idx]
        plt.scatter(sv_pts[:, 0], sv_pts[:, 1], s=120, facecolors="none", edgecolors="black", linewidths=1.5, label="SV")

    plt.title(title)
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def run():
    parser = argparse.ArgumentParser(description="Lab-03: SVM (dual via scipy.optimize.minimize) + kernels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.25)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--kernel", type=str, default="linear", choices=["linear", "rbf", "poly"])
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--coef0", type=float, default=1.0)
    parser.add_argument("--classes", type=str, default="versicolor,virginica", help="two iris classes")
    parser.add_argument("--features", type=str, default="petal_length,petal_width", help="two features for visualization")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    csv_path = here / "iris.csv"

    df = pd.read_csv(csv_path).dropna(how="any")
    classes = [c.strip() for c in str(args.classes).split(",")]
    if len(classes) != 2:
        raise ValueError("--classes must contain exactly two class names")

    df = df[df["species"].isin(classes)].copy()
    df["y"] = np.where(df["species"] == classes[1], 1.0, -1.0)

    feats = [f.strip() for f in str(args.features).split(",")]
    if len(feats) != 2:
        raise ValueError("--features must contain exactly two feature names (for 2D visualization)")

    X = df[feats].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    print("Dataset:", csv_path)
    print("Binary classes:", classes, "-> y=+1 is", classes[1])
    print("Features:", feats)
    print("\nPandas EDA:")
    print(df[feats + ["species"]].describe(include="all"))
    print("\nClass distribution:\n", df["species"].value_counts())

    (X_tr, y_tr), (X_te, y_te) = stratified_split(X, y, test_ratio=float(args.test_ratio), seed=int(args.seed))
    scaler = StandardScaler.fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    sv = solve_svm_dual(
        X_tr,
        y_tr,
        C=float(args.C),
        kernel=str(args.kernel),
        gamma=float(args.gamma),
        degree=int(args.degree),
        coef0=float(args.coef0),
    )

    yhat_tr = predict(X_tr, y_tr, sv, X_tr, kernel=args.kernel, gamma=args.gamma, degree=args.degree, coef0=args.coef0)
    yhat_te = predict(X_tr, y_tr, sv, X_te, kernel=args.kernel, gamma=args.gamma, degree=args.degree, coef0=args.coef0)

    print("\nSupport vectors:", len(sv.support_idx), "of", len(X_tr))
    print("Train acc:", accuracy(y_tr, yhat_tr))
    print("Test  acc:", accuracy(y_te, yhat_te))

    out_img = here / f"svm_{args.kernel}_C{args.C}.png"
    maybe_plot_2d_decision(
        X=np.vstack([X_tr, X_te]),
        y=np.hstack([y_tr, y_te]),
        X_train=X_tr,
        y_train=y_tr,
        sv=sv,
        out_path=out_img,
        title=f"SVM ({args.kernel}) on Iris: {classes[0]} vs {classes[1]}",
        kernel=args.kernel,
        gamma=args.gamma,
        degree=args.degree,
        coef0=args.coef0,
    )

    # sklearn baseline (optional)
    try:
        from sklearn.svm import SVC
    except Exception:
        SVC = None

    if SVC is not None:
        if args.kernel == "linear":
            sk = SVC(C=args.C, kernel="linear")
        elif args.kernel == "rbf":
            sk = SVC(C=args.C, kernel="rbf", gamma=args.gamma)
        else:
            sk = SVC(C=args.C, kernel="poly", degree=args.degree, gamma=args.gamma, coef0=args.coef0)
        sk.fit(X_tr, y_tr)
        acc_sk = accuracy(y_te, sk.predict(X_te))
        print("\n[sklearn SVC baseline]")
        print("test acc:", acc_sk)
        print("n_support:", int(np.sum(sk.n_support_)))


if __name__ == "__main__":
    run()
