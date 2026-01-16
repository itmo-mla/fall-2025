import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from model import (
    KNNParzenClassifier,
    condensed_nearest_neighbor,
    condensed_parzen_knn,
    loo_cv_risk_knn_parzen,
    pca_2d,
    stolp_prototypes,
)


SPECIES_TO_ID = {"setosa": 0, "versicolor": 1, "virginica": 2}
ID_TO_SPECIES = {v: k for k, v in SPECIES_TO_ID.items()}


def load_iris(csv_path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(csv_path).dropna(how="any")
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy(dtype=float)
    y = df["species"].map(SPECIES_TO_ID).to_numpy(dtype=int)
    return X, y, df


def stratified_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 42):
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


class StandardScaler:
    def __init__(self, mean_: np.ndarray, std_: np.ndarray):
        self.mean_ = mean_
        self.std_ = std_

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


def maybe_plot_risk_curve(k_values: list[int], risks: list[float], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, risks, marker="o")
    plt.title("LOO empirical risk vs k (Parzen KNN, variable bandwidth)")
    plt.xlabel("k")
    plt.ylabel("risk (LOO error rate)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def maybe_plot_prototypes_pca(X: np.ndarray, y: np.ndarray, proto_idx: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    Z = pca_2d(X)
    plt.figure(figsize=(7, 5))
    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
    for c in np.unique(y):
        mask = (y == c)
        plt.scatter(Z[mask, 0], Z[mask, 1], s=20, alpha=0.55, c=colors[int(c)], label=ID_TO_SPECIES[int(c)])

    # highlight prototypes
    plt.scatter(
        Z[proto_idx, 0],
        Z[proto_idx, 1],
        s=90,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label=f"prototypes (n={len(proto_idx)})",
    )
    plt.title("Condensed NN prototypes (PCA 2D projection)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def maybe_plot_prototypes_pca_stolp(X: np.ndarray, y: np.ndarray, proto_idx: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    Z = pca_2d(X)
    plt.figure(figsize=(7, 5))
    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
    for c in np.unique(y):
        mask = (y == c)
        plt.scatter(Z[mask, 0], Z[mask, 1], s=20, alpha=0.55, c=colors[int(c)], label=ID_TO_SPECIES[int(c)])

    plt.scatter(
        Z[proto_idx, 0],
        Z[proto_idx, 1],
        s=90,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label=f"STOLP prototypes (n={len(proto_idx)})",
    )
    plt.title("STOLP prototypes (PCA 2D projection)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def run():
    parser = argparse.ArgumentParser(description="Lab-02: Metric classification (KNN + Parzen, LOO k selection, prototypes)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmax", type=int, default=30)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    csv_path = here / "iris.csv"

    X, y, df = load_iris(csv_path)
    print("Dataset:", csv_path)
    print("\nPandas EDA:")
    print(df.describe(include="all"))
    print("\nClass distribution:\n", df["species"].value_counts())

    (X_tr, y_tr), (X_te, y_te) = stratified_split(X, y, test_ratio=float(args.test_ratio), seed=int(args.seed))
    scaler = StandardScaler.fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # --- LOO selection of k on train ---
    k_values = list(range(1, int(args.kmax) + 1))
    loo = loo_cv_risk_knn_parzen(X_tr_s, y_tr, k_values=k_values, show_progress=True)
    best_i = int(np.argmin(loo["risk"]))
    best_k = int(loo["k"][best_i])
    best_risk = float(loo["risk"][best_i])
    print("\nLOO best k:", best_k, "risk:", best_risk, "acc:", 1.0 - best_risk)

    maybe_plot_risk_curve(loo["k"], loo["risk"], here / "risk_vs_k.png")

    # --- evaluate our implementation ---
    clf = KNNParzenClassifier(k=best_k).fit(X_tr_s, y_tr)
    y_pred = clf.predict(X_te_s)
    acc_test = accuracy(y_te, y_pred)
    print("\n[Our Parzen-KNN] test acc:", acc_test)

    # --- baseline sklearn KNN (optional) ---
    try:
        from sklearn.neighbors import KNeighborsClassifier
    except Exception:
        KNeighborsClassifier = None

    if KNeighborsClassifier is not None:
        sk = KNeighborsClassifier(n_neighbors=best_k, weights="distance", metric="minkowski", p=2)
        sk.fit(X_tr_s, y_tr)
        acc_sk = accuracy(y_te, sk.predict(X_te_s))
        print("[Baseline sklearn KNN(weights=distance)] test acc:", acc_sk)

    # --- prototype selection (Condensed NN) ---
    proto_idx = condensed_nearest_neighbor(X_tr_s, y_tr, seed=int(args.seed))
    Xp = X_tr_s[proto_idx]
    yp = y_tr[proto_idx]
    print("\n[Prototype selection: CNN] prototypes:", len(proto_idx), "of", len(X_tr_s))

    # --- Variant A (as in classic assignment): CNN prototypes + Parzen-KNN trained on them ---
    # NOTE: for Parzen-KNN with variable bandwidth, k must be re-tuned after strong compression.
    kmax_p = max(1, min(int(args.kmax), max(1, len(Xp) - 1)))
    loo_p = loo_cv_risk_knn_parzen(Xp, yp, k_values=list(range(1, kmax_p + 1)), show_progress=False)
    best_k_p = int(loo_p["k"][int(np.argmin(loo_p["risk"]))])
    clf_p = KNNParzenClassifier(k=best_k_p).fit(Xp, yp)
    acc_p = accuracy(y_te, clf_p.predict(X_te_s))
    print("[Variant A: CNN prototypes -> Parzen-KNN] best k:", best_k_p, "test acc:", acc_p)

    # CNN is fundamentally tailored to preserve 1-NN; show that baseline too.
    clf_cnn_1nn = KNNParzenClassifier(k=1).fit(Xp, yp)
    acc_cnn_1nn = accuracy(y_te, clf_cnn_1nn.predict(X_te_s))
    print("[CNN baseline: 1-NN on CNN prototypes] test acc:", acc_cnn_1nn)

    # --- Variant B (improved): condense under Parzen-KNN rule at fixed best_k (Parzen-CNN) ---
    parzen_idx = condensed_parzen_knn(X_tr_s, y_tr, k=best_k, seed=int(args.seed))
    Xq = X_tr_s[parzen_idx]
    yq = y_tr[parzen_idx]
    clf_q = KNNParzenClassifier(k=best_k).fit(Xq, yq)
    acc_q = accuracy(y_te, clf_q.predict(X_te_s))
    print("[Variant B: Parzen-CNN prototypes -> Parzen-KNN] prototypes:", len(parzen_idx), "of", len(X_tr_s), "| k:", best_k, "| test acc:", acc_q)

    maybe_plot_prototypes_pca(X_tr_s, y_tr, proto_idx, here / "prototypes_pca.png")

    # --- prototype selection (STOLP) ---
    stolp_idx = stolp_prototypes(X_tr_s, y_tr, remove_bad=True, r_scale=1.0, seed=int(args.seed))
    if len(stolp_idx) > 0:
        Xs = X_tr_s[stolp_idx]
        ys = y_tr[stolp_idx]
        kmax_s = max(1, min(int(args.kmax), max(1, len(Xs) - 1)))
        loo_s = loo_cv_risk_knn_parzen(Xs, ys, k_values=list(range(1, kmax_s + 1)), show_progress=False)
        best_k_s = int(loo_s["k"][int(np.argmin(loo_s["risk"]))])
        clf_s = KNNParzenClassifier(k=best_k_s).fit(Xs, ys)
        acc_s = accuracy(y_te, clf_s.predict(X_te_s))
        print("\n[Prototype selection: STOLP] prototypes:", len(stolp_idx), "of", len(X_tr_s))
        print("[Our Parzen-KNN + STOLP prototypes] best k:", best_k_s, "test acc:", acc_s)
        maybe_plot_prototypes_pca_stolp(X_tr_s, y_tr, stolp_idx, here / "stolp_prototypes_pca.png")


if __name__ == "__main__":
    run()
