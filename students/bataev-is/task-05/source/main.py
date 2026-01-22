import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def add_bias(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.hstack([np.ones((X.shape[0], 1), dtype=float), X])


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


def train_test_split_stratified(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.25, seed: int = 42):
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


def neg_log_likelihood(Xb: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 0.0) -> float:
    z = Xb @ w
    p = sigmoid(z)
    eps = 1e-12
    nll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    # do not regularize bias
    if l2 != 0.0:
        nll = nll + 0.5 * float(l2) * float(np.sum(w[1:] ** 2))
    return float(nll)


def newton_raphson_logreg(
    Xb: np.ndarray,
    y: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
    l2: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """
    Newton-Raphson for logistic regression (binary).
    Minimizes negative log-likelihood with tiny L2 for numerical stability.
    """
    n, d = Xb.shape
    w = np.zeros(d, dtype=float)
    hist = {"nll": [], "step_norm": []}

    I = np.eye(d)
    I[0, 0] = 0.0  # do not regularize bias

    for _ in range(int(max_iter)):
        z = Xb @ w
        p = sigmoid(z)

        # gradient of mean NLL: X^T (p - y)/n + l2*w
        g = (Xb.T @ (p - y)) / n + float(l2) * (I @ w)

        # Hessian: X^T R X / n + l2*I, where R = diag(p*(1-p))
        r = p * (1 - p)  # (n,)
        XR = Xb * r[:, None]  # (n,d)
        H = (Xb.T @ XR) / n + float(l2) * I

        # Newton step: H * delta = g
        delta = np.linalg.solve(H, g)
        w_new = w - delta

        hist["nll"].append(neg_log_likelihood(Xb, y, w, l2=l2))
        hist["step_norm"].append(float(np.linalg.norm(delta)))

        if np.linalg.norm(delta) < tol:
            w = w_new
            break
        w = w_new

    return w, hist


def irls_logreg(
    Xb: np.ndarray,
    y: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
    l2: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """
    IRLS for logistic regression.
    For logistic regression, IRLS is equivalent to Newton-Raphson.
    """
    n, d = Xb.shape
    w = np.zeros(d, dtype=float)
    hist = {"nll": [], "step_norm": []}

    I = np.eye(d)
    I[0, 0] = 0.0

    for _ in range(int(max_iter)):
        z = Xb @ w
        p = sigmoid(z)
        r = p * (1 - p)
        r = np.maximum(r, 1e-12)  # avoid zero weights

        # working response
        z_work = z + (y - p) / r

        # solve weighted least squares: (X^T W X + l2 I) w_new = X^T W z_work
        XW = Xb * r[:, None]
        A = (Xb.T @ XW) / n + float(l2) * I
        b = (Xb.T @ (r * z_work)) / n
        w_new = np.linalg.solve(A, b)

        delta = w_new - w
        hist["nll"].append(neg_log_likelihood(Xb, y, w, l2=l2))
        hist["step_norm"].append(float(np.linalg.norm(delta)))

        w = w_new
        if np.linalg.norm(delta) < tol:
            break

    return w, hist


def predict_proba(Xb: np.ndarray, w: np.ndarray) -> np.ndarray:
    return sigmoid(Xb @ w)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def maybe_plot_convergence(nll_a: list[float], nll_b: list[float], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(nll_a, marker="o", label="Newton-Raphson")
    plt.plot(nll_b, marker="o", label="IRLS")
    plt.title("Negative log-likelihood convergence")
    plt.xlabel("iteration")
    plt.ylabel("mean NLL")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def run():
    parser = argparse.ArgumentParser(description="Lab-05: Logistic regression via Newton-Raphson and IRLS (numpy core)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.25)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--l2", type=float, default=1e-6, help="tiny L2 for numerical stability (bias not regularized)")
    parser.add_argument("--classes", type=str, default="setosa,versicolor", help="two iris classes, comma-separated")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    csv_path = here / "source" / "iris.csv"
    df = pd.read_csv(csv_path).dropna(how="any")

    classes = [c.strip() for c in str(args.classes).split(",")]
    if len(classes) != 2:
        raise ValueError("--classes must contain exactly two class names (e.g. 'setosa,versicolor')")

    df = df[df["species"].isin(classes)].copy()
    df["y"] = (df["species"] == classes[1]).astype(int)

    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)

    print("Dataset:", csv_path)
    print("Binary classes:", classes, "-> y=1 is", classes[1])
    print("\nPandas EDA:")
    print(df.describe(include="all"))
    print("\nClass distribution:\n", df["species"].value_counts())

    (X_tr, y_tr), (X_te, y_te) = train_test_split_stratified(X, y, test_ratio=float(args.test_ratio), seed=int(args.seed))

    scaler = StandardScaler.fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    Xb_tr = add_bias(X_tr)
    Xb_te = add_bias(X_te)

    # Newton-Raphson
    w_nr, h_nr = newton_raphson_logreg(Xb_tr, y_tr, max_iter=args.max_iter, tol=args.tol, l2=args.l2)
    # IRLS
    w_irls, h_irls = irls_logreg(Xb_tr, y_tr, max_iter=args.max_iter, tol=args.tol, l2=args.l2)

    print("\nCoefficients (bias + 4 features):")
    print("w_newton:", w_nr)
    print("w_irls  :", w_irls)
    print("||w_newton - w_irls||:", float(np.linalg.norm(w_nr - w_irls)))

    # evaluation
    p_nr = predict_proba(Xb_te, w_nr)
    yhat_nr = (p_nr >= 0.5).astype(int)
    acc_nr = accuracy(y_te, yhat_nr)

    p_irls = predict_proba(Xb_te, w_irls)
    yhat_irls = (p_irls >= 0.5).astype(int)
    acc_irls = accuracy(y_te, yhat_irls)

    print("\nTest accuracy:")
    print("Newton-Raphson:", acc_nr)
    print("IRLS         :", acc_irls)

    maybe_plot_convergence(h_nr["nll"], h_irls["nll"], here / "nll_convergence.png")

    # sklearn baseline (optional)
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        LogisticRegression = None

    if LogisticRegression is not None:
        # sklearn regularizes by default; use very large C to approximate MLE
        sk = LogisticRegression(C=1e6, max_iter=5000, solver="lbfgs")
        sk.fit(X_tr, y_tr)
        acc_sk = accuracy(y_te, sk.predict(X_te))
        print("\n[sklearn LogisticRegression baseline]")
        print("test accuracy:", acc_sk)
        print("bias (intercept):", float(sk.intercept_[0]))
        print("weights:", sk.coef_.reshape(-1))


if __name__ == "__main__":
    run()
