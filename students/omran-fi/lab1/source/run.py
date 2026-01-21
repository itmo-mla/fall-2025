import argparse
import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

from source.download_kaggle import download_heart_dataset
from source.dataset_heart import load_heart_csv
from source.config import FitConfig
from source.linear_classifier import LinearClassifier, margins_all
from source.metrics import accuracy, precision_recall_f1
from source.plots import ensure_dir, plot_training, plot_margins


def correlation_init(X: np.ndarray, y: np.ndarray) -> np.ndarray:
   
    yv = y.reshape(-1, 1)
    w = []
    for j in range(X.shape[1]):
        xj = X[:, j].reshape(-1, 1)
        denom = float((xj.T @ xj)[0, 0])
        wj = float((yv.T @ xj)[0, 0]) / denom if denom > 1e-30 else 0.0
        w.append(wj)
    return np.array(w, dtype=float).reshape(-1, 1)


def evaluate(tag: str, y_true: np.ndarray, y_pred: np.ndarray):
    acc = accuracy(y_true, y_pred)
    prec, rec, f1 = precision_recall_f1(y_true, y_pred)
    print(f"\n=== {tag} ===")
    print(f"accuracy:  {acc:.4f}")
    print(f"precision: {prec:.4f} | recall: {rec:.4f} | f1: {f1:.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    return acc


def run_one(tag: str, X_train, y_train, X_test, y_test, cfg: FitConfig, init_w, outdir: str):
    clf = LinearClassifier(n_features=X_train.shape[1])
    if init_w is None:
        clf.init_random(cfg.seed)
    else:
        clf.init_with(init_w)

    clf.fit(X_train, y_train, cfg)

    y_pred = clf.predict(X_test)
    acc = evaluate(tag, y_test, y_pred)

    plot_training(
        clf.loss_history, clf.Q_history,
        outpath=os.path.join(outdir, f"{tag}_training.png"),
        title=f"Training curves — {tag}"
    )

    m = margins_all(clf.w, X_test, y_test)
    plot_margins(
        m,
        outpath=os.path.join(outdir, f"{tag}_margins.png"),
        title=f"Margins (test) — {tag}",
        unsure_thr=0.3
    )

    return clf, acc


def baseline_sklearn(X_train, y_train, X_test, y_test):
    print("\n=== Baseline: sklearn SGDClassifier ===")
    model = SGDClassifier(random_state=42)
    model.fit(X_train, y_train.reshape(-1))
    y_pred = model.predict(X_test).reshape(-1, 1)
    evaluate("sklearn_baseline", y_test, y_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--multistart", type=int, default=15)

    # Kaggle download options
    ap.add_argument("--data-dir", type=str, default="data", help="Where to download/store the dataset locally")
    ap.add_argument("--download-kaggle", action="store_true", help="Download dataset from Kaggle at runtime")

    args = ap.parse_args()
    ensure_dir(args.outdir)

    # Download dataset if requested (recommended)
    if args.download_kaggle:
        csv_path = download_heart_dataset(data_dir=args.data_dir)
    else:
        # If not downloading, expect the user has already placed heart.csv in data-dir
        csv_path = os.path.join(args.data_dir, "heart.csv")

    X_train, X_test, y_train, y_test = load_heart_csv(csv_path, seed=args.seed)

    # 1) SGD without momentum
    cfg_sgd = FitConfig(n_iter=10000, lr=1e-2, l2=0.5, use_momentum=False,
                        use_fastest_step=False, use_margin_sampling=False, seed=args.seed)
    run_one("sgd_l2", X_train, y_train, X_test, y_test, cfg_sgd, None, args.outdir)

    # 2) Nesterov momentum
    cfg_nest = FitConfig(n_iter=10000, lr=1e-2, l2=0.5, use_momentum=True, gamma=0.9, nesterov=True,
                         use_fastest_step=False, use_margin_sampling=False, seed=args.seed)
    run_one("nesterov_l2", X_train, y_train, X_test, y_test, cfg_nest, None, args.outdir)

    cfg_fast = FitConfig(
        n_iter=10000,
        lr=1e-3,
        lambda_q=0.01,
        l2=0.5,
        use_momentum=True,
        gamma=0.9,
        nesterov=True,
        use_fastest_step=True,
        seed=args.seed
    )

    run_one("nesterov_fastest", X_train, y_train, X_test, y_test, cfg_fast, None, args.outdir)

    # 4) Correlation init
    w_corr = correlation_init(X_train, y_train)
    cfg_corr = FitConfig(n_iter=10000, lr=1e-2, l2=0.5, use_momentum=True, gamma=0.9, nesterov=True,
                         use_fastest_step=False, use_margin_sampling=False, seed=args.seed)
    run_one("corr_init", X_train, y_train, X_test, y_test, cfg_corr, w_corr, args.outdir)

    # 5) Multistart
    best_acc = -1.0
    best_w = None

    cfg_ms = FitConfig(n_iter=10000, lr=1e-2, l2=0.5, use_momentum=True, gamma=0.9, nesterov=True,
                       use_fastest_step=False, use_margin_sampling=False, seed=args.seed)

    for k in range(args.multistart):
        seed_k = args.seed + 1000 + k
        cfg_ms.seed = seed_k

        clf = LinearClassifier(n_features=X_train.shape[1])
        clf.init_random(seed_k)
        clf.fit(X_train, y_train, cfg_ms)
        y_pred = clf.predict(X_test)
        acc = accuracy(y_test, y_pred)
        print(f"multistart #{k:02d} seed={seed_k} acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_w = clf.w.copy()

    print(f"\nBest multistart accuracy: {best_acc:.4f}")

    # 6) Random vs margin sampling (same init)
    cfg_rand = FitConfig(n_iter=20000, lr=1e-2, l2=0.5, use_momentum=True, gamma=0.9, nesterov=True,
                         use_fastest_step=False, use_margin_sampling=False, seed=args.seed)
    run_one("random_sampling", X_train, y_train, X_test, y_test, cfg_rand, best_w, args.outdir)

    cfg_marg = FitConfig(
    n_iter=20000,
    lr=1e-3,                 
    lambda_q=0.01,
    l2=0.5,                 
    use_momentum=True,
    gamma=0.9,
    nesterov=True,

    use_margin_sampling=True,
    margin_warmup=5000,      
    margin_temperature=3.0,  
    eps_sampling=1e-6,      

    seed=args.seed,
    )


    run_one("margin_sampling", X_train, y_train, X_test, y_test, cfg_marg, best_w, args.outdir)

    # 7) Baseline
    baseline_sklearn(X_train, y_train, X_test, y_test)

    print(f"\nDone. Plots saved to: {args.outdir}/")
    print(f"Dataset CSV used: {csv_path}")


if __name__ == "__main__":
    main()
