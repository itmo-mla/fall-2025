from __future__ import annotations

import argparse
import numpy as np

from sklearn.decomposition import PCA as SklearnPCA

from dataset import load_dataset
from pca_svd import pca_via_svd, effective_dimensionality
from metrics import reconstruction_mse, subspace_distance, max_abs_diff
from plots import (
    plot_explained_variance_ratio,
    plot_cumulative_explained_variance,
    plot_reconstruction_error_curve,
    plot_cumulative_variance_comparison,
)
from utils import ensure_dir, standardize_fit, standardize_transform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab 4: PCA via SVD (no sklearn PCA for own impl)")
    p.add_argument("--dataset", type=str, default="diabetes", choices=["diabetes", "openml:house_prices"])
    p.add_argument("--standardize", action="store_true", help="Standardize features before PCA")
    p.add_argument("--threshold", type=float, default=0.95, help="Threshold for effective dimensionality")
    p.add_argument("--outdir", type=str, default="outputs", help="Directory to save plots")
    p.add_argument("--max_components", type=int, default=10, help="Max components for error curve (<= n_features)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)

    ds = load_dataset(args.dataset)
    X = ds.X
    y = ds.y  # not used in PCA itself, but dataset is regression-type per task

    # Preprocess
    if args.standardize:
        params = standardize_fit(X)
        Xp = standardize_transform(X, params)
        prep_title = " (standardized)"
    else:
        Xp = X
        prep_title = " (centered only)"

    # Our PCA
    pca_res = pca_via_svd(Xp)
    evr = pca_res.explained_variance_ratio_

    k_eff = effective_dimensionality(evr, threshold=args.threshold)

    # Reference PCA (allowed as baseline)
    sk_pca = SklearnPCA(svd_solver="full")  # full SVD
    sk_pca.fit(Xp)

    plot_cumulative_variance_comparison(
    evr_ours=evr,
    evr_ref=sk_pca.explained_variance_ratio_,
    out_path=f"{args.outdir}/cumulative_comparison.png",
    title="Cumulative Explained Variance (Our PCA vs sklearn)" + prep_title,
    )


    # Compare explained variance ratio
    k_total = min(len(sk_pca.explained_variance_ratio_), len(evr))
    evr_diff = max_abs_diff(evr[:k_total], sk_pca.explained_variance_ratio_[:k_total])

    # Compare subspace for first k_eff components
    W_ours = pca_res.components_[:k_eff]
    W_ref = sk_pca.components_[:k_eff]
    # Both should be orthonormal; compare projection matrices distance (sign-invariant)
    sub_dist = subspace_distance(W_ours, W_ref)

    # Reconstruction error curve using our PCA
    max_k = min(args.max_components, Xp.shape[1], len(evr))
    errors = np.zeros(max_k, dtype=float)
    for k in range(1, max_k + 1):
        Z = pca_res.transform(Xp, n_components=k)
        X_rec = pca_res.inverse_transform(Z, n_components=k)
        errors[k - 1] = reconstruction_mse(Xp, X_rec)

    # Save plots
    plot_explained_variance_ratio(
        evr[:max_k],
        out_path=f"{args.outdir}/explained_variance_ratio.png",
        title="Explained Variance Ratio" + prep_title,
    )
    plot_cumulative_explained_variance(
        evr,
        threshold=args.threshold,
        out_path=f"{args.outdir}/cumulative_explained_variance.png",
        title="Cumulative Explained Variance" + prep_title,
    )
    plot_reconstruction_error_curve(
        errors,
        out_path=f"{args.outdir}/reconstruction_mse_curve.png",
        title="Reconstruction MSE vs #Components" + prep_title,
    )

    # Print key results
    print("=== Lab 4: PCA via SVD ===")
    print(f"Dataset: {ds.name}")
    print(f"X shape: {Xp.shape}")
    print(f"Standardize: {args.standardize}")
    print(f"Effective dimensionality (threshold={args.threshold:.2f}): k = {k_eff}")
    print()
    print("Equivalence check with sklearn PCA:")
    print(f"Max abs diff explained_variance_ratio (first {k_total} comps): {evr_diff:.6e}")
    print(f"Subspace distance (k={k_eff}) via projection Frobenius norm: {sub_dist:.6e}")
    print()
    print(f"Plots saved to: {args.outdir}/")
    print("- explained_variance_ratio.png")
    print("- cumulative_explained_variance.png")
    print("- reconstruction_mse_curve.png")

    with open(f"{args.outdir}/results.txt", "w", encoding="utf-8") as f:
        f.write("=== Lab 4: PCA via SVD ===\n")
        f.write(f"Dataset: {ds.name}\n")
        f.write(f"X shape: {Xp.shape}\n")
        f.write(f"Standardize: {args.standardize}\n")
        f.write(f"Effective dimensionality (threshold={args.threshold:.2f}): k = {k_eff}\n\n")
        f.write("Equivalence check with sklearn PCA:\n")
        f.write(f"Max abs diff explained_variance_ratio (first {k_total} comps): {evr_diff:.6e}\n")
        f.write(f"Subspace distance (k={k_eff}) via projection Frobenius norm: {sub_dist:.6e}\n")


if __name__ == "__main__":
    main()
