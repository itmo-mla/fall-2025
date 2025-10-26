"""Главный сценарий: запускает PCA, рисует графики, вычисляет эффективную размерность."""
import os
import numpy as np
from pca_svd import PCA_SVD
from utils import load_regression_dataset, ensure_dir, plot_scree, plot_cumulative_variance, plot_reconstruction_errors


RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
ensure_dir(RESULTS)


def main():
    X, y, feature_names = load_regression_dataset()
    pca = PCA_SVD()
    pca.fit(X)

    # Scree
    plot_scree(pca.singular_values_, savepath=os.path.join(RESULTS, 'scree.png'))

    # Cumulative
    plot_cumulative_variance(pca.explained_variance_ratio_, savepath=os.path.join(RESULTS, 'cumulative_variance.png'))

    # effective dim for eps=0.05 и 0.01
    for eps in [0.05, 0.01]:
        m = pca.effective_dim(eps=eps)
        print(f"Effective dimension for eps={eps}: m={m}")

    # Reconstruction error vs m
    ms = list(range(1, pca.n_features_ + 1))
    errors = [pca.reconstruction_error(X, m=m) for m in ms]
    plot_reconstruction_errors(ms, errors, savepath=os.path.join(RESULTS, 'reconstruction_vs_m.png'))

    # Save a small CSV with explained variance
    import pandas as pd
    df = pd.DataFrame({
        'component': list(range(1, len(pca.explained_variance_)+1)),
        'explained_variance': pca.explained_variance_,
        'explained_variance_ratio': pca.explained_variance_ratio_
    })
    df.to_csv(os.path.join(RESULTS, 'explained_variance.csv'), index=False)

    print('Done. Results saved to results/ directory.')


if __name__ == '__main__':
    main()