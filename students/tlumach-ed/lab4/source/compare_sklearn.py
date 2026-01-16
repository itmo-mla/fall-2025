import numpy as np
from sklearn.decomposition import PCA as SKPCA
from utils import load_regression_dataset
from pca_svd import PCA_SVD


def comparable_up_to_sign(A, B, tol=1e-6):
    A = np.asarray(A)
    B = np.asarray(B)
    na = A.shape[1]
    nb = B.shape[1]
    used = set()
    for i in range(na):
        col = A[:, i]
        found = False
        for j in range(nb):
            if j in used:
                continue
            bcol = B[:, j]
            # normalise
            if np.allclose(col, 0):
                if np.allclose(bcol, 0):
                    found = True
                    used.add(j)
                    break
                else:
                    continue
            corr = np.dot(col, bcol) / (np.linalg.norm(col) * np.linalg.norm(bcol) + 1e-12)
            if abs(abs(corr) - 1.0) < 1e-6:
                found = True
                used.add(j)
                break
        if not found:
            return False
    return True


def main():
    X, y, _ = load_regression_dataset()
    ours = PCA_SVD()
    ours.fit(X)

    sk = SKPCA(svd_solver='full')
    sk.fit(X)

    print('Explained variance (ours)     :', np.round(ours.explained_variance_, 6))
    print('Explained variance (sklearn) :', np.round(sk.explained_variance_, 6))

    print('\nExplained variance ratio difference (L1):', np.abs(ours.explained_variance_ratio_ - sk.explained_variance_ratio_).sum())

    # сравним компоненты по направлению
    are_similar = comparable_up_to_sign(ours.components_, sk.components_.T)
    print('\nComponents equal up to sign & order?:', are_similar)

    # проверим восстановление X из m компонент
    for m in [2, 5, 10]:
        Z_ours = ours.transform(X, n_components=m)
        Xhat_ours = ours.inverse_transform(Z_ours)
        V = sk.components_.T[:, :m]
        Xc = X - X.mean(axis=0)
        Xhat_sk_manual = (Xc.dot(V)).dot(V.T) + X.mean(axis=0)

        mse_ours = ((X - Xhat_ours) ** 2).mean()
        mse_sk = ((X - Xhat_sk_manual) ** 2).mean()
        print(f"m={m}: MSE ours={mse_ours:.6e}, MSE sklearn={mse_sk:.6e}, diff={abs(mse_ours-mse_sk):.6e}")

if __name__ == '__main__':
    main()