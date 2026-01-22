import numpy as np
from sklearn.datasets import load_diabetes


def load_diabetes_regression(
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Load the Diabetes regression dataset and split it into train/val/test parts.

    Only sklearn.datasets is used here, as required in the task.

    Parameters
    ----------
    test_size : float
        Fraction of the whole dataset to use as test set.
    val_size : float
        Fraction of the remaining (non-test) data to use as validation set.
    random_state : int
        Seed for the random number generator.

    Returns
    -------
    dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test,
        mean, std, feature_names
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1).")
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be in (0, 1).")

    X, y = load_diabetes(return_X_y=True)
    feature_names = load_diabetes().feature_names  # type: ignore[attr-defined]

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_samples = X.shape[0]
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    X = X[indices]
    y = y[indices]

    n_test = int(n_samples * test_size)
    n_train_val = n_samples - n_test
    n_val = int(n_train_val * val_size)
    n_train = n_train_val - n_val

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    # Standardize features using training statistics
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    std[std == 0.0] = 1.0

    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    X_test_std = (X_test - mean) / std

    return {
        "X_train": X_train_std,
        "y_train": y_train,
        "X_val": X_val_std,
        "y_val": y_val,
        "X_test": X_test_std,
        "y_test": y_test,
        "mean": mean,
        "std": std,
        "feature_names": feature_names,
    }

