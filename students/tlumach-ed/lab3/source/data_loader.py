import numpy as np
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split

def load_dataset(
    dataset_type="classification",
    n_samples=500,
    n_features=2,
    random_state=42
):
    if dataset_type == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=2,
            n_redundant=0,
            n_classes=2,
            n_clusters_per_class=1,
            flip_y=0.1,
            random_state=random_state
        )

    elif dataset_type == "moons":
        X, y = make_moons(
            n_samples=n_samples,
            noise=0.2,
            random_state=random_state
        )

    else:
        raise ValueError("Unknown dataset_type")

    y = 2 * y - 1  # {0,1} → {-1,+1}
    return train_test_split(X, y, test_size=0.3, random_state=random_state)
