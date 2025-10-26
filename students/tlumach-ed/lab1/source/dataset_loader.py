from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataManager:
    def __init__(self, test_size: float = 0.3, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self):
        data = load_breast_cancer()
        X, y = data.data, data.target
        # convert {0,1} -> {-1, +1}
        y = 2 * y - 1

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test
