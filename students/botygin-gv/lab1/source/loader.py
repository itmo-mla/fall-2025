from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoader:
    def __init__(self, test_size: float = 0.3, dataset_name: str = 'breast_cancer'):
        self.dataset_name = dataset_name
        self.test_size = test_size

    def __train_test_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=42
        )
        return X_train, X_test, y_train, y_test

    @staticmethod
    def __load_iris():
        data = load_iris()
        X, y = data.data, data.target
        mask = y < 2
        return X[mask], y[mask]

    @staticmethod
    def __load_breast_cancer():
        data = load_breast_cancer()
        X, y = data.data, data.target
        return X, y

    def load_data(self):
        if self.dataset_name == 'iris':
            X, y = self.__load_iris()
        elif self.dataset_name == 'breast_cancer':
            X, y = self.__load_breast_cancer()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        y = 2 * y - 1

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return self.__train_test_split(X, y)
