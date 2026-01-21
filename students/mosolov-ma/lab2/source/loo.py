from knn import KNN
from data_workflow import load_and_prepare_data, scale_features
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut

df = load_and_prepare_data()

X = scale_features(df.drop(columns=['target']))
y = df['target']

X = X.to_numpy()
y = y.to_numpy()


class LOO():
    def __init__(self, X, y, k_max, split_method='custom', model='custom'):
        self.X = X
        self.y = y
        self.k_max = k_max
        self.split_method = split_method
        self.model = model

    def _get_model(self,k):
        match self.model:
            case 'custom':
                return KNN(k=k)
            case 'sklearn':
                return KNeighborsClassifier(n_neighbors=k, metric="euclidean")
            case _:
                raise ValueError("Unknown model")
            
    def _custom_split(self, X):
        for i in range(len(X)):
            yield [j for j in range(len(X)) if j != i], [i]
            
    def _get_split(self):
        match self.split_method:
            case 'custom':
                return self._custom_split(self.X)
            case 'sklearn':
                return LeaveOneOut().split(self.X)
            case _:
                raise ValueError("Unknown init_method")
            
    def iteration(self, k):
        error = 0
        for train_idx, test_idx in self._get_split():
            model = self._get_model(k)
            model.fit(self.X[train_idx], self.y[train_idx])
            error += (model.predict(self.X[test_idx]) != self.y[test_idx]).sum()
        return error / len(self.X)
        
    def get_average_errors_for_all_k(self):
        return [self.iteration(k) for k in range(1, self.k_max)]
    

    def plot(self, errors):
        plt.plot(range(1, self.k_max), errors, label=f"model: {self.model}\nsplit method: {self.split_method}")
        plt.xlabel("k")
        plt.ylabel("Error")
        plt.title("LOO")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    custom_loo = LOO(X, y, k_max=200)

    sklearn_loo = LOO(X, y, k_max=200, split_method='sklearn', model='sklearn')

    v1 = custom_loo.get_average_errors_for_all_k()
    v2 = sklearn_loo.get_average_errors_for_all_k()

    custom_loo.plot(v1)
    sklearn_loo.plot(v2)

