import numpy as np
class KNN:
    def __init__(self, kernel="gauss", k=3):
        self.X = None
        self.Y = None
        self.unique_y = None
        self.k = k

        # self.kernel = np.vectorize(lambda x: np.exp(-2 * x ** 2))
        self.kernel = np.vectorize(lambda x: abs(x) < 1)

    def ro(self, x, x_i, p=2):
        return np.sum((x - x_i) ** p) ** (1 / p)

    def fit(self, x, y):
        self.X = x
        self.Y = y
        self.unique_y = np.unique(self.Y)

    def predict_(self, x):

        X_ro = []
        for ind in range(self.X.shape[0]):
            X_ro.append(self.ro(self.X[ind, :], x))
        X_ro = np.array(X_ro).reshape(-1, 1)
        X_ro = np.array(sorted(np.concatenate((X_ro, self.Y.reshape(-1, 1)), axis=1), key=lambda x: x[0]))

        X_ro = X_ro[X_ro[:, 0] > 1e-10, :]
        K_plus_ro = X_ro[self.k, 0]

        X_ro[:, 0] = self.kernel(X_ro[:, 0] / K_plus_ro)
        ans = {}
        for y in self.unique_y:
            ans[y] = 0
            for ind in range(X_ro.shape[0]):
                ans[y] += int(X_ro[ind, 1] == y) * X_ro[ind, 0]
        mx = max(ans.values())

        for i in ans.items():
            if i[1] == mx:
                return i[0]

    def predict(self, x):
        ans = []
        for i in range(x.shape[0]):
            ans.append(self.predict_(x[i, :]))
        return np.array(ans)