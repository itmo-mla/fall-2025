import numpy as np
from knn import KNN
from utils import accuracy_score
class CCV:
    def __init__(self):
        self.metric = accuracy_score
        self.history = None

    def fit(self, x, y, model=KNN(k=1)):
        from tqdm import tqdm
        ans = []
        mask = np.ones(x.shape[0]).astype(bool)
        flag = True
        while flag:  # or np.sum(~mask) < 10
            mn = 1e5
            mn_ind = -1
            for ind in tqdm(range(x.shape[0])):
                if not mask[ind]:
                    continue
                mask[ind] = 0
                # mask_test = (1 - mask).astype(bool)
                mask_train = mask.astype(bool)
                x_train = x[mask_train, :]
                y_train = y[mask_train]
                model.fit(x_train, y_train)

                y_pred = model.predict(x_train)
                emp_ = (1 - self.metric(y_true=y_train, y_pred=y_pred))
                if emp_ < mn:
                    mn = emp_
                    mn_ind = ind

                mask[ind] = 1

            flag = self.count(ans, mn)

            ans.append(mn)
            mask[mn_ind] = 0
            self.history = [mask, mn]

        return mask, ans

    @staticmethod
    def count(x_prev, x_new):
        if len(x_prev) == 0:
            return True
        if x_new < 1e-10:
            print(f"small x_new: {x_new}")
            return False
        x_mean = np.array(x_prev[-5:]).mean()
        print(x_new, x_mean, x_new - x_mean)
        return (x_new - x_mean < 1e-5)

