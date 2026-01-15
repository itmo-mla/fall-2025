import numpy as np
from utils import accuracy_score
class LOO:
    def __init__(self, metric="accuracy"):
        self.metric = accuracy_score
    def fit(self, model, x, y):
        from tqdm import tqdm
        ans = []
        for ind in tqdm(range(x.shape[0])):
            mask = np.zeros(x.shape[0])
            mask[ind] = 1
            mask_test = mask.astype(bool)
            mask_train = (1 - mask).astype(bool)
            x_train = x[mask_train, :]
            x_test = x[mask_test, :]
            y_train = y[mask_train]
            y_test = y[mask_test]
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            ans.append(1 - self.metric(y_true=y_test, y_pred=y_pred))
        ans = np.array(ans)
        return ans.mean(), ans.std()
