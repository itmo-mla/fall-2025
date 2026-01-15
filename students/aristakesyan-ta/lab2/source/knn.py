import numpy as np

class KNNParzenVariableH:
    def __init__(self, k=10):
        self.k = k
    
    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_)
        return self
    
    @staticmethod
    def _gaussian_kernel(r):
        return np.exp(-0.5 * (r ** 2))
    
    def _predict_one(self, x, leave_out_idx=None):
        x = np.asarray(x, dtype=float)
        
        diff = self.X_ - x
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        
        if leave_out_idx is not None:
            dists[leave_out_idx] = np.inf
        
        idx_sorted = np.argsort(dists)
        
        h = dists[idx_sorted[self.k - 1]]
        if not np.isfinite(h) or h == 0:
            h = 1e-8
        
        r = dists / h
        weights = self._gaussian_kernel(r)
        weights[~np.isfinite(weights)] = 0.0
        
        class_scores = np.zeros(len(self.classes_))
        for i, c in enumerate(self.classes_):
            mask_c = (self.y_ == c)
            class_scores[i] = np.sum(weights[mask_c])
        
        return self.classes_[np.argmax(class_scores)]
    
    def predict(self, X):
        X = np.asarray(X)
        preds = [self._predict_one(x) for x in X]
        return np.array(preds)
    
    def loo_error(self):
        n = self.X_.shape[0]
        errors = 0
        for i in range(n):
            y_pred_i = self._predict_one(self.X_[i], leave_out_idx=i)
            if y_pred_i != self.y_[i]:
                errors += 1
        return errors / n
