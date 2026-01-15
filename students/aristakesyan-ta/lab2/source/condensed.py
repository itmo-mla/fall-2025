import numpy as np

def condensed_nn(X, y, max_iter=10):
    """
    Алгоритм конденсированного ближайшего соседа (Condensed Nearest Neighbor, CNN).
    
    Отбирает подмножество эталонов из обучающей выборки таким образом, 
    чтобы 1-NN на этом подмножестве правильно классифицировал исходную выборку
    (или минимизировал ошибку).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    
    n = X.shape[0]
    classes = np.unique(y)
    
    proto_idx = []
    for c in classes:
        idx_c = np.where(y == c)[0][0]
        proto_idx.append(idx_c)
    
    proto_idx = list(set(proto_idx))
    
    def predict_1nn(x, Xp, yp):
        diff = Xp - x
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        return yp[np.argmin(dists)]
    
    changed = True
    it = 0
    
    while changed and it < max_iter:
        changed = False
        it += 1
        print(f"Итерация CNN {it} ... текущих эталонов: {len(proto_idx)}")
        
        for i in range(n):
            if i in proto_idx:
                continue
            
            x_i = X[i]
            y_i = y[i]
            
            Xp = X[proto_idx]
            yp = y[proto_idx]
            
            y_pred_i = predict_1nn(x_i, Xp, yp)
            
            if y_pred_i != y_i:
                proto_idx.append(i)
                changed = True
    
    return np.array(proto_idx, dtype=int)

