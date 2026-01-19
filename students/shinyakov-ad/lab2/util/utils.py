from model.model import KNNParzen
import numpy as np

def loo_select_k(X, y, k_values):
    n = len(y)
    results = {}
    for k in k_values:
        clf = KNNParzen(k=k)
        correct = 0
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            clf.fit(X[mask], y[mask])
            pred = clf.predict(X[i:i+1])[0]
            if pred == y[i]:
                correct += 1
    
        acc = correct / n
        results[k] = acc
        print(f"k={k}: LOO accuracy={acc:.4f}")
    return results

def condensed_nn(X, y, k=1):
    X = np.asarray(X)
    y = np.asarray(y)
    idxs = list()
    classes = np.unique(y)
    for c in classes:
        first_idx = np.where(y == c)[0][0]
        idxs.append(first_idx)
    changed = True
    while changed:
        changed = False
        for i in range(len(X)):
            if i in idxs:
                continue
            clf = KNNParzen(k)
            clf.fit(X[idxs], y[idxs])
            pred = clf.predict(X[i:i+1])[0]
            if pred != y[i]:
                idxs.append(i)
                changed = True
    return np.array(idxs)

def edited_nn(X, y, k=3):
    X = np.asarray(X)
    y = np.asarray(y)
    mask = np.ones(len(y), dtype=bool)
    clf = KNNParzen(k=k)
    changed = True
    while changed:
        changed = False
        clf.fit(X[mask], y[mask])
        for i in range(len(y)):
            if not mask[i]:
                continue
            pred = clf.predict(X[i:i+1])[0]
            if pred != y[i]:
                mask[i] = False
                changed = True
    return np.where(mask)[0]