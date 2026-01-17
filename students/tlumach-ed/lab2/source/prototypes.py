import numpy as np


def ccv_error_with_prototypes(X, y, proto_idx, classifier_ctor, k=31,  **kwargs):
    n = X.shape[0]
    errs = 0
    for i in range(n):
        protos = [p for p in proto_idx if p != i]
        if len(protos) < k:
            continue
        clf = classifier_ctor()
        clf.fit(X[protos], y[protos])
        pred = clf.predict(X[i:i+1],k=k)[0]
        if pred != y[i]:
            errs += 1
    return errs / n


def greedy_remove(X, y, classifier_ctor, k=31, tol=1e-6):
    # начинаем с всех точек как эталонов, поочередно убираем точку, которая
    # максимизирует уменьшение LOO-ошибки (или минимально увеличивает), пока есть выигрыш
    proto = list(range(X.shape[0]))
    best_err = ccv_error_with_prototypes(X, y, proto, classifier_ctor, k)
    history = [(len(proto), best_err)]
    improved = True
    while improved and len(proto) > k:
        improved = False
        best_candidate = None
        best_candidate_err = float('inf')
        for p in proto:
            cand = [q for q in proto if q != p]
            err = ccv_error_with_prototypes(X, y, cand, classifier_ctor, k)
            if err < best_candidate_err:
                best_candidate_err = err
                best_candidate = p
        if best_candidate is not None and best_candidate_err <= best_err + tol:
            proto.remove(best_candidate)
            best_err = best_candidate_err
            history.append((len(proto), best_err))
            improved = True
        else:
            break
    return proto, best_err, history


