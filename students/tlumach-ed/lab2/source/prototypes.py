# алгоритмы отбора эталонов (жадное удаление/добавление)
"""Отбор эталонов: жадное удаление и добавление по LOO-ошибке (CCV эмпирически через LOO)."""
import numpy as np
from copy import deepcopy


def loo_error_with_prototypes(X, y, proto_idx, classifier_ctor, **kwargs):
    # classifier_ctor: callable returning объект с fit/predict
    # proto_idx: список индексов прототипов в X
    n = X.shape[0]
    errs = 0
    for i in range(n):
        # тренируем на прототипах, но не включаем xi как прототип, даже если он есть
        protos = [p for p in proto_idx if p != i]
        if len(protos) == 0:
            # без эталонов — классифицатор случайно (или по наибольшему классу)
            # считаем ошибку = 1 для безопасности
            errs += 1
            continue
        clf = classifier_ctor()
        clf.fit(X[protos], y[protos])
        pred = clf.predict(X[i:i+1])[0]
        if pred != y[i]:
            errs += 1
    return errs / n


def greedy_remove(X, y, classifier_ctor, tol=1e-6):
    # начинаем с всех точек как эталонов, поочередно убираем точку, которая
    # максимизирует уменьшение LOO-ошибки (или минимально увеличивает), пока есть выигрыш
    proto = list(range(X.shape[0]))
    best_err = loo_error_with_prototypes(X, y, proto, classifier_ctor)
    improved = True
    while improved and len(proto) > 1:
        improved = False
        best_candidate = None
        best_candidate_err = best_err
        for p in proto:
            cand = [q for q in proto if q != p]
            err = loo_error_with_prototypes(X, y, cand, classifier_ctor)
            if err <= best_candidate_err - tol:
                best_candidate_err = err
                best_candidate = p
        if best_candidate is not None:
            proto.remove(best_candidate)
            best_err = best_candidate_err
            improved = True
        else:
            break
    return proto, best_err


def greedy_add(X, y, classifier_ctor):
    # начинаем с одного представителя на класс, добавляем жадно
    classes = np.unique(y)
    proto = []
    for c in classes:
        # берем один случайный объект из класса c
        idx = np.where(y == c)[0][0]
        proto.append(int(idx))
    best_err = loo_error_with_prototypes(X, y, proto, classifier_ctor)
    improved = True
    remaining = [i for i in range(X.shape[0]) if i not in proto]
    while improved and len(remaining) > 0:
        improved = False
        best_candidate = None
        best_candidate_err = best_err
        for p in remaining:
            cand = proto + [p]
            err = loo_error_with_prototypes(X, y, cand, classifier_ctor)
            if err < best_candidate_err:
                best_candidate_err = err
                best_candidate = p
        if best_candidate is not None:
            proto.append(best_candidate)
            remaining.remove(best_candidate)
            best_err = best_candidate_err
            improved = True
        else:
            break
    return proto, best_err