from sklearn.metrics import f1_score

from utils import *
from logreg import *
from visualisation import *


if __name__ == '__main__':
    seed: int = 42
    max_iter: int = 50
    tol: float = 1e-10

    train_raw, test_raw, train, test = load_dataset(seed=seed)

    res_newton = newton_raphson_fit(*train, max_iter=max_iter, tol=tol)
    res_irls = irls_fit(*train, max_iter=max_iter, tol=tol)

    ref = LogisticRegression(
        penalty=None,
        solver="newton-cg",
        max_iter=max_iter,
        tol=tol,
        fit_intercept=True,
        random_state=seed,
    )
    ref.fit(*train_raw)
    w_ref = np.concatenate(([ref.intercept_[0]], ref.coef_.ravel()))

    convergence_plot(res_newton, res_irls)
    roc_plot(res_newton, res_irls, w_ref, *test)

    print("f1 (Newton vs IRLS): ", f1_score(sigmoid(test[0] @ res_newton.w).round(), sigmoid(test[0] @ res_irls.w).round()))
    print("f1 (Newton vs sklearn): ", f1_score(sigmoid(test[0] @ res_newton.w).round(), ref.predict(test_raw[0])))
    print("f1 (IRLS vs sklearn): ", f1_score(sigmoid(test[0] @ res_irls.w).round(), ref.predict(test_raw[0])))
