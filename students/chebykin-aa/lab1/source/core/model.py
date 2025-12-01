import numpy as np

from .optimizer import SGD

class BinaryClassificator():
    def __init__(
        self,
        init_type: str = "none",
        optimizer_type: str = "SGD",
        loss_type: str = "hinge",
        subsampling_type: str = "random",
        lr: float = 1e-3,
        reg_coef: float = 0.0,
        m: int = 3,
        momentum: float = 0.0,
        nesterov: bool = False,
        h_optimization = True,
        batch_size: int = 4,
        n_iters: int = 100
    ):
        # Валидируем значения параметров
        if init_type not in ("multi_start", "corr", "none"):
            raise ValueError(
                "Параметр init_type должен принимать одно из следующих значений: "
                "multi_start, corr, none"
            )
        if optimizer_type not in ("SGD"):
            raise ValueError(
                "Параметр init_type должен принимать одно из следующих значений: "
                "SGD"
            )
        if loss_type not in ("hinge"):
            raise ValueError(
                "Параметр loss_type должен принимать одно из следующих значений: "
                "hinge"
            )
        if subsampling_type not in ("random", "margin_abs"):
            raise ValueError(
                "Параметр subsampling_type должен принимать одно из следующих значений: "
                "random, margin_abs"
            )
        if nesterov and momentum == 0.0:
            raise ValueError(
                "Параметр 'nesterov' может быть активен только при значениях параметра 'momentum' > 0.0"
            )           
        # Инициализируем параметры
        self.w = None
        optimizer_params = {
            "init_type": init_type,
            "loss_type": loss_type,
            "subsampling_type": subsampling_type,
            "lr": lr,
            "reg_coef": reg_coef,
            "m": m,
            "momentum": momentum,
            "nesterov": nesterov,
            "batch_size": batch_size,
            "n_iters": n_iters,
            "h_optimization": h_optimization
        }
        if optimizer_type == "SGD":
            self.optimizer = SGD(**optimizer_params)

    def get_train_info(
        self
    )-> tuple[dict, dict, dict]:
        """Метод, позволяющий получить данные об обучении модели"""
        return self.optimizer.get_train_info()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    )-> np.ndarray:
        """Метод обучения модели"""
        self.w = self.optimizer(X, y)

    def predict(
        self,
        X: np.ndarray,
    )-> np.ndarray:
        """Метод предсказания модели"""
        if self.w is None:
            raise ValueError(
                "Перед тем как использользовать метод 'predict' нужно обучить модель!"
            )
        preds = np.dot(X, self.w)
        return preds, np.sign(preds)