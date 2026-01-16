import numpy as np

class Optimizer:
    """momentum or fast"""
    def __init__(self, lr: float = 1e-4, momentum: float = 0.5, mode: str = "momentum", n_search_steps: int = 10):
        self.lr = lr
        self.momentum = momentum
        self.mode = mode
        self.n_search_steps = n_search_steps
        self.velocity = {}

    def update(self, name: str, param: np.ndarray, grad: np.ndarray, **kwargs) -> np.ndarray:
        param_arr = np.array(param, copy=False)
        grad_arr = np.array(grad, copy=False)

        if self.mode == "momentum":
            v = self.velocity.get(name, np.zeros_like(param_arr))
            v = self.momentum * v + self.lr * grad_arr
            self.velocity[name] = v
            return param_arr - v

        else:
            # === FAST MODE ===
            X_batch = kwargs.get("X_batch", None)
            y_batch = kwargs.get("y_batch", None)
            loss_fn = kwargs.get("loss_fn", None)
            reg_fn = kwargs.get("reg_fn", None)
            bias = kwargs.get("bias", 0.0)

            if X_batch is None or y_batch is None or loss_fn is None:
                # fallback simple SGD step
                return param_arr - self.lr * grad_arr

            best_loss = float('inf')
            best_param = param_arr.copy()

            # перебираем lr-кандидаты в геометрической прогрессии
            lrs = self.lr * np.logspace(-2, 2, self.n_search_steps)
            for lr_try in lrs:
                trial = param_arr - lr_try * grad_arr


                trial_scores = (X_batch @ trial + bias).ravel()

                trial_loss = loss_fn(y_batch, trial_scores)
                if reg_fn is not None:
                    trial_loss += reg_fn(trial)

                if trial_loss < best_loss:
                    best_loss = trial_loss
                    best_param = trial.copy()

            return best_param
