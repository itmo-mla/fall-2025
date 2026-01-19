import random
import numpy as np
from typing import Literal
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier, LogisticRegression

# ------------------------

SEED = 18092025
random.seed(SEED)
np.random.seed(SEED)

# ------------------------------ Main Class ----------------------------------

class LogRegNumpy():

    def __init__(
        self,
        # --- base params
        initial_weights:   list[list[float]] = None, # (n_features, n_classes)
        initial_bias:      list[float] = None,       # (1, n_classes)
        init_strategy:     Literal['normal', 'corr', 'multistart'] = 'normal',
        # --- fit params
        total_steps:    int = 1000,
        learning_rate:  float = 1e-3, # TODO: add lr reduction similar to sklearn strategies for SGDClassifier/LogisticRegression
        gd_algo:        Literal['gd', 'sgd', 'minibatch'] = 'sgd',
        batch_size:     int | None = None, # None = full dataset
        momentum:       float = 0.0,
        l2:             float = 0.0,
        optim_step:     bool = False,
        # early stopping
        early_stopping: bool = False,
        tolerance:      float = 1e-3,
        n_startup_rounds:  int = 10,
        early_stop_rounds: int = 5,
        validation_fraction: float = 0.1, # (0.0, 1.0)
        # recurrent loss function estimation
        rec_mode:                Literal['off','mean','ema'] = 'off',
        ema_lambda:              float = 0.1,
        # sampling strategy
        sampling_mode:           Literal['uniform','by_margin'] = 'uniform',
        shuffle:                 bool = True,
        sampling_tau:            float = 0.2,
        sampling_min_prob:       float = 0.01,
        refresh_rate:            int = 100, # how often to update samples probability distribution
        # --- logs
        steps_per_epoch:         int | None = 100, # how often to update logs, to evaluate intermediate loss, etc.
        verbose:                 bool = False,
        # --- misc
        use_best_weights:        bool = False,
        return_weights_history:  bool = False,
        random_seed: int = SEED,
        eps: float = 1e-9
    ):
        # --- base params init
        self.weights = (np.array(initial_weights) if initial_weights is not None 
                        else np.array([]))
        self.bias = (np.array(initial_bias) if initial_bias is not None 
                     else np.array([]))
        self.init_strategy = init_strategy
        # --- fit params init
        self.total_steps = total_steps
        self.learning_rate = learning_rate
        self.gd_algo = gd_algo
        self.batch_size = batch_size
        self.momentum = momentum
        self.l2 = l2
        self.optim_step = optim_step
        # early stopping
        self.early_stopping = early_stopping
        self.tolerance = tolerance
        self.n_startup_rounds = n_startup_rounds
        self.early_stop_rounds = early_stop_rounds
        self.validation_fraction = validation_fraction
        # recurrent loss function estimation
        self.rec_mode = rec_mode
        self.ema_lambda = ema_lambda
        # sampling strategy
        self.sampling_mode = sampling_mode
        self.shuffle = shuffle
        self.sampling_tau = sampling_tau
        self.sampling_min_prob = sampling_min_prob
        self.refresh_rate = refresh_rate
        # --- logs
        self.steps_per_epoch = steps_per_epoch
        self.verbose = verbose
        # --- misc init
        self.use_best_weights = use_best_weights
        self.return_weights_history = return_weights_history
        self.random_seed = random_seed
        self.eps = eps

        self.rng_ = np.random.default_rng(seed=random_seed)

        # для рекуррентной оценки
        self.rec_value = None
        self.rec_count = 0
        self.rec_history = []

    
    def fit(
        self,
        X, y,
    ) -> None | list[list[float]]:

        input_check = lambda data, dtype: (
            np.array(data, dtype=dtype).squeeze()
            if not isinstance(data, np.ndarray)
            else deepcopy(data).astype(dtype, copy=False)
        )
        X, y = input_check(X, np.float32), input_check(y, np.int32)
        if X.ndim == 1:
            X = X[np.newaxis, :]

        if self.early_stopping:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, 
                random_state=self.random_seed, shuffle=self.shuffle, stratify=y
            )

        N = X.shape[0]

        # TODO: try python switch-case
        if self.gd_algo == 'gd':
            self.batch_size = N
        elif self.gd_algo == 'sgd':
            self.batch_size = 1

        if self.steps_per_epoch is None:
            self.steps_per_epoch = (N + self.batch_size - 1) // self.batch_size

        # --- init
        # Initialization is done on the whole dataset (sorry for this, but I'm tired)
        self._init_weights(X, y, n_starts=5, search_steps=50, lr=self.learning_rate)
        if self.return_weights_history:
            weights_values = dict(weights=[self.weights.copy()], bias=[self.bias.copy()])
        Vdw = np.zeros_like(self.weights)
        Vdb = np.zeros_like(self.bias)
        self._rec_reset()
        self.loss_values = ([], []) if self.early_stopping else [] # Loss is cleared each fit call
        rng = self.rng_
        learning_rate = self.learning_rate
            
        def uniform_next_batch_stateful():
            perm = rng.permutation(N) if self.shuffle else np.arange(N, dtype=np.int64)
            ptr = 0

            def get_batch():
                nonlocal perm, ptr
                if ptr >= N:
                    perm = rng.permutation(N) if self.shuffle else np.arange(N, dtype=np.int64)
                    ptr = 0

                remaining = N - ptr
                take = self.batch_size if self.batch_size <= remaining else remaining

                idx = perm[ptr:ptr + take]
                ptr += take
                return idx

            return get_batch

        margin_probs = lambda: self._margin_sampling_probs(
            X, y, use_abs=True, tau=self.sampling_tau, min_prob=self.sampling_min_prob
        )

        # ---- training loop (one unified path)
        step = 0
        block_loss_sum = 0.0
        block_count = 0

        if self.sampling_mode == 'uniform':
            next_uniform_batch = uniform_next_batch_stateful()
        elif self.sampling_mode == 'by_margin':
            probs = margin_probs()
            next_uniform_batch = None
        else:
            raise ValueError("sampling_mode must be 'uniform' or 'by_margin'")

        # --- init validation score
        # Train
        loss = self._get_loss(X, y)
        if self.early_stopping:
            self.loss_values[0].append(loss)
        else:
            self.loss_values.append(loss)
        # recurrent quality update (train loss)
        train_rec_val = self._rec_update(loss)
        self.rec_history.append(train_rec_val)
        # Validation
        if self.early_stopping:
            loss = self._get_loss(X_val, y_val)
            self.loss_values[1].append(loss)

        # for early stopping similarly to sklearn
        no_improvement_counter = 0   
        best_loss = loss
        best_step = 0

        while step < self.total_steps:
            if self.sampling_mode == 'uniform':
                batch_idx = next_uniform_batch()
            else:  # by_margin
                if step % self.refresh_rate == 0:
                    probs = margin_probs()
                batch_idx = rng.choice(N, size=self.batch_size, replace=True, p=probs)

            xi = X[batch_idx, :]
            yi = y[batch_idx]

            # forward / loss
            logits = self.forward(xi)
            loss   = self._loss_fn_opt(yi, logits, reduction=None)

            # Self-Normalized Importance Sampling Loss
            if self.sampling_mode == 'by_margin':
                pi = probs[batch_idx]
                sample_weights = 1.0 / np.clip(pi, 1e-12, None)
                loss = (loss * sample_weights).sum() / sample_weights.sum()
            else:
                sample_weights = None
                loss = loss.mean()
            
            block_loss_sum += loss
            block_count    += 1

            # recurrent quality update (train loss)
            train_rec_val = self._rec_update(loss)
            self.rec_history.append(train_rec_val)

            # gradients
            w_grad, b_grad = self._gradient(xi, yi, logits)
            # L2 - regularization
            if self.l2 > 0.0:
                w_grad += self.l2 * self.weights

            # momentum (EMA style)
            Vdw = self.momentum * Vdw - (1.0 - self.momentum) * w_grad
            Vdb = self.momentum * Vdb - (1.0 - self.momentum) * b_grad
            
            if self.optim_step:
                # learning_rate = self._golden_ratio_search(
                #     xi, yi, Vdw, Vdb, 0, 1, 1e-5, 1000
                # )
                learning_rate = self._line_search_backtracking(
                    xi, yi, w_grad, b_grad, Vdw, Vdb
                )

            # update
            self.weights += learning_rate * Vdw
            self.bias    += learning_rate * Vdb

            step += 1

            # Logging once per “epoch-sized” number of steps.
            # Early stopping criteria check
            if block_count >= self.steps_per_epoch:
                mean_block_loss = block_loss_sum / block_count

                if self.early_stopping:
                    self.loss_values[0].append(mean_block_loss)
                    loss = self._get_loss(X_val, y_val)
                    self.loss_values[1].append(loss)
                else:
                    self.loss_values.append(mean_block_loss)
                    # Workaround, loss value should represent global tendency in loss change.
                    # Current loss is value for a single opt. step.
                    loss = mean_block_loss

                # # recurrent quality update - for better plots maybe should log once per block, not every opt. step
                # rec_val = self._rec_update(mean_block_loss, mode=rec_mode, ema_lambda=ema_lambda)
                # self.rec_history.append(rec_val)

                # Early stopping on monitored series (smoothed if rec_mode != 'off').
                # Done on training set, if early_stopping = False, else on validation set
                # Recalculating loss on validation each opt. step is too costly actually
                if (self.tolerance is not None) and (step > self.n_startup_rounds + 1):
                    if loss > best_loss - self.tolerance:
                        no_improvement_counter += 1
                        if no_improvement_counter >= self.early_stop_rounds:
                            # if self.verbose:
                            print(f"Last loss : {loss}\t Best loss: {best_loss} on step {best_step}({best_step // self.steps_per_epoch})\t")
                            print(f"Early stopping at step {step}({step // self.steps_per_epoch})")
                            break
                    else:
                        best_loss = loss
                        no_improvement_counter = 0
                        best_step = step
                        if self.use_best_weights:
                            best_weights = self.weights.copy()
                            best_bias = self.bias.copy()


                if self.return_weights_history:
                    weights_values['weights'].append(self.weights.copy())
                    weights_values['bias'].append(self.bias.copy())
                if self.verbose:
                    print(
                        f"step {step:6d} ({step // self.steps_per_epoch}) "
                        f"| block_loss={mean_block_loss:.6f} "
                        f"| val_loss={loss:.6f} "
                        f"| batch_size={self.batch_size} "
                        f"| mode={self.sampling_mode}"
                    )

                block_loss_sum = 0.0
                block_count = 0

        if self.use_best_weights:
            self.weights = best_weights
            self.bias = best_bias

        if self.return_weights_history:
            return np.array(weights_values)
        
        return
        
    def _get_loss(self, X, y):
        logits = self.forward(X)
        loss = self._loss_fn_opt(y, logits, reduction='mean')
        return loss
    
    
    def _golden_ratio_search(
        self,
        X, y,
        dir_w:  np.ndarray, dir_b:  np.ndarray,
        lo: float = 0.0, hi: float = 1.0,
        tol: float = 1e-6, max_iters: int = 1000
    ):  
        # zero iteration
        logits_0 = self.forward(X)
        Xdw = X @ dir_w
        Wdw = np.sum(self.weights * dir_w)
        dw_norm2 = np.sum(dir_w * dir_w)
        ######################################

        def f(t):
            logits_t = logits_0 + t * (Xdw + dir_b)
            loss_t = self._loss_fn_opt(y, logits_t, reduction='mean')
            if self.l2 > 0.0:
                loss_t += self.l2 * 0.5 * (2.0 * t * Wdw + t * t * dw_norm2)
            return loss_t
        
        inv_phi = (5**0.5 - 1) / 2
        x1 = hi - inv_phi * (hi - lo)
        x2 = lo + inv_phi * (hi - lo)
        f1 = f(x1)
        f2 = f(x2)

        i = 0
        while (hi - lo) / 2 >= tol and i < max_iters:            

            if f1 > f2:
                lo, f1 = x1, f2
                x1 = hi - inv_phi * (hi - lo)
                x2 = lo + inv_phi * (hi - lo)
                f2 = f(x2)
            else:
                hi, f2 = x2, f1
                x1 = hi - inv_phi * (hi - lo)
                x2 = lo + inv_phi * (hi - lo)
                f1 = f(x1)
            
            i += 1
            
        if i == max_iters:
            print(f'Optimal search max iter reached, step is unoptimal')

        return (lo + hi) / 2.
        

    def _line_search_backtracking(
        self,
        X, y, 
        grad_w: np.ndarray, grad_b: np.ndarray, 
        dir_w:  np.ndarray, dir_b:  np.ndarray,
        step: float = 1.0, 
        alpha: float = 1e-4, 
        beta: float = 0.5, 
        tol: float = 1e-8,
        default_lr: float = 1e-9,
        eps: float = 1e-12
    ):
        logits_0 = self.forward(X)
        loss_0 = self._loss_fn_opt(y, logits_0, reduction='mean')
        if self.l2 > 0.0:
            loss_0 += self.l2 * np.sum(np.pow(self.weights, 2)) * 0.5
        
        W0 = self.weights
        b0 = self.bias

        # directional derivatives
        dw = (grad_w * dir_w).sum()
        db = (grad_b * dir_b).sum()
        dd = dw + db
        
        if self.l2 > 0.0:
            dd += self.l2 * (W0 * dir_w).sum()

        if dd >= eps:
            if np.allclose(dir_w, -grad_w) and np.allclose(dir_b, -grad_b):
                return 0.0
            # switching to regular gradient descent
            np.copyto(dst=dir_w, src=-grad_w); np.copyto(dst=dir_b, src=-grad_b)
            dd = (grad_w * dir_w).sum() + (grad_b * dir_b).sum()
            if self.l2 > 0.0:
                dd += self.l2 * (W0 * dir_w).sum()
            if dd >= eps:
                return 0.0

        t = step

        # TODO: добавить оптимизации через раскрытие logits_t и l2_norm и предварительного подсчета неизменных членов
        while t > tol:
            Wt = W0 + t * dir_w
            bt = b0 + t * dir_b
            logits_t = np.matmul(X, Wt) + bt
            loss_t = self._loss_fn_opt(y, logits_t, reduction='mean')
            if self.l2 > 0.0:
                loss_t += self.l2 * np.sum(np.pow(Wt, 2)) * 0.5

            if loss_t <= loss_0 + alpha * t * dd:
                return t
            
            t *= beta

        return default_lr


    def predict(self, features: list[list[float]]):
        X = (np.array(features).squeeze() if not isinstance(features, np.ndarray) 
             else deepcopy(features).astype(np.float32, copy=False))
        if X.ndim == 1:
            X = X[np.newaxis, :]
        logits = self.forward(X) # (n_samples, n_classes)
        probs  = self._softmax(logits) # не обязательно
        return np.argmax(probs, axis=1)

    def predict_proba(self, features: list[list[float]]):
        X = (np.array(features).squeeze() if not isinstance(features, np.ndarray) 
             else deepcopy(features).astype(np.float32, copy=False))
        if X.ndim == 1:
            X = X[np.newaxis, :]
        logits = self.forward(X) # (n_samples, n_classes)
        probs  = self._softmax(logits) # не обязательно
        return probs
    
    def _create_onehot_target(self, y: np.array):
        ohe_enc = OneHotEncoder(categories=[np.unique(y)], sparse_output=False)
        y_enc = ohe_enc.fit_transform(y.reshape(-1, 1))
        return y_enc # output -> (n_samples, n_classes)
    
    # TODO: add option, when non-standardized feature come
    def _init_weights(
        self, X: np.ndarray, y: np.ndarray,
        n_starts: int = 5, search_steps: int = 50, lr: float = 1e-2,
    ):
        N, d = X.shape
        K = np.max(y) + 1

        if self.init_strategy == 'normal':
            if self.weights.size == 0:
                self.weights = self.rng_.standard_normal((d, K), dtype=np.float32)
            if self.bias.size == 0:
                self.bias = self.rng_.standard_normal((1, K), dtype=np.float32)
            return

        if self.init_strategy == 'corr':
            # by default this strategy assumes that input data already was standardized
            if self.weights.size != 0 and self.bias.size != 0:
                return
            # евклидова норма
            # denom = np.sum(X * X, axis=0)       # shape (d,)
            denom = np.float64(N)

            W = np.zeros((d, K), dtype=np.float64)
            b = np.zeros((1, K), dtype=np.float64)

            for k in range(K):
                t = (y == k).astype(np.float64) # 1 for class k, else 0
                # weights: elementwise division by per-feature squared norm
                # With centered X, X^T t == X^T (t - mean(t)), so no need to center t explicitly.
                numer = X.T @ t                           # shape (d,)
                W[:, k] = numer / denom

                # intercept: with centered features, LS gives b_k = mean(t^{(k)})
                b[0, k] = t.mean()

            if self.weights.size == 0:
                self.weights = W.astype(np.float32, copy=False)
            if self.bias.size == 0:
                self.bias = b.astype(np.float32, copy=False)
            return
        
        if self.init_strategy == 'multistart':
            best_loss = np.inf
            best_W, best_b = None, None

            for _ in range(n_starts):
                W = self.rng_.standard_normal((d, K), dtype=np.float32)
                b = self.rng_.standard_normal((1, K), dtype=np.float32)
                
                # short warmup
                w, b, loss = self._warmup(X, y, W, b, steps=search_steps, lr=lr)
                if loss < best_loss:
                    best_loss = loss
                    best_W, best_b = w, b

                self.weights = best_W
                self.bias    = best_b
            return
        
        raise ValueError("init_strategy must be 'normal' or 'corr'")
    
    # Warmup is performed on the whole input dataset, not on batches
    def _warmup(self, X, y, W, b, steps=50, lr=1e-2):
        W = W.copy(); b = b.copy()
        for _ in range(steps):
            logits = np.matmul(X, W) + b
            loss   = self._loss_fn_opt(y, logits, reduction='mean')
            w_grad, b_grad = self._gradient(X, y, logits)
            W -= lr * w_grad
            b -= lr * b_grad
        return W, b, float(loss)
    
        
    def _softmax(self, X: np.array) -> np.array:
        Z = X - np.max(X, axis=1, keepdims=True)
        numerator = np.exp(Z)
        denominator = np.sum(numerator, axis=1, keepdims=True)
        softmax_probs = numerator / denominator
        return softmax_probs # -> (n_samples, n_classes)
    
    def forward(self, X):
        # (n_samples, n_features) * (n_features, n_classes)
        logits = np.matmul(X, self.weights) + self.bias # -> (n_samples, n_classes)
        return logits
    
    # def loss_fn_expanded(self, X, y_true):
    #     # (n_samples, n_features) * (n_features, n_classes) + (n_samples, 1) * (1, n_classes) = (n_samples, n_classes)
    #     logits = np.matmul(X, self.weights) + np.matmul(np.ones((X.shape[0], 1)), self.bias)
    #     exp_logits = np.exp(logits)
    #     logits_sum = np.sum(exp_logits, axis=1) # -> (n_samples, 1)
    #     # (n_samples, n_classes) * (n_samples, n_classes)
    #     true_class_logits = logits[np.arange(X.shape[0]), y_true]
    #     return np.mean(np.log(logits_sum) - true_class_logits)

    # def loss_fn(self, y_true, logits):
    #     log_probs = np.log(self.softmax(logits)) # -> (n_samples, classes)
    #     # y_true_ohe = self.create_onehot_target(y_true) # -> (n_samples, classes)
    #     # likelihood = (log_probs * y_true_ohe).sum(axis=1).mean()
    #     likelihood = (log_probs[np.arange(log_probs.shape[0]), y_true]).mean()
    #     return -likelihood
    
    def _loss_fn_opt(self, y_true, logits, reduction=None):
        lse = logsumexp(logits, axis=1, keepdims=True)
        nll = lse - logits
        loss = nll[np.arange(nll.shape[0]), y_true]
        if reduction == 'mean':
            loss = loss.mean()
        return loss
    
    def _rec_reset(self):
        self.rec_value = None
        self.rec_count = 0
        self.rec_history = []

    def _rec_update(self, xi):
        # Smoothing is applied only to train loss history.
        # Validation loss shouldn't be smoothed.
        # One possible explanation for this - validation loss is always calculated on the whole val set,
        # so there is no more reason to apply smoothing to it. 
        if self.rec_mode == "off":
            return xi

        if self.rec_value is None:
            # инициализация последовательности
            self.rec_value = xi
            self.rec_count = 1
            return self.rec_value

        if self.rec_mode == "mean":
            # running mean: Q_m = (1/m)*xi_m + (1 - 1/m)*Q_{m-1}
            self.rec_count += 1
            m = self.rec_count
            self.rec_value = (1.0 / m) * xi + (1.0 - 1.0 / m) * self.rec_value
            return self.rec_value

        if self.rec_mode == "ema":
            # EMA: Q_m = λ xi_m + (1 - λ) Q_{m-1}
            self.rec_value = self.ema_lambda * xi + (1.0 - self.ema_lambda) * self.rec_value
            return self.rec_value

        return xi

    def _gradient(self, X, y_true, logits):
        y_prob = self._softmax(logits)
        y_prob[np.arange(y_prob.shape[0]), y_true] -= 1
        y_prob /= y_prob.shape[0]
        w_grad = np.matmul(X.T, y_prob)
        b_grad = y_prob.sum(axis=0, keepdims=True)
        return w_grad, b_grad
    
    # Only for multiclass, no separate implementation for binary case
    # TODO: make binary case margin estimation
    def calc_margins(self, X, y_true):
        logits = self.forward(X)
        true_logits = logits[np.arange(X.shape[0]), y_true]
        logits[np.arange(logits.shape[0]), y_true] = -np.inf
        false_logits = logits.max(axis=1)
        margins = true_logits - false_logits
        return margins

    # TODO: сделать сэмплер с разной логикой выбора сложных случаев
    # 1) -abs(margins) - для любых (правильных или нет) случаев с малой долей уверенности
    # 2) -margins вместо -abs(margins) - для точно неправильно классифицированных случаев
    def _margin_sampling_probs(
        self, X, y, use_abs: bool = True, tau: float = 0.2, min_prob: float = 0.01 # small tau -> harder samples, large tau -> closer to uniform
    ):
        margins = self.calc_margins(X, y)

        diff = -np.abs(margins) if use_abs else -margins
        scores = diff / max(tau, 1e-8)
        probs = self._softmax(scores.reshape(1, -1)).squeeze()

        floor = min_prob / X.shape[0]
        probs = (1.0 - min_prob) * probs + floor

        return probs
    

# ---------------------- Factory Class for sklearn models ---------------------------
class MarginMixin:
    def calc_margins(self, X: np.ndarray, y_true: np.ndarray):
        logits = self.decision_function(X)
        # gather true-class scores
        true_logits = logits[np.arange(X.shape[0]), y_true]
        # mask out the true class to get the best 'false' score per row
        logits[np.arange(logits.shape[0]), y_true] = -np.inf
        false_logits = logits.max(axis=1)
        margins = true_logits - false_logits
        return margins
    
def with_margins(BaseCls):
    bases = (BaseCls, MarginMixin) if issubclass(BaseCls, BaseEstimator) else (BaseCls, MarginMixin, BaseEstimator)
    name = f"{BaseCls.__name__}SK"
    cls = type(name, bases, {}) # metaclass?
    cls.__module__ = BaseCls.__module__
    cls.__doc__ = (BaseCls.__doc__ or "") + (
        "\n\nThis subclass adds `calc_margins(X, y_true)` to compute per-sample "
        "margins = score(true) - max_{c!=true} score(c)."
    )
    return cls

# ---------------------- Sklearn Logistic Regression extended classes ---------------------------
SGDClassifierSK = with_margins(SGDClassifier)
LogisticRegressionSK = with_margins(LogisticRegression)