import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc


class LossFunction(ABC):
    def __init__(self, ):
        pass


    @abstractmethod
    def get_loss(self, ):
        pass

    
    @abstractmethod
    def get_grad(self, ):
        pass


class MSELoss(LossFunction):
    def get_loss(self, y_pred: np.ndarray, y_truth: np.ndarray):
        return np.sum((y_pred - y_truth) ** 2) / len(y_pred)


    def get_grad(self, y_pred: np.ndarray, y_truth: np.ndarray):
        return (2 / len(y_pred)) * (y_pred - y_truth)
    

class MAELoss(LossFunction):
    def get_loss(self, y_pred: np.ndarray, y_truth: np.ndarray):
        return np.sum(np.abs(y_pred - y_truth)) / len(y_pred)


    def get_grad(self, y_pred: np.ndarray, y_truth: np.ndarray):
        return np.sign(y_pred - y_truth) / len(y_pred)


class ActivationFunction(ABC):
    def __init__(self, ):
        pass


    @abstractmethod
    def apply(self, ):
        pass


    @abstractmethod
    def get_derivative(self, ):
        pass


class ActivationDummy(ActivationFunction):
    def apply(self, X: np.ndarray):
        return X
    

    def get_derivative(self, X: np.ndarray):
        return np.ones_like(X)
    

class TanhActivation(ActivationFunction):
    def apply(self, X: np.ndarray):
        return np.tanh(X)

    def get_derivative(self, X: np.ndarray):
        return 1.0 - np.tanh(X)**2


class Optimizer(ABC):
    def __init__(self, ):
        pass


    @abstractmethod
    def update(self, ):
        pass


class SteepestGradientDescentOptimizer(Optimizer):
    def __init__(self, weight_decay: float = 0.0):
        self.weight_decay = weight_decay

    
    def update(self, params: dict, grad_storage: dict):
        for layer_key, layer_params in grad_storage.items():
            X_layer = params[layer_key]['input']
            best_step = 1.0 / max(np.sum(X_layer**2), 1e-9)

            for param in layer_params.keys():
                grad = grad_storage[layer_key][param]

                if param == 'weight' and self.weight_decay > 0:
                    grad += self.weight_decay * params[layer_key][param]

                params[layer_key][param] -= best_step * grad
                

class SGDOptimizer(Optimizer):
    def __init__(self, lr: float = 1e-4, momentum: float = 0.0, weight_decay: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.vel = {}


    def update(self, params: dict, grad_storage: dict):
        for layer_key, layer_params in grad_storage.items():
            for param in layer_params.keys():
                grad = grad_storage[layer_key][param]

                if param == 'weight' and self.weight_decay > 0:
                    grad += self.weight_decay * params[layer_key][param]

                if self.momentum:
                    if layer_key not in self.vel.keys():
                        self.vel[layer_key] = {}
                    if param not in self.vel[layer_key].keys():
                        self.vel[layer_key][param] = np.zeros_like(layer_params[param])

                    self.vel[layer_key][param] *= self.momentum
                    self.vel[layer_key][param] += grad

                    params[layer_key][param] -= self.lr * self.vel[layer_key][param]
                else:
                    params[layer_key][param] -= self.lr * grad


class Sampler(ABC):
    def __init__(self, ):
        pass

    
    @abstractmethod
    def get_batch_indices(self, X: np.ndarray, margin: np.ndarray, batch_size: int):
        pass


class RandomBasedSampler(Sampler):
    def __init__(self, ):
        pass

    
    def get_batch_indices(self, X, margin, batch_size):
        amount = len(X)
        indices = np.random.choice(amount, size=min(batch_size, amount), replace=False)

        return indices
    

class MarginBasedSampler(Sampler):
    def __init__(self, temp: float = 0.95):
        self.temp = temp


    def get_batch_indices(self, X, margin, batch_size):
        margin_metric = 1.0 / (np.abs(margin) + 1e-9)
        margin_metric **= 1.0 / self.temp
        probability = margin_metric / np.sum(margin_metric)

        amount = len(X)
        chosen_index = np.random.choice(amount, size=min(batch_size, amount), p=probability.flatten())
 
        return chosen_index


class ModelBaseline:
    def __init__(self, 
                 net_conf: list[int] = [10, 1],
                 activation_func: ActivationFunction = None,
                 resulting_func: ActivationFunction = None,
                 loss_func: LossFunction = None,
                 corr_vector: np.ndarray = None):
        
        self.params = {}
        self.net_conf = net_conf
        self.net_depth = len(net_conf)

        self.activation_func = activation_func if activation_func else ActivationDummy()
        self.resulting_func = resulting_func if resulting_func else ActivationDummy()
        self.loss_func = loss_func if loss_func else MSELoss()
        self.corr_vector = corr_vector


    def fit(self, X: np.ndarray, Y: np.ndarray, num_epochs: int, batch_size: int, optim: Optimizer, sampling: str = None):
        if X.shape[1] != self.net_conf[0]:
            self.net_conf[0] = X.shape[1]

        self._init_params()
        self.loss_history = []

        if sampling == 'random':
            sampler = RandomBasedSampler()
        elif sampling == 'margin':
            sampler = MarginBasedSampler(temp=0.95)
        else:
            sampler = None
        
        for _ in range(num_epochs):
            epoch_loss = 0
            for l, r in ((i, i + batch_size) for i in range(0, len(X), batch_size)):
                if sampler is not None:
                    margin = self._calculate_margin(X, Y)
                    batch_indices = sampler.get_batch_indices(X, margin, batch_size)

                    X_batch = X[batch_indices]
                    Y_batch = Y[batch_indices]
                else:
                    X_batch = X[l:r]
                    Y_batch = Y[l:r]

                y_pred = self._forward(X_batch)
                grad_storage = self._backprop(y_pred, Y_batch)
                self._update_params(grad_storage, optim)

                batch_loss = self.loss_func.get_loss(y_pred, Y_batch)
                epoch_loss += batch_loss * len(X_batch)
            epoch_loss /= len(X)
            self.loss_history.append(epoch_loss)
            

    def predict(self, X: np.ndarray, bin_classifier: bool = False):
        y_pred = self._forward(X)

        if bin_classifier:
            y_pred = np.sign(y_pred)

        return y_pred


    def _forward(self, X: np.ndarray):
        layer_in = X
        for i in range(self.net_depth - 1):
            layer_weights = self.params[f'layer_{i}']['weights']
            layer_bias = self.params[f'layer_{i}']['biases']
            activation_func = self.activation_func if i != self.net_depth - 2 else self.resulting_func
            output = self._forward_layer(layer_in, layer_weights, layer_bias, activation_func)
            
            self.params[f'layer_{i}']['input'] = layer_in
            self.params[f'layer_{i}']['signal_sum'] = np.dot(layer_in, layer_weights.T) + layer_bias

            layer_in = output

        return output


    @staticmethod
    def _forward_layer(layer_in: np.ndarray, layer_weights: np.ndarray, layer_bias: np.ndarray, activation_func: ActivationFunction):
        layer_out = np.dot(layer_in, layer_weights.T) + layer_bias
        layer_out = activation_func.apply(layer_out)
        
        return layer_out
    

    def _backprop(self, y_pred: np.ndarray, y_truth: np.ndarray):
        grad_storage = {}

        grad = self.loss_func.get_grad(y_pred, y_truth)
        for i in range(self.net_depth - 2, -1, -1):
            signal_sum = self.params[f'layer_{i}']['signal_sum']
            layer_in = self.params[f'layer_{i}']['input']
            weights = self.params[f'layer_{i}']['weights']
            activation_func = self.activation_func if i != self.net_depth - 2 else self.resulting_func

            grad *= activation_func.get_derivative(signal_sum)
            d_weights = np.dot(grad.T, layer_in)
            d_bias = np.sum(grad, axis=0)

            grad_storage[f'layer_{i}'] = {
                'weights': d_weights,
                'biases': d_bias
            }

            # переносим градиент на следующий слой
            grad = np.dot(grad, weights).reshape(layer_in.shape)
        
        return grad_storage


    def _init_params(self, ):
        for i in range(self.net_depth - 1):
            self.params.update({
                f'layer_{i}': {
                    'weights': np.array(
                        [[self._init_weight(i, input_neuron_id) for input_neuron_id in range(self.net_conf[i])] for _ in range(self.net_conf[i+1])]
                    ),
                    'biases': np.array([self._init_bias()] * self.net_conf[i+1])
                }
            })


    def _update_params(self, grad_storage: dict, optim: Optimizer):
        optim.update(self.params, grad_storage)


    def _init_weight(self, layer_id: int, input_neuron_id: int):

        if layer_id == 0 and self.corr_vector is not None:
            weight = self.corr_vector[input_neuron_id] * 0.01
        else:
            weight = np.random.randn() * 0.01

        return weight
    

    def _init_bias(self, ):
        return 1.0
    

    def evaluate(self, X: np.ndarray, Y: np.ndarray, bin_classifier: bool = False):
        y_pred = self.predict(X, bin_classifier=bin_classifier)
        loss = self.loss_func.get_loss(y_pred, Y)

        if bin_classifier:
            y_true = (Y > 0.5).astype(int).ravel()
            y_pred_bin = (y_pred > 0.5).astype(int).ravel()

            accuracy = np.mean(y_true == y_pred_bin)
            precision = precision_score(y_true, y_pred_bin, zero_division=0)
            recall = recall_score(y_true, y_pred_bin, zero_division=0)
            f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        else:
            accuracy = 1.0 - np.mean(np.abs(y_pred - Y) > 0.5)
            precision = recall = f1 = None

        return {
            "accuracy": accuracy,
            "loss": loss,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    

    def evaluate_sequence(self, X_seq: np.ndarray, Y_seq: np.ndarray, alpha: float = 0.95):
        Q = 0 # рекуррентная оценка качества

        for step in range(X_seq.shape[0]):
            y_pred = self._forward(X_seq[step:step+1])
            step_loss = self.loss_func.get_loss(y_pred, Y_seq[step:step+1])
            Q *= alpha
            Q += step_loss
        
        return Q
    

    def margin_statistics(self, X: np.ndarray, Y: np.ndarray):
        margin_stats = {}
        margin_vec = self._calculate_margin(X, Y)

        margin_stats.update({
            'min': np.min(margin_vec),
            'max': np.max(margin_vec),
            'mean': np.mean(margin_vec),
            'median': np.median(margin_vec),
            'std': np.std(margin_vec),
        })

        return margin_stats
    

    def margin_plot(self, X: np.ndarray, Y: np.ndarray, 
                    red_thresh: float = -0.3, yellow_thresh: float = 0.3):
        margin_vec = self._calculate_margin(X, Y).ravel()
        sorted_margins = np.sort(margin_vec)
        n = len(sorted_margins)

        red_mask = sorted_margins < red_thresh
        yellow_mask = (sorted_margins >= red_thresh) & (sorted_margins <= yellow_thresh)
        green_mask = sorted_margins > yellow_thresh

        plt.figure(figsize=(10, 5))
        plt.plot(range(n), sorted_margins, color="blue", lw=1)

        plt.fill_between(range(n), sorted_margins, 0, where=red_mask,
                        color="red", alpha=0.6, label="шумовые")
        plt.fill_between(range(n), sorted_margins, 0, where=yellow_mask,
                        color="yellow", alpha=0.6, label="пограничные")
        plt.fill_between(range(n), sorted_margins, 0, where=green_mask,
                        color="green", alpha=0.6, label="надёжные")

        plt.xlabel("Объект")
        plt.ylabel("Размер отступа")
        plt.title("Распределение отступа по объектам выборки")
        plt.legend()
        plt.grid(True)
        plt.show()

    
    def plot_loss(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, label="Функция потерь", color="blue")
        plt.xlabel("Эпоха")
        plt.ylabel("Значение функции")
        plt.title("График функции потерь на обучающей выборке")
        plt.legend()
        plt.grid(True)
        plt.show()

    
    def plot_roc(self, X: np.ndarray, Y: np.ndarray):
        y_true = (Y > 0).astype(int)
        y_score = self.predict(X, bin_classifier=False).ravel()

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC-кривая (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("FP Rate")
        plt.ylabel("TP Rate")
        plt.title("ROC-кривая")
        plt.legend(loc="lower right")
        plt.show()


    def _calculate_margin(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        y_pred = self._forward(X)
        margin_vec = y_pred * Y

        return margin_vec
    

    def summary(self, ):
        for layer_key in self.params.keys():
            print(f'Model {self.__class__.__name__}: \n')
            print(f'{layer_key} parameter values: \n')
            print(f"{self.params[layer_key]['weights']} weights: \n")
            print(f"{self.params[layer_key]['biases']} biases: \n")
        