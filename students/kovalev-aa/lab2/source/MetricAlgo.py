import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils import plot_risk_summary

class MetricClasifier:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.k = 0
        self.nearest_idx = None
        self.classes, self.y_idx = np.unique(y_train, return_inverse=True)

    def evklid_distance_matrix(self, x1, x2):
        """Матрица евклидовых расстояний"""
        X1_sq = np.sum(x1**2, axis=1, keepdims=True)
        X2_sq = np.sum(x2**2, axis=1, keepdims=True)
        dist_sq = X1_sq + X2_sq.T - 2 * (x1 @ x2.T)
        dist_sq[dist_sq < 0] = 0
        return np.sqrt(dist_sq)

    def parzen_window(self, x_test, x_train, y_idx, k):
        """Взвешенные голоса соседей по расстоянию (как в sklearn)"""
        diff_matrix = self.evklid_distance_matrix(x_test, x_train)
        self.nearest_idx = np.argsort(diff_matrix, axis=1)[:, :k]
        distances = np.take_along_axis(diff_matrix, self.nearest_idx, axis=1)
        distances[distances == 0] = 1e-10  # чтобы не делить на 0
        weights = 1 / distances

        # суммируем веса по классам
        weighted_votes = np.zeros((x_test.shape[0], len(self.classes)))
        for i, cls in enumerate(self.classes):
            mask = (y_idx[self.nearest_idx] == i)
            weighted_votes[:, i] = np.sum(weights * mask, axis=1)
        return weighted_votes

    def loo_loss(self, x_train, y_train, k):
        """Ошибка Leave-One-Out (для подбора k и эталонов)"""
        n = x_train.shape[0]
        _, y_idx = np.unique(y_train, return_inverse=True)
        errors = []
        for i in range(n):
            mask = np.arange(n) != i
            x_sub = x_train[mask]
            y_sub_idx = y_idx[mask]
            x_test = x_train[i:i+1]
            y_true_idx = y_idx[i]
            weights = self.parzen_window(x_test, x_sub, y_sub_idx, k)
            pred = np.argmax(weights[0])
            errors.append(int(pred != y_true_idx))
        return errors

    def train_k(self, is_plot=True):
        """Автоподбор k через LOO"""
        n = self.x_train.shape[0]
        k_array = np.arange(1, int(np.sqrt(n)) + 1)
        k_errors = np.zeros((len(k_array), n))
        for idx, k in enumerate(k_array):
            k_errors[idx] = self.loo_loss(self.x_train, self.y_train, k)
        mean_errors = np.mean(k_errors, axis=1)
        self.k = k_array[np.argmin(mean_errors)]
        if is_plot:
            plot_risk_summary(k_array, k_errors)
        print(f"Лучшее k = {self.k}")
        return k_errors

    def standart_select(self):
        """Удаление «плохих» эталонов для улучшения качества"""
        x_train = self.x_train
        y_train = self.y_train
        n = x_train.shape[0]
        selected_mask = np.ones(n, dtype=bool)

        prev_errors = np.array(self.loo_loss(x_train, y_train, self.k))
        prev_error_mean = np.mean(prev_errors)

        for i in range(n):
            if not selected_mask[i]:
                continue
            mask = selected_mask.copy()
            mask[i] = False
            x_sub = x_train[mask]
            y_sub = y_train[mask]
            new_errors = np.array(self.loo_loss(x_sub, y_sub, self.k))
            new_error_mean = np.mean(new_errors)
            if new_error_mean <= prev_error_mean:
                selected_mask[i] = False
                prev_error_mean = new_error_mean

        self.x_train = x_train[selected_mask]
        self.y_train = y_train[selected_mask]

    def predict(self, x_test):
        """Предсказание классов"""
        _, y_idx = np.unique(self.y_train, return_inverse=True)
        weights = self.parzen_window(x_test, self.x_train, y_idx, self.k)
        preds_idx = np.argmax(weights, axis=1)
        return self.classes[preds_idx]

    def metrics(self, y_true, y_pred, average='macro'):
        """Расчёт метрик классификации"""
        metrics_dict = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        return metrics_dict
