import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
import matplotlib.pyplot as plt



class Classifier(ABC):
    def __init__(self):
        self.w = None
        self.b = 0 
        self.regress_error = 0 
        self.margin = None
        self.reg_koef = None
        self.v_w = None
        self.v_b = None
        self.losses = []
        self.emp_losses = []

        self.accuracies = []
        self.test_loss = None
        self.test_accuracy = None
        self.test_precision  = None
        self.test_recall  = None
        self.test_f1  = None
        self.epoches = 0

        self.is_trained = False

    @abstractmethod
    def activation(self, z):
        pass

    @abstractmethod
    def margin_calculate(self, y,x):
        pass

    @abstractmethod
    def gradient(self, x_train, y_train): 
        pass

    @abstractmethod
    def loss(self, x, y):
        pass

    @abstractmethod
    def predict_p(self, x):
        pass

    @abstractmethod
    def regulazer(self):
        pass

    def reset(self):
        """Сброс всех параметров модели к исходному состоянию"""
        self.w = None
        self.b = 0
        self.regress_error = 0
        self.margin = None
        self.reg_koef = None
        self.v_w = None
        self.v_b = None
        self.losses.clear()
        self.emp_losses.clear()
        self.accuracies.clear()
        self.accuracies = []
        self.test_loss = None
        self.test_accuracy = None
        self.test_precision = None
        self.test_recall = None
        self.test_f1 = None
        self.epoches = 0
        self.is_trained = False

    def accuracy(self,y_true, y_pred):
        return np.mean(y_true == y_pred)

    def precision(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == -1) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def recall(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == -1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def f1_score(self, y_true, y_pred):
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    def margin_plot(self):
        if self.is_trained:
            margins = self.margin
            sorted_idx = np.argsort(margins)
            margins_sorted = margins[sorted_idx]

            plt.figure(figsize=(12,5))

            # Определяем границы зон
            noise_end = np.sum(margins_sorted < -0.2)          # шумовые
            borderline_end = np.searchsorted(margins_sorted, 0.2)  # порог для пограничных
            reliable_end = len(margins_sorted)             # надёжные

            # Построение цветных столбцов
            plt.bar(range(noise_end), margins_sorted[:noise_end], color='red', width=1)
            plt.bar(range(noise_end, borderline_end), margins_sorted[noise_end:borderline_end], color='yellow', width=1)
            plt.bar(range(borderline_end, reliable_end), margins_sorted[borderline_end:], color='green', width=1)

            # Синяя линия марджинов
            plt.plot(range(len(margins_sorted)), margins_sorted, color='blue', linewidth=1)

            plt.axhline(0, color='black', linestyle='--')
            plt.xlabel('Samples sorted by margin')
            plt.ylabel('Margin')
            plt.title('Ranking of margins')
            plt.show()
        else:
            raise RuntimeError("Model is not trained yet. Cannot plot margins.")

    def losses_plot(self): 
        if self.is_trained:
            # отбрасываем первый элемент self.losses
            losses_to_plot = self.losses[10:]
            epochs_to_plot = range(10, self.epoches)  # начиная с 1
            plt.figure(figsize=(10,5))
            plt.plot(epochs_to_plot, losses_to_plot, label='RegLoss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training loss over epochs (без нулевого лосса)')
            plt.legend()
            plt.show()
        else:
            raise RuntimeError("Model is not trained yet. Cannot plot margins.")
    
    def emp_plot(self):
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")

        step = 100
        epochs = list(range(0, self.epoches, step))
        losses = self.emp_losses

 
        if epochs[-1] != self.epoches - 1:
            epochs.append(self.epoches - 1)
            losses.append(self.emp_losses[-1])

 
        min_len = min(len(epochs), len(losses))
        epochs = epochs[:min_len]
        losses = losses[:min_len]

        plt.figure(figsize=(10,5))
        plt.plot(epochs, losses, marker='o', label='EmpRisk')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Empirical Risk")
        plt.legend()
        plt.show()


    def desc_metrics(self):
        if  self.is_trained:
            print(f'Test Accuracy: {self.test_accuracy}')
            print(f'Test Loss: {self.test_loss}')
            print(f'Test Precision: {self.test_precision}')
            print(f'Test Recall: {self.test_recall}')
            print(f'Test F1: {self.test_f1}')

        else:
            raise RuntimeError("Model is not trained yet. Cannot plot margins.")




    @abstractmethod
    def batching(self,x_train,y_train, batch_count=3): 
        pass


    @abstractmethod
    def init_w(self,x_train,y_train,method,n_start=10): 
        pass


    def train(self, x_train, y_train, x_test, y_test,
                epoches=500, lr=0.0005, reg_koef=0.01, init_subset=100, beta=0.9,
                batching_method='margin', init_method='correlation',
                log_file='training_log.txt',koef_attenuation=0.001):

        self.epoches = epoches
        self.reg_koef = reg_koef 
        self.regress_error = 0

        # инициализация весов
        if init_method == 'correlation':
            self.init_w(x_train, y_train)
        elif init_method == 'random':
            self.init_w(x_train, y_train, method='random')

        # инициализация инерции
        self.v_w = np.zeros_like(self.w)
        self.v_b = 0

        # инициализация Q
        idx = np.random.choice(len(x_train), size=init_subset, replace=False)
        x_sub, y_sub = x_train[idx], y_train[idx]
        Q_init = self.loss(x_sub, y_sub)
        msg_init = f"[Init] Q on random subset: {Q_init:.6f}"
        print(msg_init)
        with open(log_file, 'w') as f:
            f.write(msg_init + '\n')

        # обучение
        for i in range(epoches):
            batches = self.batching(x_train, y_train, batch_count=10)
            x_sub, y_sub = batches[0]
            grad_w, grad_b = self.gradient(x_sub, y_sub)
            koef_attenuation = 1 / len(batches)

            loss_val = self.loss(x_sub, y_sub)
            emp_loss = self.loss(x_test, y_test)
            emp_loss = self.loss(x_test, y_test)
            if i % 100 == 0 or i == self.epoches - 1:
                self.emp_losses.append(emp_loss)
            # градиентный шаг с инерцией
            self.v_w = beta * self.v_w + lr * grad_w
            self.v_b = beta * self.v_b + lr * grad_b
            self.w -= self.v_w
            self.b -= self.v_b

            self.regress_error = koef_attenuation * loss_val + (1 - koef_attenuation) * self.regress_error
            # emp_error = koef_attenuation * emp_loss + (1 - koef_attenuation) * self.regress_error

            y_pred = self.predict(x_sub)
            accuracy = self.accuracy(y_sub, y_pred)

            self.losses.append(self.regress_error)
 
            self.accuracies.append(accuracy)

            # логирование каждые 10% эпох
            if i % max(1, int(epoches * 0.1)) == 0 or i == epoches - 1:
                msg = f"[Epoch {i:4d}] RegLoss: {self.regress_error:.6f} | Accuracy: {accuracy:.4f}"
                print(msg)
                with open(log_file, 'a') as f:
                    f.write(msg + '\n')

        # вычисляем марджины и тестовые метрики
        self.margin = self.margin_calculate(y=y_test, x=x_test)
        y_pred_test = self.predict(x_test)
        self.test_loss = self.loss(x_test, y_test)
        self.test_accuracy = self.accuracy(y_test, y_pred_test)
        self.test_precision = self.precision(y_test, y_pred_test)
        self.test_recall = self.recall(y_test, y_pred_test)
        self.test_f1 = self.f1_score(y_test, y_pred_test)

        # финальные метрики
        final_msg = (
            f"\n[Test Metrics] Loss: {self.test_loss:.6f} | "
            f"Accuracy: {self.test_accuracy:.4f} | "
            f"Precision: {self.test_precision:.4f} | "
            f"Recall: {self.test_recall:.4f} | "
            f"F1: {self.test_f1:.4f}"
        )
        print(final_msg)
        with open(log_file, 'a') as f:
            f.write(final_msg + '\n')

        self.is_trained = True


    def train_gd(self, x_train, y_train, x_test, y_test, 
                 epoches=500, lr=0.0005, reg_koef=0.01, batch_count=10, log_file='gd_log.txt'):

        self.epoches = epoches
        self.reg_koef = reg_koef
        self.regress_error = 0

        self.init_w(x_train, y_train, method='random')
        self.b = 0

        for i in range(epoches):

            batches = self.batching(x_train, y_train, method='margin', batch_count=batch_count)
            
            grad_w_total = np.zeros_like(self.w)
            grad_b_total = 0

            for x_sub, y_sub in batches:
                gw, gb = self.gradient(x_sub, y_sub)
                grad_w_total += gw
                grad_b_total += gb

            grad_w = grad_w_total / len(batches)
            grad_b = grad_b_total / len(batches)

            self.w -= lr * grad_w
            self.b -= lr * grad_b

            train_loss = self.loss(x_train, y_train)
            test_loss = self.loss(x_test, y_test)

            self.losses.append(train_loss)
            self.emp_losses.append(test_loss)
            self.accuracies.append(self.accuracy(y_train, self.predict(x_train)))

            msg = f"[Epoch {i:4d}] TrainLoss: {train_loss:.6f} | TestLoss: {test_loss:.6f}"
            print(msg)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')

        y_pred_test = self.predict(x_test)
        self.test_loss = self.loss(x_test, y_test)
        self.test_accuracy = self.accuracy(y_test, y_pred_test)
        self.test_precision = self.precision(y_test, y_pred_test)
        self.test_recall = self.recall(y_test, y_pred_test)
        self.test_f1 = self.f1_score(y_test, y_pred_test)

        print("\nGD finished")
        self.is_trained = True


    @abstractmethod
    def predict(self, x):
        pass


class ClassifierLogisticReg(Classifier):
    def activation(self, z):
        return z

    def loss(self, x, y):
        """Лосс для меток ±1"""
        z = np.dot(x,self.w) + self.b
        return np.mean(np.square(z-y)) + self.reg_koef* self.regulazer()

    def gradient(self, x, y):
        """Градиент для меток ±1"""
        z = np.dot(x, self.w) + self.b 
        grad_w = 2/len(x) * (x.T @ (z - y)) + self.reg_koef * self.regulazer(is_diff=True)
        grad_b = 2 * np.mean(z-y)
        return grad_w, grad_b

    def predict_p(self, x):
        z = np.dot(x, self.w) + self.b
        return self.activation(z)  # вероятность класса 1

    def regulazer(self,is_diff = False):
        if is_diff:
            return 2*self.w
        else:
            return np.sum(self.w**2)
    def predict(self, x):
        z = np.dot(x, self.w) + self.b
        return np.sign(z)



    def margin_calculate(self,y,x): 
        z = np.dot(x, self.w) + self.b
        return y*z

    def batching(self, x_train, y_train, method='margin', batch_count=3):
        batch_size = len(x_train) // batch_count
        batches = []
        if method == 'margin':
            # считаем марджины на обучающем наборе
            margin = self.margin_calculate(y=y_train, x=x_train)
            margin_abs = np.abs(margin)
            sample_weight = 1 / (margin_abs + 1e-6)
            sample_weight /= sample_weight.sum()

            

            used_idx = set()
            for _ in range(batch_count):
                # выбираем батч с учётом вероятностей
                available_idx = list(set(range(len(x_train))) - used_idx)
                if len(available_idx) < batch_size:
                    idx = np.array(available_idx)
                else:
                    probs = sample_weight[available_idx]
                    probs /= probs.sum()
                    idx = np.random.choice(available_idx, size=batch_size, replace=False, p=probs)
                used_idx.update(idx)
                batches.append((x_train[idx], y_train[idx]))

            return batches
        if method == 'random':
            used_idx = set()
            for _ in range(batch_count):
                # выбираем батч с учётом вероятностей
                available_idx = list(set(range(len(x_train))) - used_idx)
                if len(available_idx) < batch_size:
                    idx = np.array(available_idx)
                else: 
                    idx = np.random.choice(available_idx, size=batch_size, replace=False)
                used_idx.update(idx)
                batches.append((x_train[idx], y_train[idx]))

            return batches


    def init_w(self, x_train, y_train, method='correlation', n_starts=10): 
        x_train = np.atleast_2d(x_train)  # гарантируем, что x_train 2D
        n_features = x_train.shape[1]
    
        if method == 'correlation':
            w = np.zeros(n_features)
            for j in range(n_features):
                f_j = x_train[:, j]
                w[j] = np.dot(y_train, f_j) / np.dot(f_j, f_j)
            self.w = w
    
        elif method == 'random':
            # случайная инициализация: для мультистарта вернем массив (n_starts, n_features)
            self.multistart(x_train,y_train, np.random.randn(n_starts, n_features),10)
    
        else:
            raise ValueError("Unknown method. Use 'correlation' or 'random'.")

    def multistart(self, x_train, y_train, random_starts, n_steps=5, lr=0.001): 
        best_loss = np.inf
        best_w = None
        best_b = None

        for w_start in random_starts:
            self.w = w_start.copy()
            self.b = 0
            # короткое обучение для оценки
            for _ in range(n_steps):
                for x_sub, y_sub in self.batching(x_train, y_train, method='margin'):
                    grad_w, grad_b = self.gradient(x_sub, y_sub)
                    self.w -= lr * grad_w
                    self.b -= lr * grad_b
            trial_loss = self.loss(x_train, y_train)
            if trial_loss < best_loss:
                best_loss = trial_loss
                best_w = self.w.copy()
                best_b = self.b
        # присваиваем лучшие веса после проверки всех стартов
        self.w = best_w
        self.b = best_b
