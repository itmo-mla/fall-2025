import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import os

PLOTS_DIR = "plots_logistic"
os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv('heart.csv')
# print("Первые 5 строк данных:")
# print(df.head())
# print("\nИнформация о данных:")
# print(df.info())
# print("\nРаспределение целевой переменной:")
# print(df['target'].value_counts())

X = df.drop('target', axis=1)
y = df['target']
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)
y_transform = np.where(y == 0, -1, 1)

X_temp, X_test, y_temp, y_test = train_test_split(X_scaler, y_transform, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
# print(f"\nРазмер обучающей выборки: {X_train.shape}")
# print(f"Размер валидационной выборки: {X_val.shape}")
# print(f"Размер тестовой выборки: {X_test.shape}")


class MyLinearClassifier:
    def __init__(self, n_features, init_strategy='random', regularization_coef=0.01):
        self.n_features = n_features
        self.regularization_coef = regularization_coef

        if init_strategy == 'random':
            self.weights = np.random.randn(n_features) * 0.01
        elif init_strategy == 'correlation':
            self.weights = np.zeros(n_features)

        self.bias = 0
        self.margin_history = []
        self.velocity_w = np.zeros(n_features)
        self.velocity_b = 0.0
        self.ema_loss = None
        self.step_count = 0
        self.loss_history = []
        self.accuracy_history = []
        self.train_accuracy_history = []

    def initialize_with_correlation(self, X, y):
        """Инициализация весов через корреляцию с целевой переменной"""
        correlations = []
        for i in range(X.shape[1]):
            correlation = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(correlation if not np.isnan(correlation) else 0)

        self.weights = np.array(correlations) * 0.1
        self.bias = np.mean(y) * 0.1

    def discriminant_func(self, X):
        """g(x, w) = ⟨x, w⟩"""
        return X.dot(self.weights) + self.bias

    def margin(self, X, y):
        """M_i(w) = g(x_i, w)y_i"""
        return self.discriminant_func(X) * y

    def logistic_loss(self, margin):
        """Логистическая функция потерь L(M) = log(1 + exp(-M))"""
        # ограничиваем отступ для численной устойчивости
        safe_margin = np.clip(margin, -500, 500)
        # exp(-M)
        exp_neg_m = np.exp(-safe_margin)
        # 1 + exp(-M)
        one_plus_exp = 1.0 + exp_neg_m
        # логарифм
        loss = np.log(one_plus_exp)
        return loss

    def logistic_loss_gradient(self, margin, x, y):
        """Градиент логистической потери по весам -σ(-M) * y * x"""
        # ограничиваем отступ для численной устойчивости
        # при margin < -500: σ(-M) примерно 1 (полная ошибка)
        # при margin > 500: σ(-M) примерно 0 (полная уверенность)
        safe_margin = np.clip(margin, -500, 500)

        # вычисляем σ(-M) = 1 / (1 + exp(M))
        # σ(-M) - это вероятность того, что модель ошибается
        # при M -> -∞: σ(-M) → 1 (100% ошибка)
        # при M -> +∞: σ(-M) → 0 (0% ошибка)
        # при M = 0: σ(-M) = 0.5 (50% ошибка)
        sigma_neg_m = 1 / (1 + np.exp(safe_margin))

        # вычисляем градиент ∂L/∂w = -σ(-M) * y * x
        # -σ(-M) отрицательная вероятность ошибки (направление антиградиента)
        # y истинная метка (-1 или +1)
        # x вектор признаков объекта
        gradient = -sigma_neg_m * y * x

        return gradient

    def logistic_loss_gradient_bias(self, margin, y):
        """Градиент логистической потери по смещению -σ(-M) * y"""
        # ограничиваем отступ для численной устойчивости, те же причины, что и для градиента по весам
        safe_margin = np.clip(margin, -500, 500)

        # вычисляем σ(-M) = 1 / (1 + exp(M))
        sigma_neg_m = 1 / (1 + np.exp(safe_margin))

        # вычисляем градиент ∂L/∂b = -σ(-M) * y
        # Отличается от градиента по весам только отсутствием x, смещение b влияет на все признаки одинаково
        gradient = -sigma_neg_m * y

        return gradient

    def compute_regularization_gradient(self):
        """
        Градиент L2 регуляризации: τ * w
        """
        return self.regularization_coef * self.weights

    def compute_regularization_gradient_bias(self):
        """
        Градиент L2 регуляризации для смещения (обычно не регуляризуется)
        """
        return 0

    def loss_L2(self, X, y):
        """
        Q̃(w) = Σ L(w, x_i) + (τ/2) * ‖w‖²
        """
        margins = self.margin(X, y)
        losses = self.logistic_loss(margins)
        L2_reg = (self.regularization_coef / 2) * np.sum(self.weights ** 2)
        return np.mean(losses) + L2_reg

    def gradient(self, X_batch, y_batch):
        """
        ∇Q̃(w) = ∇L(w, x_i) + τ * w
        """
        batch_size = len(y_batch)
        grad_w = np.zeros(self.n_features)
        grad_b = 0

        for i in range(batch_size):
            x_i = X_batch[i]
            y_i = y_batch[i]

            # вычисляем отступ для текущего объекта
            margin = self.margin_single(x_i, y_i)

            # градиент функции потерь
            grad_w_i = self.logistic_loss_gradient(margin, x_i, y_i)
            grad_b_i = self.logistic_loss_gradient_bias(margin, y_i)
            grad_w += grad_w_i
            grad_b += grad_b_i

        # усредняем по батчу и добавляем регуляризацию
        avg_grad_w = grad_w / batch_size + self.compute_regularization_gradient()
        avg_grad_b = grad_b / batch_size + self.compute_regularization_gradient_bias()

        return avg_grad_w, avg_grad_b

    def margin_single(self, x, y):
        """Отступ для одного объекта: M = (⟨x, w⟩ + b) * y"""
        return (np.dot(x, self.weights) + self.bias) * y

    def update_ema_loss(self, current_loss, lambda_ema=0.01):
        """
        Экспоненциальное скользящее среднее для оценки потерь
        Q̄m = λξ_m + (1 - λ)Q̄{m-1}
        """
        if self.ema_loss is None:
            self.ema_loss = current_loss
        else:
            self.ema_loss = lambda_ema * current_loss + (1 - lambda_ema) * self.ema_loss
        return self.ema_loss

    def update_weights_nesterov(self, X_batch, y_batch, learning_rate=0.01, gamma=0.9):
        """
        Ускоренный градиент Нестерова
        v := γv + (1-γ)ℒ'(w - hγv, x_i)
        w := w - hv
        """
        original_weights = self.weights.copy()
        original_bias = self.bias

        # делаем прикидочный шаг по инерции
        self.weights = original_weights - learning_rate * gamma * self.velocity_w
        self.bias = original_bias - learning_rate * gamma * self.velocity_b

        # вычисляем градиент в будущей точке
        lookahead_grad_w, lookahead_grad_b = self.gradient(X_batch, y_batch)

        self.weights = original_weights
        self.bias = original_bias

        self.velocity_w = gamma * self.velocity_w + (1 - gamma) * lookahead_grad_w
        self.velocity_b = gamma * self.velocity_b + (1 - gamma) * lookahead_grad_b

        self.weights -= learning_rate * self.velocity_w
        self.bias -= learning_rate * self.velocity_b

    def steepest_gradient_step(self, X_batch, y_batch):
        grad_w, grad_b = self.gradient(X_batch, y_batch)
        # аппроксимируем гессиан через конечные разности
        epsilon = 1e-10

        original_weights = self.weights.copy()
        original_bias = self.bias
        direction_w = -grad_w # направление антиградиента
        direction_b = -grad_b

        # аппроксимируем вторую производную вдоль направления, первая производная в точке w
        f_prime = np.dot(grad_w, direction_w) + grad_b * direction_b

        # вычисляем градиент в точке w + epsilon * direction
        self.weights = original_weights + epsilon * direction_w
        self.bias = original_bias + epsilon * direction_b
        grad_w_plus, grad_b_plus = self.gradient(X_batch, y_batch)

        # вторая производная
        f_double_prime = (np.dot(grad_w_plus, direction_w) + grad_b_plus * direction_b - f_prime) / epsilon

        # оптимальный шаг
        if abs(f_double_prime) > 1e-10:
            optimal_step = -f_prime / f_double_prime
        else:
            optimal_step = 0.01

        self.weights = original_weights + optimal_step * direction_w
        self.bias = original_bias + optimal_step * direction_b

        return optimal_step

    def sample_by_margin(self, X, y, strategy='uncertainty'):
        """
        Предъявление объектов по модулю отступа
        """
        margins = self.margin(X, y)
        abs_margins = np.abs(margins)

        if strategy == 'uncertainty':
            # чаще берем объекты с меньшей уверенностью (меньший |M_i|)
            probabilities = 1.0 / (abs_margins + 1e-10)
        elif strategy == 'hard_only':
            # берем только трудные объекты (M_i < 1)
            mask = margins < 1
            if np.any(mask):
                probabilities = np.zeros_like(margins)
                # для трудных объектов используем величину потерь
                probabilities[mask] = self.logistic_loss(margins[mask])
            else:
                probabilities = np.ones_like(margins) / len(margins)
        else:
            # равномерное распределение
            probabilities = np.ones_like(margins) / len(margins)

        # нормализуем вероятности
        probabilities = probabilities / np.sum(probabilities)

        # выбираем случайный индекс
        idx = np.random.choice(len(X), p=probabilities)
        return idx, margins[idx]

    def analyze_margins(self, X, y, title="Анализ отступов"):
        margins = self.margin(X, y)
        print(f"\n=== {title} ===")
        print(f"Минимальный отступ: {margins.min():.4f}")
        print(f"Максимальный отступ: {margins.max():.4f}")
        print(f"Средний отступ: {margins.mean():.4f}")
        print(f"Медианный отступ: {np.median(margins):.4f}")
        print(f"Стандартное отклонение: {margins.std():.4f}")
        print(f"Доля объектов с отрицательным отступом: {np.mean(margins < 0):.2%}")
        print(f"Доля объектов с положительным отступом: {np.mean(margins > 0):.2%}")
        print(f"Доля объектов на границе решения (|M| < 0.5): {np.mean(np.abs(margins) < 0.5):.2%}")
        print(f"Доля объектов с уверенной классификацией (|M| > 1): {np.mean(np.abs(margins) > 1):.2%}")
        for class_label in [-1, 1]:
            class_margins = margins[y == class_label]
            if len(class_margins) > 0:
                print(f"\nКласс {class_label}:")
                print(f"  Средний отступ: {class_margins.mean():.4f}")
                print(f"  Минимальный отступ: {class_margins.min():.4f}")
                print(f"  Доля отрицательных отступов: {np.mean(class_margins < 0):.2%}")
        return margins

    def track_margin_statistics(self, X, y, epoch):
        """
        Отслеживание статистики отступов во время обучения
        """
        margins = self.margin(X, y)
        stats = {
            'epoch': epoch,
            'mean_margin': np.mean(margins),
            'std_margin': np.std(margins),
            'min_margin': np.min(margins),
            'max_margin': np.max(margins),
            'negative_margin_ratio': np.mean(margins < 0),
            'positive_margin_ratio': np.mean(margins > 0)
        }
        self.margin_history.append(stats)
        return stats

    def plot_margin_evolution(self):
        """
        Визуализация эволюции отступов во время обучения
        """
        epochs = [stats['epoch'] for stats in self.margin_history]
        mean_margins = [stats['mean_margin'] for stats in self.margin_history]
        negative_ratios = [stats['negative_margin_ratio'] for stats in self.margin_history]

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, mean_margins, 'b-', linewidth=2)
        plt.xlabel('Эпоха')
        plt.ylabel('Средний отступ')
        plt.title('Эволюция среднего отступа')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, negative_ratios, 'r-', linewidth=2)
        plt.xlabel('Эпоха')
        plt.ylabel('Доля отрицательных отступов')
        plt.title('Эволюция ошибок классификации')
        plt.grid(True, alpha=0.3)

        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(PLOTS_DIR, "margin_evolution.png"), dpi=150, bbox_inches='tight')
        # plt.show()

    def plot_margin_ranking(self, X, y, title="Распределение отступов по рангу"):
        margins = self.margin(X, y)
        sorted_indices = np.argsort(margins)
        sorted_margins = margins[sorted_indices]
        x = np.arange(len(sorted_margins))
        plt.figure(figsize=(12, 6))

        # красная зона M < 0
        mask_noise = sorted_margins < 0
        if np.any(mask_noise):
            plt.fill_between(x[mask_noise], sorted_margins[mask_noise], 0, color='red', alpha=0.3, label='Шумовые (M < 0)')

        # желтая зона 0 <= M < 1
        mask_boundary = (sorted_margins >= 0) & (sorted_margins < 1)
        if np.any(mask_boundary):
            plt.fill_between(x[mask_boundary], sorted_margins[mask_boundary], 0, color='yellow', alpha=0.3, label='Пограничные (0 <= M < 1)')

        # зелёная зона M >= 1
        mask_confident = sorted_margins >= 1
        if np.any(mask_confident):
            plt.fill_between(x[mask_confident], sorted_margins[mask_confident], 0, color='green', alpha=0.3, label='Надёжные (M >= 1)')

        plt.plot(x, sorted_margins, 'b-', linewidth=2, label='Отступы')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=1, color='black', linestyle='--', linewidth=1)
        min_margin = np.min(sorted_margins)
        max_margin = np.max(sorted_margins)
        plt.ylim(min_margin - 0.1, max(1.1, max_margin + 0.1))
        plt.xlabel('Индекс объекта')
        plt.ylabel('Отступ')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
        plt.savefig(os.path.join(PLOTS_DIR, f"{safe_title}.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def predict(self, X):
        return np.sign(self.discriminant_func(X))

    def fit(self, X_train, y_train, X_val=None, y_val=None, n_epochs=100, batch_size=32, learning_rate=0.01, gamma=0.9,
            optimizer='sgd', sampling_strategy='random', verbose=True, track_margins=False):
        """
        Обучение модели с логистической функцией потерь и L2 регуляризацией
        """
        n_samples = X_train.shape[0]
        self.loss_history = []
        self.accuracy_history = []
        self.train_accuracy_history = []
        self.margin_history = []

        for epoch in range(n_epochs):
            epoch_losses = []

            if sampling_strategy == 'random':
                # перемешиваем данные
                indices = np.random.permutation(n_samples)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]
            else:
                # используем стратегию выборки по отступам
                X_shuffled = X_train.copy()
                y_shuffled = y_train.copy()

            for i in range(0, n_samples, batch_size):
                if sampling_strategy == 'random':
                    if i + batch_size > n_samples:
                        continue
                    X_batch = X_shuffled[i:i + batch_size]
                    y_batch = y_shuffled[i:i + batch_size]
                else:
                    # выбираем батч по стратегии отступов
                    batch_indices = []
                    for _ in range(batch_size):
                        idx, _ = self.sample_by_margin(X_train, y_train, sampling_strategy)
                        batch_indices.append(idx)
                    X_batch = X_train[batch_indices]
                    y_batch = y_train[batch_indices]

                if optimizer == 'nesterov':
                    self.update_weights_nesterov(X_batch, y_batch, learning_rate, gamma)
                elif optimizer == 'steepest':
                    self.steepest_gradient_step(X_batch, y_batch)
                elif optimizer == 'sgd':
                    grad_w, grad_b = self.gradient(X_batch, y_batch)
                    self.weights -= learning_rate * grad_w
                    self.bias -= learning_rate * grad_b

                loss = self.loss_L2(X_batch, y_batch)
                epoch_losses.append(loss)
                self.update_ema_loss(loss)

            avg_loss = np.mean(epoch_losses)
            self.loss_history.append(avg_loss)

            train_acc = self.score(X_train, y_train)
            self.train_accuracy_history.append(train_acc)

            if X_val is not None and y_val is not None:
                val_acc = self.score(X_val, y_val)
                self.accuracy_history.append(val_acc)

            # отслеживание статистики отступов
            if track_margins and epoch % 5 == 0:
                self.track_margin_statistics(X_train, y_train, epoch)

            if verbose and epoch % 10 == 0:
                margin_info = f", Mean Margin = {self.margin_history[-1]['mean_margin']:.4f}" if track_margins and self.margin_history else ""
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Train Acc = {train_acc:.4f}" +
                      (f", Val Acc = {val_acc:.4f}" if X_val is not None else "") + margin_info)

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions, zero_division=0),
            'f1': f1_score(y, predictions, zero_division=0)
        }
        return metrics


tau = 0.01
all_models = []
all_model_names = []

print("\n=== Обучение с инициализацией через корреляцию ===")
model_corr = MyLinearClassifier(X_train.shape[1], init_strategy='correlation', regularization_coef=tau)
model_corr.initialize_with_correlation(X_train, y_train)
model_corr.fit(X_train, y_train, X_val, y_val, n_epochs=50, learning_rate=0.01, optimizer='sgd', track_margins=True)
print("\n--- Анализ отступов для модели с инициализацией через корреляцию ---")
model_corr.plot_margin_evolution()
all_models.append(model_corr)
all_model_names.append('Correlation init')

print("\n=== Мультистарт со случайной инициализацией ===")
n_restarts = 3
multistart_models = []

for i in range(n_restarts):
    print(f"\n--- Запуск {i + 1}/{n_restarts} ---")
    model = MyLinearClassifier(X_train.shape[1], init_strategy='random', regularization_coef=tau)
    model.fit(X_train, y_train, X_val, y_val, n_epochs=30, learning_rate=0.01, optimizer='nesterov', verbose=False,
              track_margins=True)

    val_acc = model.score(X_val, y_val)
    print(f"Validation Accuracy: {val_acc:.4f}")

    print(f"--- Анализ отступов для запуска {i + 1} ---")
    margins_val = model.margin(X_val, y_val)
    print(f"Средний отступ на валидации: {np.mean(margins_val):.4f}")
    print(f"Доля ошибок на валидации: {np.mean(margins_val < 0):.2%}")

    multistart_models.append((model, val_acc))

best_multistart_model, best_multistart_acc = max(multistart_models, key=lambda x: x[1])
all_models.append(best_multistart_model)
all_model_names.append('Multistart best')

print("\n=== Обучение с выбором объектов по отступам ===")
model_margin = MyLinearClassifier(X_train.shape[1], init_strategy='random', regularization_coef=tau)
model_margin.fit(X_train, y_train, X_val, y_val, n_epochs=50, learning_rate=0.01, optimizer='sgd', sampling_strategy='uncertainty', track_margins=True)
print("\n--- Анализ отступов для модели с выбором по отступам ---")
model_margin.analyze_margins(X_train, y_train, "Обучающая выборка - Margin sampling")
model_margin.analyze_margins(X_val, y_val, "Валидационная выборка - Margin sampling")
model_margin.plot_margin_evolution()
all_models.append(model_margin)
all_model_names.append('Margin sampling')

print("\n=== Обучение со случайной инициализацией ===")
model_random = MyLinearClassifier(X_train.shape[1], init_strategy='random', regularization_coef=tau)
model_random.fit(X_train, y_train, X_val, y_val, n_epochs=50, learning_rate=0.01, optimizer='sgd', track_margins=True)
print("\n--- Анализ отступов для модели со случайной инициализацией ---")
model_random.analyze_margins(X_train, y_train, "Обучающая выборка - Random init")
model_random.analyze_margins(X_val, y_val, "Валидационная выборка - Random init")
model_random.plot_margin_evolution()
all_models.append(model_random)
all_model_names.append('Random init')

print("\n=== Обучение с Nesterov momentum ===")
model_nesterov = MyLinearClassifier(X_train.shape[1], init_strategy='random', regularization_coef=tau)
model_nesterov.fit(X_train, y_train, X_val, y_val, n_epochs=50, learning_rate=0.01, optimizer='nesterov', gamma=0.9, track_margins=True)
print("\n--- Анализ отступов для модели с Nesterov momentum ---")
model_nesterov.analyze_margins(X_train, y_train, "Обучающая выборка - Nesterov")
model_nesterov.analyze_margins(X_val, y_val, "Валидационная выборка - Nesterov")
model_nesterov.plot_margin_evolution()
all_models.append(model_nesterov)
all_model_names.append('Nesterov')

print("\n" + "=" * 60)
print("ВЫБОР ЛУЧШЕЙ МОДЕЛИ")
print("=" * 60)

best_model = None
best_val_acc = 0
best_model_name = ""

for model, name in zip(all_models, all_model_names):
    val_acc = model.score(X_val, y_val)
    print(f"{name}: Validation аccuracy = {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model
        best_model_name = name

print(f"\nЛучшая модель: {best_model_name} с точностью на валидации: {best_val_acc:.4f}")

print("\n=== Детальный анализ отступов для лучшей модели ===")
best_model.plot_margin_ranking(X_train, y_train, f"Обучающая выборка - {best_model_name} - Отступы по рангу")
best_model.plot_margin_ranking(X_val, y_val, f"Валидационная выборка - {best_model_name} - Отступы по рангу")
best_model.plot_margin_ranking(X_test, y_test, f"Тестовая выборка - {best_model_name} - Отступы по рангу")
best_model.plot_margin_evolution()
print("Графики распределения отступов сохранены в plots_logistic")

print("\n--- Статистика отступов лучшей модели ---")
margins_train = best_model.margin(X_train, y_train)
margins_val = best_model.margin(X_val, y_val)
margins_test = best_model.margin(X_test, y_test)

datasets = [
    ("Обучающая", margins_train),
    ("Валидационная", margins_val),
    ("Тестовая", margins_test)
]

for name, margins in datasets:
    print(f"\n{name} выборка:")
    print(f"Средний отступ: {np.mean(margins):.4f}")
    print(f"Медианный отступ: {np.median(margins):.4f}")
    print(f"Min отступ: {np.min(margins):.4f}")
    print(f"Max отступ: {np.max(margins):.4f}")
    print(f"Доля ошибок (M < 0): {np.mean(margins < 0):.2%}")
    print(f"Доля уверенных (|M| > 1): {np.mean(np.abs(margins) > 1):.2%}")

print("\n=== Оценка лучшей модели ===")
test_metrics = best_model.evaluate(X_test, y_test)
print("Метрики на тестовой выборке:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\n=== Сравнение с эталонной моделью ===")
lr_model = LogisticRegression(penalty='l2', C=1.0 / tau, random_state=42, max_iter=1000)
lr_model.fit(X_train, np.where(y_train == -1, 0, 1))
lr_pred = lr_model.predict(X_test)
lr_pred_transformed = np.where(lr_pred == 0, -1, 1)
lr_metrics = {
    'accuracy': accuracy_score(y_test, lr_pred_transformed),
    'precision': precision_score(y_test, lr_pred_transformed, zero_division=0),
    'recall': recall_score(y_test, lr_pred_transformed, zero_division=0),
    'f1': f1_score(y_test, lr_pred_transformed, zero_division=0)
}

print("Метрики Logistic regression:")
for metric, value in lr_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nРазница в accuracy (моя модель - эталонная):")
print(f"{test_metrics['accuracy'] - lr_metrics['accuracy']:.4f}")


def plot_training_history(models, model_names):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    for i, model in enumerate(models):
        plt.plot(model.loss_history, label=model_names[i])
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.title('Функция потерь во время обучения')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for i, model in enumerate(models):
        if hasattr(model, 'accuracy_history') and model.accuracy_history:
            plt.plot(model.accuracy_history, label=model_names[i])
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.title('Точность на валидационной выборке')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(PLOTS_DIR, "training_history.png"), dpi=150, bbox_inches='tight')
    # plt.show()


plot_training_history(all_models, all_model_names)

print("\n" + "=" * 80)
print("Вывод")
print("=" * 80)

print(f"\nЛучшая модель: {best_model_name}")
print(f"Точность на валидации: {best_val_acc:.4f}")
print(f"Точность на тесте: {test_metrics['accuracy']:.4f}")
print(f"Точность эталонной модели: {lr_metrics['accuracy']:.4f}")
print(f"Разница: {test_metrics['accuracy'] - lr_metrics['accuracy']:.4f}")

if test_metrics['accuracy'] >= lr_metrics['accuracy']:
    print("Моя модель показала сопоставимое или лучшее качество!")
else:
    print("Эталонная модель показала лучшее качество")

print("\nСравнение всех моделей на валидации:")
for model, name in zip(all_models, all_model_names):
    val_acc = model.score(X_val, y_val)
    print(f"{name}: {val_acc:.4f}")

print("\nДетальные метрики лучшей модели:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\n" + "=" * 50)
print("Анализ отступов лучшей модели")
print("=" * 50)

margins_train = best_model.margin(X_train, y_train)
margins_val = best_model.margin(X_val, y_val)
margins_test = best_model.margin(X_test, y_test)

print(f"\nОбучающая выборка:")
print(f"  Средний отступ: {np.mean(margins_train):.4f}")
print(f"  Доля ошибок: {np.mean(margins_train < 0):.2%}")

print(f"\nВалидационная выборка:")
print(f"  Средний отступ: {np.mean(margins_val):.4f}")
print(f"  Доля ошибок: {np.mean(margins_val < 0):.2%}")

print(f"\nТестовая выборка:")
print(f"  Средний отступ: {np.mean(margins_test):.4f}")
print(f"  Доля ошибок: {np.mean(margins_test < 0):.2%}")

print(f"\nУверенность классификации на тесте:")
print(f"  Доля объектов с |M| > 1: {np.mean(np.abs(margins_test) > 1):.2%}")
print(f"  Доля объектов с |M| > 2: {np.mean(np.abs(margins_test) > 2):.2%}")
print(f"  Доля объектов на границе (|M| < 0.5): {np.mean(np.abs(margins_test) < 0.5):.2%}")

print(f"\nАнализ по классам на тестовой выборке:")
for class_label in [-1, 1]:
    class_margins = margins_test[y_test == class_label]
    if len(class_margins) > 0:
        print(f"Класс {class_label}:")
        print(f"  Средний отступ: {np.mean(class_margins):.4f}")
        print(f"  Доля ошибок: {np.mean(class_margins < 0):.2%}")
        print(f"  Минимальный отступ: {np.min(class_margins):.4f}")
        print(f"  Максимальный отступ: {np.max(class_margins):.4f}")
