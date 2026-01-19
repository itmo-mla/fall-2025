import matplotlib.pyplot as plt
import numpy as np

from linear_classifier import (
    initialize_weights,
    quadratic_loss,
    quadratic_loss_gradient,
    logistic_loss,
    logistic_loss_gradient
)
from margins import calculate_all_margins, calculate_margin

def _mean_quadratic_loss_full(w, X, y):
    """
    Средняя квадратичная потеря на всей выборке (векторизовано):
    L(M) = max(0, 1 - M)^2, где M = y * (Xw)
    """
    margins = y * (X @ w)
    t = np.maximum(0.0, 1.0 - margins)
    return float(np.mean(t * t))


def stochastic_gradient_descent(
    X,
    y,
    w,
    learning_rate=0.01,
    n_epochs=100,
    batch_size=1,
    plot=True,
    *,
    track_full_losses=False,
    X_train_full=None,
    y_train_full=None,
    X_val_full=None,
    y_val_full=None,
):
    
    n_samples = X.shape[0]
    loss_history = []
    train_full_loss_history = []
    val_full_loss_history = []
    
    for epoch in range(n_epochs):
        total_epoch_loss = 0
        num_batches = 0
        
        # Случайное перемешивание данных каждую эпоху
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, batch_size):
            # Берем мини-батч (или один объект если batch_size=1)
            batch_indices = indices[i:i + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            batch_loss = 0
            batch_gradient = np.zeros_like(w)
            
            # Вычисляем градиент для батча
            for j in range(len(X_batch)):
                margin = calculate_margin(w, X_batch[j], y_batch[j])
                loss = quadratic_loss(margin)
                batch_loss += loss
                
                if margin < 1:
                    grad = quadratic_loss_gradient(w, X_batch[j], y_batch[j])
                    batch_gradient += grad
            
            # Усредняем градиент по батчу
            if len(X_batch) > 0:
                batch_gradient /= len(X_batch)
                batch_loss /= len(X_batch)
                
                # Обновляем веса
                w = w - learning_rate * batch_gradient
            
            total_epoch_loss += batch_loss
            num_batches += 1
        
        # Средняя потеря за эпоху
        avg_epoch_loss = total_epoch_loss / max(1, num_batches)
        loss_history.append(avg_epoch_loss)

        if track_full_losses:
            if X_train_full is None or y_train_full is None or X_val_full is None or y_val_full is None:
                raise ValueError("track_full_losses=True требует X_train_full/y_train_full и X_val_full/y_val_full")
            train_full_loss_history.append(_mean_quadratic_loss_full(w, X_train_full, y_train_full))
            val_full_loss_history.append(_mean_quadratic_loss_full(w, X_val_full, y_val_full))
        
        if epoch % 20 == 0:
            print(f'Эпоха {epoch}, средняя потеря: {avg_epoch_loss:.4f}')
    
    if plot:
        sgd_plot(loss_history, '')
    
    if track_full_losses:
        return w, loss_history, train_full_loss_history, val_full_loss_history
    return w, loss_history

def sgd_with_reg(
    X,
    y,
    w,
    learning_rate=0.01,
    n_epochs=100,
    batch_size=32,
    reg_strength=0.01,
    plot=True,
    *,
    track_full_losses=False,
    X_train_full=None,
    y_train_full=None,
    X_val_full=None,
    y_val_full=None,
):
    """
    SGD с L2-регуляризацией
    """
    n_samples = X.shape[0]
    
    loss_history = []
    train_full_loss_history = []
    val_full_loss_history = []
    
    for epoch in range(n_epochs):
        total_epoch_loss = 0
        num_batches = 0
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            batch_loss = 0
            batch_gradient = np.zeros_like(w)
            
            for j in range(len(X_batch)):
                margin = calculate_margin(w, X_batch[j], y_batch[j])
                loss = quadratic_loss(margin)
                batch_loss += loss
                
                if margin < 1:
                    grad = quadratic_loss_gradient(w, X_batch[j], y_batch[j])
                    batch_gradient += grad
            
            if len(X_batch) > 0:
                batch_gradient /= len(X_batch)
                batch_loss /= len(X_batch)
                
                # Градиент L2-регуляризации (исключаем bias)
                reg_gradient = np.copy(w)
                reg_gradient[0] = 0
                batch_gradient += reg_strength * reg_gradient
                
                # Обновляем веса
                w = w - learning_rate * batch_gradient
            
            total_epoch_loss += batch_loss
            num_batches += 1
        
        avg_epoch_loss = total_epoch_loss / max(1, num_batches)
        loss_history.append(avg_epoch_loss)

        if track_full_losses:
            if X_train_full is None or y_train_full is None or X_val_full is None or y_val_full is None:
                raise ValueError("track_full_losses=True требует X_train_full/y_train_full и X_val_full/y_val_full")
            train_full_loss_history.append(_mean_quadratic_loss_full(w, X_train_full, y_train_full))
            val_full_loss_history.append(_mean_quadratic_loss_full(w, X_val_full, y_val_full))
        
        if epoch % 20 == 0:
            print(f"Эпоха {epoch}, средняя потеря: {avg_epoch_loss:.4f}")
    
    if plot:
        sgd_plot(loss_history, ' с L2-регуляризацией')
    
    if track_full_losses:
        return w, loss_history, train_full_loss_history, val_full_loss_history
    return w, loss_history

def sgd_with_momentum(
    X,
    y,
    w,
    learning_rate=0.01,
    n_epochs=100,
    batch_size=32,
    momentum=0.9,
    plot=True,
    *,
    track_full_losses=False,
    X_train_full=None,
    y_train_full=None,
    X_val_full=None,
    y_val_full=None,
):
    """
    SGD с инерцией (momentum)
    """
    n_samples = X.shape[0]
    velocity = np.zeros_like(w)
    
    loss_history = []
    train_full_loss_history = []
    val_full_loss_history = []
    
    for epoch in range(n_epochs):
        total_epoch_loss = 0
        num_batches = 0
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            batch_loss = 0
            batch_gradient = np.zeros_like(w)
            
            for j in range(len(X_batch)):
                margin = calculate_margin(w, X_batch[j], y_batch[j])
                loss = quadratic_loss(margin)
                batch_loss += loss
                
                if margin < 1:
                    grad = quadratic_loss_gradient(w, X_batch[j], y_batch[j])
                    batch_gradient += grad
            
            if len(X_batch) > 0:
                batch_gradient /= len(X_batch)
                batch_loss /= len(X_batch)
                
                # Обновляем скорость
                velocity = momentum * velocity + (1 - momentum) * batch_gradient
                # Обновляем веса
                w = w - learning_rate * velocity
            
            total_epoch_loss += batch_loss
            num_batches += 1
        
        avg_epoch_loss = total_epoch_loss / max(1, num_batches)
        loss_history.append(avg_epoch_loss)

        if track_full_losses:
            if X_train_full is None or y_train_full is None or X_val_full is None or y_val_full is None:
                raise ValueError("track_full_losses=True требует X_train_full/y_train_full и X_val_full/y_val_full")
            train_full_loss_history.append(_mean_quadratic_loss_full(w, X_train_full, y_train_full))
            val_full_loss_history.append(_mean_quadratic_loss_full(w, X_val_full, y_val_full))
    
    if plot:
        sgd_plot(loss_history, ' с инерцией')
    
    if track_full_losses:
        return w, loss_history, train_full_loss_history, val_full_loss_history
    return w, loss_history

def sgd_with_ema(X, y, learning_rate=0.01, n_epochs=100, batch_size=32, lambda_ema=0.01, plot=True):
    """
    SGD с рекуррентной оценкой функционала качества
    """
    n_samples, n_features = X.shape
    w = initialize_weights(n_features)
    
    loss_history = []
    ema_loss = 0
    
    for epoch in range(n_epochs):
        total_epoch_loss = 0
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            batch_loss = 0
            batch_gradient = np.zeros_like(w)
            
            for j in range(len(X_batch)):
                margin = calculate_margin(w, X_batch[j], y_batch[j])
                loss = quadratic_loss(margin)
                batch_loss += loss
                
                if margin < 1:
                    grad = quadratic_loss_gradient(w, X_batch[j], y_batch[j])
                    batch_gradient += grad
            
            if len(X_batch) > 0:
                batch_gradient /= len(X_batch)
                batch_loss /= len(X_batch)
                
                # Обновляем веса
                w = w - learning_rate * batch_gradient
                
                # Рекуррентная оценка функционала
                if ema_loss == 0:
                    ema_loss = batch_loss
                else:
                    ema_loss = lambda_ema * batch_loss + (1 - lambda_ema) * ema_loss
            
            total_epoch_loss += batch_loss
        
        avg_epoch_loss = total_epoch_loss / (n_samples // batch_size)
        loss_history.append(avg_epoch_loss)
        
        if epoch % 20 == 0:
            print(f"Эпоха {epoch}, средняя потеря: {avg_epoch_loss:.4f}, EMA: {ema_loss:.4f}")
    
    if plot:
        sgd_plot(loss_history, ' с рекуррентной оценкой функционала качества')
    
    return w, loss_history, ema_loss

def steepest_gradient_descent(X, y, n_epochs=100, batch_size=32, epsilon=1e-6):
    """
    Скорейший градиентный спуск с адаптивным шагом по второй производной
    Использует численное дифференцирование для аппроксимации гессиана
    """
    n_samples, n_features = X.shape
    w = initialize_weights(n_features)
    
    loss_history = []
    
    for epoch in range(n_epochs):
        total_epoch_loss = 0
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Вычисляем градиент в текущей точке
            batch_gradient = np.zeros_like(w)
            batch_loss = 0
            
            for j in range(len(X_batch)):
                margin = calculate_margin(w, X_batch[j], y_batch[j])
                loss = quadratic_loss(margin)
                batch_loss += loss
                
                if margin < 1:
                    grad = quadratic_loss_gradient(w, X_batch[j], y_batch[j])
                    batch_gradient += grad
            
            if len(X_batch) > 0:
                batch_gradient /= len(X_batch)
                batch_loss /= len(X_batch)
                
                # Вычисляем градиент в точке w + epsilon * direction для аппроксимации второй производной
                direction = -batch_gradient  # антиградиентное направление
                w_plus = w + epsilon * direction

                grad_w_plus = np.zeros_like(w)
                for j in range(len(X_batch)):
                    margin_plus = calculate_margin(w_plus, X_batch[j], y_batch[j])
                    if margin_plus < 1:
                        grad_plus = quadratic_loss_gradient(w_plus, X_batch[j], y_batch[j])
                        grad_w_plus += grad_plus

                if len(X_batch) > 0:
                    grad_w_plus /= len(X_batch)

                # Аппроксимация второй производной вдоль направления
                f_prime = np.dot(batch_gradient, direction)
                f_double_prime = (np.dot(grad_w_plus, direction) - f_prime) / epsilon

                # Вычисляем оптимальный шаг
                if abs(f_double_prime) > 1e-10:
                    optimal_step = -f_prime / f_double_prime
                    # Ограничиваем шаг разумными пределами
                    optimal_step = np.clip(optimal_step, 0.001, 10.0)
                else:
                    optimal_step = 0.01

                w = w + optimal_step * direction
            
            total_epoch_loss += batch_loss
        
        avg_epoch_loss = total_epoch_loss / (n_samples // batch_size)
        loss_history.append(avg_epoch_loss)
    
        if epoch % 20 == 0:
            print(f"Эпоха {epoch}, средняя потеря: {avg_epoch_loss:.4f}")

    return w, loss_history

def margin_based_sampling(X, y, learning_rate=0.01, n_epochs=100, batch_size=32, strategy='uncertainty'):
    """
    SGD с выбором объектов по стратегии отступов
    strategy:
    - 'uncertainty': чаще выбираем объекты с меньшей уверенностью (меньший |M|)
    - 'hard_only': выбираем только трудные объекты (M < 1)
    """
    n_samples, n_features = X.shape
    w = initialize_weights(n_features)
    
    loss_history = []
    
    for epoch in range(n_epochs):
        # Вычисляем отступы для всех объектов
        margins = calculate_all_margins(w, X, y)
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
                probabilities[mask] = np.maximum(0, 1 - margins[mask])  # положительная часть для ошибок
            else:
                probabilities = np.ones_like(margins) / len(margins)
        else:
            # равномерное распределение
            probabilities = np.ones_like(margins) / len(margins)

        # нормализуем вероятности
        probabilities = probabilities / np.sum(probabilities)

        # выбираем батч объектов по вероятностям
        total_epoch_loss = 0
        
        for _ in range(0, n_samples, batch_size):
            # выбираем batch_size объектов с учетом вероятностей
            batch_indices = np.random.choice(len(X), size=min(batch_size, len(X)), p=probabilities, replace=False)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            batch_loss = 0
            batch_gradient = np.zeros_like(w)
            
            for j in range(len(X_batch)):
                margin = calculate_margin(w, X_batch[j], y_batch[j])
                loss = quadratic_loss(margin)
                batch_loss += loss
                
                if margin < 1:
                    grad = quadratic_loss_gradient(w, X_batch[j], y_batch[j])
                    batch_gradient += grad
            
            if len(X_batch) > 0:
                batch_gradient /= len(X_batch)
                batch_loss /= len(X_batch)
                
                w = w - learning_rate * batch_gradient
            
            total_epoch_loss += batch_loss
        
        avg_epoch_loss = total_epoch_loss / (n_samples // batch_size)
        loss_history.append(avg_epoch_loss)
    
        if epoch % 20 == 0:
            print(f"Эпоха {epoch}, средняя потеря: {avg_epoch_loss:.4f}")

    return w, loss_history

def initialize_weights_correlation(X, y):
    """
    Инициализация весов на основе корреляции с целевой переменной
    """
    n_features = X.shape[1]
    
    # Вычисляем корреляцию каждого признака с целевой переменной
    correlations = []
    for i in range(n_features):
        if np.std(X[:, i]) > 0:  # избегаем деления на 0
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        else:
            correlations.append(0)
    
    correlations = np.array(correlations)
    
    # Нормализуем и масштабируем
    weights = correlations * 0.1  # масштабируем для маленьких значений
    return weights

def stochastic_gradient_descent_logistic(
    X,
    y,
    w,
    learning_rate=0.01,
    n_epochs=100,
    batch_size=32,
    plot=True,
    *,
    track_full_losses=False,
    X_train_full=None,
    y_train_full=None,
    X_val_full=None,
    y_val_full=None,
):
    """
    SGD с логистической функцией потерь
    """
    
    def _mean_logistic_loss_full(w, X, y):
        """Средняя логистическая потеря на всей выборке"""
        margins = y * (X @ w)
        return float(np.mean([logistic_loss(m) for m in margins]))
    
    n_samples = X.shape[0]
    loss_history = []
    train_full_loss_history = []
    val_full_loss_history = []
    
    for epoch in range(n_epochs):
        total_epoch_loss = 0
        num_batches = 0
        
        # Случайное перемешивание данных каждую эпоху
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, batch_size):
            # Берем мини-батч
            batch_indices = indices[i:i + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            batch_loss = 0
            batch_gradient = np.zeros_like(w)
            
            # Вычисляем градиент для батча (логистическая потеря)
            for j in range(len(X_batch)):
                margin = calculate_margin(w, X_batch[j], y_batch[j])
                loss = logistic_loss(margin)
                batch_loss += loss
                
                # Градиент логистической потери
                grad = logistic_loss_gradient(w, X_batch[j], y_batch[j])
                batch_gradient += grad
            
            # Усредняем градиент по батчу
            if len(X_batch) > 0:
                batch_gradient /= len(X_batch)
                batch_loss /= len(X_batch)
                
                # Обновляем веса
                w = w - learning_rate * batch_gradient
            
            total_epoch_loss += batch_loss
            num_batches += 1
        
        # Средняя потеря за эпоху
        avg_epoch_loss = total_epoch_loss / max(1, num_batches)
        loss_history.append(avg_epoch_loss)

        if track_full_losses:
            if X_train_full is None or y_train_full is None or X_val_full is None or y_val_full is None:
                raise ValueError("track_full_losses=True требует X_train_full/y_train_full и X_val_full/y_val_full")
            train_full_loss_history.append(_mean_logistic_loss_full(w, X_train_full, y_train_full))
            val_full_loss_history.append(_mean_logistic_loss_full(w, X_val_full, y_val_full))
        
        if epoch % 20 == 0:
            print(f'Эпоха {epoch}, средняя потеря (logistic): {avg_epoch_loss:.4f}')
    
    if plot:
        sgd_plot(loss_history, ' (логистическая потеря)')
    
    if track_full_losses:
        return w, loss_history, train_full_loss_history, val_full_loss_history
    return w, loss_history


def sgd_plot(loss_history, title):
    # Визуализируем процесс обучения
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Эпоха')
    plt.ylabel('Средняя потеря')
    plt.title(f'Сходимость стохастического градиентного спуска{title}')
    plt.grid(True, alpha=0.3)
    plt.close()
