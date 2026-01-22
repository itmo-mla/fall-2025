import time
import logging
from contextlib import contextmanager


def setup_logging(log_file='results/experiment_log.txt'):
    """
    Настраивает логирование экспериментов.
    
    Args:
        log_file: путь к файлу лога
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


@contextmanager
def timer(name="Operation", logger=None):
    """
    Контекстный менеджер для измерения времени выполнения.
    
    Usage:
        with timer("Training KNN"):
            knn.fit(X, y)
    
    Args:
        name: название операции
        logger: логгер для записи
    """
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    
    message = f"{name} завершено за {elapsed_time:.2f}s"
    
    if logger:
        logger.info(message)
    else:
        print(message)


def measure_inference_time(model, X, n_runs=3):
    """
    Измеряет среднее время предсказания модели.
    
    Args:
        model: обученная модель с методом predict
        X: данные для предсказания
        n_runs: количество запусков для усреднения
    
    Returns:
        dict: статистика времени
    """
    times = []
    
    for _ in range(n_runs):
        start = time.time()
        _ = model.predict(X)
        elapsed = time.time() - start
        times.append(elapsed)
    
    mean_time = sum(times) / len(times)
    time_per_sample = mean_time / len(X) * 1000  # в миллисекундах
    
    return {
        'mean_time': mean_time,
        'time_per_sample_ms': time_per_sample,
        'throughput': len(X) / mean_time  # объектов в секунду
    }
