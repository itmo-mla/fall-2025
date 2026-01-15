from typing import Iterator
import numpy as np

from source.data_loaders import ABCLoader


class ModuleMarginLoader(ABCLoader):
    """
    Loader для предъявления объектов на основе модуля отступа Mi.
    """
    def __init__(self, batch_size: int = 32, *, margins: np.ndarray = None, threshold_good_objects: float = None, threshold_outliners: float = None,
                 exclude_good_objects: bool = False, exclude_outliers: bool = False,
                 strategy: str = "error"):
        """
        :param batch_size: размер батча
        :param margins: отступы объектов (shape: (n_samples,))
        :param threshold_good_objects: порог для "хороших" объектов
        :param threshold_outliners: порог для "выбросов"
        :param exclude_good_objects: исключать ли объекты с M > threshold_good_objects
        :param exclude_outliers: исключать ли объекты с M < threshold_outliners
        :param strategy: "error" - чаще брать объекты с большой ошибкой (M меньше)
                         "uncertainty" - чаще брать объекты с маленькой уверенностью (|M| меньше)
        """
        super().__init__()

        self.batch_size = batch_size
        self.M = margins
        self.threshold_good_objects = threshold_good_objects
        self.threshold_outliners = threshold_outliners
        self.exclude_good_objects = exclude_good_objects
        self.exclude_outliers = exclude_outliers
        assert strategy in ["error", "uncertainty"], "strategy должен быть 'error' или 'uncertainty'"
        self.strategy = strategy

    def get_data(self, X: np.ndarray, y: np.ndarray) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """
        :param X: массив признаков, shape (n_samples, n_features)
        :param y: массив меток, shape (n_samples,)
        """
        n_samples = len(y)
        inds = np.arange(n_samples)

        # Исключаем хорошие объекты
        if self.exclude_good_objects and self.threshold_good_objects is not None:
            inds = inds[self.M[inds] <= self.threshold_good_objects]
        
        # Исключаем выбросы
        if self.exclude_outliers and self.threshold_outliners is not None:
            inds = inds[self.M[inds] >= self.threshold_outliners]

        if len(inds) == 0:
            raise ValueError("Нет объектов для выборки после применения фильтров.")
 
        # Вычисляем вероятности для выборки
        if self.strategy == "error":
            # чаще брать объекты с маленьким M (сдвигаем на минимум, чтобы избежать отрицательных probs)
            probs = 1 / (self.M[inds] - self.M[inds].min() + 1e-8)  # добавляем eps, чтобы избежать деления на ноль
        elif self.strategy == "uncertainty":
            # чаще брать объекты с маленькой уверенностью |M|
            probs = 1 / (np.abs(self.M[inds]) + 1e-8)

        probs = probs / probs.sum()  # нормируем вероятности

        n_batches = len(inds) // self.batch_size
        for n_batch in range(n_batches):
            batch_inds = np.random.choice(inds, size=self.batch_size, replace=False, p=probs)
            yield n_batch, X[batch_inds], y[batch_inds]
