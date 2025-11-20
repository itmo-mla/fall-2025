import numpy as np

from source.models import LinearClassificator


def get_margins(model: LinearClassificator, X: np.ndarray, y: np.ndarray, is_sorted: bool = True) -> tuple[np.ndarray]:
    model.eval()
    # Вычисляем отступы
    margins = model.get_layers()[0](X)*y
    # Получаем индексы сортировки по margins
    idx = np.argsort(margins.squeeze())
    # Возвращаем индексы сортировки и отступы
    return idx, margins[idx].squeeze() if is_sorted else margins
