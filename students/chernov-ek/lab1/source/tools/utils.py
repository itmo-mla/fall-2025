import numpy as np


def get_margins(model: "LinearClassificator", X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray]:
    # Вычисляем отступы
    margins = model.get_weights_layers()[0](X)*y
    # Получаем индексы сортировки по margins
    idx = np.argsort(margins.squeeze())
    # Возвращаем индексы сортировки и отступы
    return idx, margins[idx].squeeze()
