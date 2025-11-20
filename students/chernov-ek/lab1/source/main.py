import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from source.weights_initializers import correlated_init
from source.models import LinearClassificator
from source.losses import PerceptronLoss
from source.optimizers import GDOptimizer
from source.data_loaders import ShuffleLoader, ModuleMarginLoader


random_state = 42


def load_data(labels: list[int] = [1, -1]) -> tuple[np.ndarray]:
    """
    Функция возвращает тренировочный, валидационный и тестировочный набор данных.

    :param labels: Список с метками (тк бинарная классификация, то [1, 0] или [1, -1]).
    :type labels: list[int]

    :return: Кортеж с наборами данных (X_train, X_val, X_test, y_train, y_val, y_test).
    :rtype: tuple[np.ndarray]
    """
    # Загружаем данные
    mushroom = fetch_ucirepo(id=73)

    # Получаем признаки и метки
    X = mushroom.data.features
    y = mushroom.data.targets

    # Предобрабатываем данные
    # Инструменты для предобработки
    label_encoder = LabelEncoder()
    scaler = StandardScaler()
    # poisonous=1, edible=-1
    y = np.where(y == 'p', labels[0], labels[1])
    # Кодируем все признаки
    cols = X.keys()
    for col in cols:
        new_col = str(col) + "_n"
        X[new_col] = label_encoder.fit_transform(X[col])
        del X[col]

    # Разделяем на выборки
    X_train, X_test, y_train, y_test = map(np.array, train_test_split(X, y, test_size=0.2, random_state=random_state, shuffle=True, stratify=y))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state, shuffle=True, stratify=y_train)

    # Скелим данные
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    X_test = scaler.fit_transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Вычисление отступа объекта +

    # Вывод объектов по модулю отступа +

    # Градиент функции потерь +

    # Реккурентная оценка функционала качества +

    # SGD +

    # SGD с инерцией

    # L2 регуляризация +

    # Обучение линейного классификатора
    # инициализация весов через корреляцию
    # инициализация весов через мультистарт
    # обучить со случайным предъявлением и с предъявление объектов мо модулю отступа
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    model = LinearClassificator(in_features=X_train.shape[1])
    loss = PerceptronLoss()

    regularizer = None
    lr = 0.001
    batch_size = 32
    data_loader = ShuffleLoader(batch_size)
    optimizer = GDOptimizer(model.get_layers(), data_loader, lr)

    n_epochs = 100
    count_metric = accuracy_score
    verbose_n_batch_multiple = 10
    model.train_model(
        n_epochs,
        X_train, y_train, X_val, y_val,
        loss, optimizer, regularizer,
        count_metric=count_metric,
        verbose_n_batch_multiple=verbose_n_batch_multiple, verbose_statistic='EMA'
    )

    # Оценка качества классификатора +

    # Сравнение лучшей реализацией с эталонной +
