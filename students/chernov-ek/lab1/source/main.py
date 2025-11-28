from sklearn.metrics import accuracy_score

from source.data_loaders.datasets import load_mushroom_dataset
from source.weights_initializers import correlated_init
from source.models import LinearClassificator
from source.losses import PerceptronLoss
from source.optimizers import GDOptimizer
from source.data_loaders import ShuffleLoader, ModuleMarginLoader


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
    X_train, X_val, X_test, y_train, y_val, y_test = load_mushroom_dataset()

    model = LinearClassificator(in_features=X_train.shape[1])
    loss = PerceptronLoss()

    regularizer = None
    lr = 0.001
    batch_size = 32
    data_loader = ShuffleLoader(batch_size)
    optimizer = GDOptimizer(model.get_weights_layers(), data_loader, lr)

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
