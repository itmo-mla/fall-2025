from sklearn.metrics import classification_report, accuracy_score

from source.data_loaders.datasets import load_mushroom_dataset
from source.weights_initializers import correlation_init, multistart_init
from source.models import LinearClassificator
from source.losses import PerceptronLoss
from source.optimizers import GDOptimizer
from source.data_loaders import ShuffleLoader, ModuleMarginLoader
from source.tools import vis_graphics
from source.regularizers import L2Regularizer


def train_model_w_correlation_init():
    # Загружаем данные
    X_train, X_val, X_test, y_train, y_val, y_test = load_mushroom_dataset(labels=[1, -1], targets_shape=1)

    model = LinearClassificator(in_features=X_train.shape[1])

    # Инициализируем веса с помощью корреляции
    model.eval()
    layer = model.get_weights_layers()[0]
    W, b = layer.get_weights()
    W = correlation_init(X_val, y_val.ravel())
    layer.update_weights(W, b)

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

    # Evaluate
    # vis_graphics.visualize_losses_metrics(model.losses_train, model.losses_val, model.metrics_val)
    model.eval()
    y_pred = model(X_test)
    print(classification_report(y_test, y_pred))


def train_model_w_multistart_init():
    # Загружаем данные
    X_train, X_val, X_test, y_train, y_val, y_test = load_mushroom_dataset(labels=[1, -1], targets_shape=1)

    model = LinearClassificator(in_features=X_train.shape[1])
    loss = PerceptronLoss()
    # Инициализируем веса с помощью мультистарта
    model = multistart_init(
        model,
        X_val, y_val,
        loss,
        n_starts=10
    )

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

    # Evaluate
    # vis_graphics.visualize_losses_metrics(model.losses_train, model.losses_val, model.metrics_val)
    model.eval()
    y_pred = model(X_test)
    print(classification_report(y_test, y_pred))


def train_model_w_margins():
    # Загружаем данные
    X_train, X_val, X_test, y_train, y_val, y_test = load_mushroom_dataset(labels=[1, -1], targets_shape=1)

    model = LinearClassificator(in_features=X_train.shape[1])

    loss = PerceptronLoss()
    regularizer = None

    batch_size = 32
    exclude_good_objects = True
    exclude_outliers = False
    strategy = 'error'
    data_loader = ModuleMarginLoader(
        batch_size,
        exclude_good_objects=exclude_good_objects,
        exclude_outliers=exclude_outliers,
        strategy=strategy
    )
    lr = 0.001
    optimizer = GDOptimizer(model.get_weights_layers(), data_loader, lr)

    n_epochs = 90
    warmup_epochs = 10
    count_metric = accuracy_score
    verbose_n_batch_multiple = 10
    model.train_model(
        n_epochs,
        X_train, y_train, X_val, y_val,
        loss, optimizer, regularizer,
        count_metric=count_metric,
        warmup_epochs=warmup_epochs,
        verbose_n_batch_multiple=verbose_n_batch_multiple, verbose_statistic='EMA'
    )

    # Evaluate
    # vis_graphics.visualize_losses_metrics(model.losses_train, model.losses_val, model.metrics_val)
    model.eval()
    y_pred = model(X_test)
    print(classification_report(y_test, y_pred))


def train_model_best():
    # Загружаем данные
    X_train, X_val, X_test, y_train, y_val, y_test = load_mushroom_dataset(labels=[1, -1], targets_shape=1)

    model = LinearClassificator(in_features=X_train.shape[1])
    loss = PerceptronLoss()

    regularizer = L2Regularizer(lambda_q=0.4)
    lr = 0.001
    batch_size = 8
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
        verbose_n_batch_multiple=verbose_n_batch_multiple, verbose_statistic='EMI'
    )

    # Evaluate
    # vis_graphics.visualize_losses_metrics(model.losses_train, model.losses_val, model.metrics_val)
    model.eval()
    y_pred = model(X_test)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    train_model_w_correlation_init()
