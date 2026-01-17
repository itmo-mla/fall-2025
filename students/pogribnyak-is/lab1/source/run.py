import time
import yaml
from pathlib import Path
from typing import Dict
import numpy as np
from sklearn.linear_model import SGDClassifier

from core.loss import HingeLoss
from data.datasets import DATASETS
from core.model import LinearClassifier
from core.regularizer import L2Regularizer
from core.optimizer import SGDWithMomentum, SteepestDescent
from core.initializer import CorrelationInitializer, RandomInitializer
from utils.metrics import compute_metrics, print_metrics
from utils.visualization import (
    plot_training_history,
    plot_margins_analysis,
    plot_comparison_histogram,
    plot_weights_visualization
)


def load_training_config(config_path: str = "config.yml"):
    with open(config_path, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
    return type('Config', (), config['training'])()


def choose_dataset(datasets: list) -> str:
    print("Доступные датасеты:")
    for i, name in enumerate(datasets, 1): print(f"  {i}. {name}")
    
    while True:
        try:
            choice = input("Выберите датасет (номер или название): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(datasets): return datasets[idx]
            elif choice in datasets: return choice
            print(f"Неверный выбор. Пожалуйста, введите число от 1 до {len(datasets)} или название датасета.")
        except (ValueError, KeyboardInterrupt):
            print("Прервано пользователем.")
            exit(1)


def run_training(
    name: str,
    X_train, y_train, X_val, y_val,
    config,
    experiment_dir: Path,
    *,
    initializer,
    optimizer,
    batch_size=1,
    shuffle=False,
    order_by_margin=False,
    verbose=True,
    seed=None
):
    print(name)

    model = LinearClassifier(
        loss=HingeLoss(),
        regularizer=L2Regularizer(lambda_reg=config.l2),
        optimizer=optimizer,
        initializer=initializer
    )

    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=config.epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        order_by_margin=order_by_margin,
        seed=config.seed if seed is None else seed,
        verbose=verbose,
    )

    base = name.lower().replace(" ", "_")

    plot_training_history(model.history, str(experiment_dir / f"{base}_history.png"))
    plot_margins_analysis(model, X_train, y_train, str(experiment_dir / f"{base}_margins.png"))

    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val, y_pred)
    print_metrics(metrics, name)

    return model, metrics


def train_with_correlation_init(X_train, y_train, X_val, y_val, config, experiment_dir):
    return run_training(
        "Correlation init",
        X_train, y_train, X_val, y_val, config, experiment_dir,
        initializer=CorrelationInitializer(),
        optimizer=SGDWithMomentum(config.lr, config.momentum),
        batch_size=1,
        shuffle=False,
        verbose=True
    )


def train_with_random_init_multistart(X_train, y_train, X_val, y_val, config, experiment_dir):
    best_model, best_metrics, best_acc = None, None, -1

    for i in range(config.n_starts):
        print(f"Запуск {i+1}/{config.n_starts}")
        model, metrics = run_training(
            f"Random init run {i+1}",
            X_train, y_train, X_val, y_val, config, experiment_dir,
            initializer=RandomInitializer(scale=0.01),
            optimizer=SGDWithMomentum(config.lr, config.momentum),
            batch_size=32,
            shuffle=False,
            verbose=False,
            seed=config.seed + i,
        )
        acc = metrics["accuracy"]
        print(f"Валидационная точность: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model, best_metrics = model, metrics

    print(f"Лучшая валидационная точность: {best_acc:.4f}")
    return best_model, best_metrics


def train_with_random_shuffle(X_train, y_train, X_val, y_val, config, experiment_dir):
    return run_training(
        "Random shuffle",
        X_train, y_train, X_val, y_val, config, experiment_dir,
        initializer=RandomInitializer(scale=0.1),
        optimizer=SGDWithMomentum(config.lr, config.momentum),
        batch_size=32,
        shuffle=True,
        verbose=True
    )


def train_with_margin_order(X_train, y_train, X_val, y_val, config, experiment_dir):
    return run_training(
        "Margin order",
        X_train, y_train, X_val, y_val, config, experiment_dir,
        initializer=RandomInitializer(scale=0.05),
        optimizer=SGDWithMomentum(config.lr, config.momentum),
        batch_size=32,
        shuffle=False,
        order_by_margin=True,
        seed=config.seed,
        verbose=True
    )


def train_with_steepest_descent(X_train, y_train, X_val, y_val, config, experiment_dir):
    return run_training(
        "Steepest Descent",
        X_train, y_train, X_val, y_val, config, experiment_dir,
        initializer=RandomInitializer(scale=0.05),
        optimizer=SteepestDescent(alpha_init=config.lr, beta=0.5, c=1e-4, max_iters=20),
        batch_size=32,
        shuffle=True,
        seed=config.seed,
        verbose=True
    )


def train_sgd_classifier(
    X_train, y_train, X_val, y_val, config
) -> tuple[SGDClassifier, Dict]:
    print("Обучение эталонного SGDClassifier")

    y_train_binary = (y_train == 1).astype(int)

    sgd = SGDClassifier(
        loss="squared_error",
        alpha=config.l2,
        learning_rate='constant',
        eta0=config.lr,
        max_iter=config.epochs,
        random_state=config.seed,
        verbose=0
    )

    sgd.fit(X_train, y_train_binary)

    y_pred_binary = sgd.predict(X_val)
    y_pred = (y_pred_binary * 2 - 1).astype(np.float32)  # Обратно в -1/1

    metrics = compute_metrics(y_val, y_pred)
    print_metrics(metrics, "SGDClassifier")

    return sgd, metrics


def main():
    config = load_training_config()
    choice = choose_dataset(list(DATASETS.keys()))
    experiment_dir = Path(config.save_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}_{choice}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Загрузка датасета: {choice}")
    data = DATASETS[choice](seed=config.seed)
    X_train, y_train, X_val, y_val, X_test, y_test = data.split_and_scale()
    
    print(f"Размеры данных:")
    print(f"Обучающая выборка: {X_train.shape}")
    print(f"Валидационная выборка: {X_val.shape}")
    print(f"Тестовая выборка: {X_test.shape}")
    results = {}
    models = {}
    model_corr, metrics_corr = train_with_correlation_init(
        X_train, y_train, X_val, y_val, config, experiment_dir
    )
    results["Correlation Init"] = metrics_corr['accuracy']
    models["Correlation Init"] = model_corr

    model_rand, metrics_rand = train_with_random_init_multistart(
        X_train, y_train, X_val, y_val, config, experiment_dir
    )
    results["Random Init (Multistart)"] = metrics_rand['accuracy']
    models["Random Init (Multistart)"] = model_rand

    model_shuffle, metrics_shuffle = train_with_random_shuffle(
        X_train, y_train, X_val, y_val, config, experiment_dir
    )
    results["Random Shuffle"] = metrics_shuffle['accuracy']
    models["Random Shuffle"] = model_shuffle

    model_margin, metrics_margin = train_with_margin_order(
        X_train, y_train, X_val, y_val, config, experiment_dir
    )
    results["Margin Order"] = metrics_margin['accuracy']
    models["Margin Order"] = model_margin

    model_steepest, metrics_steepest = train_with_steepest_descent(
        X_train, y_train, X_val, y_val, config, experiment_dir
    )
    results["Steepest Descent"] = metrics_steepest['accuracy']
    models["Steepest Descent"] = model_steepest

    sgd_model, metrics_sgd = train_sgd_classifier(
        X_train, y_train, X_val, y_val, config
    )
    results["SGDClassifier"] = metrics_sgd['accuracy']
    models["SGDClassifier"] = sgd_model
    plot_comparison_histogram(results, str(experiment_dir / "comparison_histogram.png"))
    best_method = max(results, key=results.get)
    best_model = models[best_method]
    
    print(f"Лучший метод: {best_method} с точностью на валидации: {results[best_method]:.4f}")
    
    if isinstance(best_model, LinearClassifier):
        y_test_pred = best_model.predict(X_test)
        test_metrics = compute_metrics(y_test, y_test_pred)
        print_metrics(test_metrics, f"{best_method} (Test Set)")
        plot_weights_visualization(best_model, save_path=str(experiment_dir / "best_model_weights.png"))
    else:
        y_test_pred_binary = best_model.predict(X_test)
        y_test_pred = (y_test_pred_binary * 2 - 1).astype(np.float32)
        test_metrics = compute_metrics(y_test, y_test_pred)
        print_metrics(test_metrics, f"{best_method} (Test Set)")


if __name__ == "__main__":
    main()
