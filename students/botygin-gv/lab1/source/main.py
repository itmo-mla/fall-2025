import os
import pandas as pd
from sklearn.linear_model import SGDClassifier

from loader import DataLoader
from linear import LinearClassifier
from mertics import calculate_metrics
from visualization import Visualizer

LR = 2e-4
MOMENTUM = 0.7
ALPHA = 1e-3
LOSS_SMOOTHING = 1e-3
N_EPOCHS = 50
BATCH_SIZE = 64
VERBOSE = False

N_RESTARTS = 5

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


def evaluate_sklearn_baseline(X_tr, y_tr, X_te, y_te):
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=ALPHA,
        eta0=LR,
        learning_rate="constant",
        max_iter=100,
        random_state=42
    )
    clf.fit(X_tr, y_tr)
    metrics = calculate_metrics(y_te, clf.predict(X_te))
    return {
        "Method": "Sklearn SGD Classifier",
        "Initialization": "sklearn",
        "Optimizer": "sklearn",
        "Sampling": "sklearn",
        **metrics
    }


def run_model(config, X_train, y_train, X_test, y_test, seed=None):
    """Запускает один эксперимент с заданной конфигурацией и seed'ом."""
    model = LinearClassifier(
        init_method=config["init"],
        sampling_strategy=config["sampling"],
        optimizer_type=config["opt"],
        lr=LR,
        momentum=MOMENTUM,
        alpha=ALPHA,
        loss_smoothing=LOSS_SMOOTHING,
        random_seed=seed
    )

    train_loss, val_loss = model.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=VERBOSE
    )

    test_preds = model.predict(X_test)
    metrics = calculate_metrics(y_test, test_preds)

    return model, train_loss, val_loss, metrics


def run_multistart(config, X_train, y_train, X_test, y_test):
    """Запускает эксперимент с multistart для случайной инициализации."""
    best_acc = -1.0
    best_metrics = None
    best_model = None
    best_train_loss = []
    best_val_loss = []

    print(f"  Запуск multistart: {N_RESTARTS} перезапусков...")
    for r in range(N_RESTARTS):
        model, train_loss, val_loss, metrics = run_model(config, X_train, y_train, X_test, y_test, seed=r)
        if metrics['Accuracy'] > best_acc:
            best_acc = metrics['Accuracy']
            best_metrics = metrics
            best_model = model
            best_train_loss = train_loss
            best_val_loss = val_loss
            print(f"    Перезапуск {r+1}: Новый лучший результат: Accuracy = {best_acc:.4f}")

    return best_model, best_train_loss, best_val_loss, best_metrics


def run_single(config, X_train, y_train, X_test, y_test):
    """Запускает один стандартный эксперимент (correlation или single random)."""
    model, train_loss, val_loss, metrics = run_model(config, X_train, y_train, X_test, y_test)
    return model, train_loss, val_loss, metrics


def execute_experiment(config, X_train, y_train, X_test, y_test):
    print(f"\nЗапуск: {config['name']}")

    if config['init'] == 'random_multistart':
        model, train_loss, val_loss, metrics = run_multistart(config, X_train, y_train, X_test, y_test)
        print(
            f"Лучшие метрики {config['name']}: Accuracy = {metrics['Accuracy']:.4f}, "
            f"Precision = {metrics['Precision']:.4f}, Recall = {metrics['Recall']:.4f}, "
            f"F1 = {metrics['F1']:.4f} ")
    else:
        model, train_loss, val_loss, metrics = run_single(config, X_train, y_train, X_test, y_test)
        print(
            f"Метрики {config['name']}: Accuracy = {metrics['Accuracy']:.4f}, "
            f"Precision = {metrics['Precision']:.4f}, Recall = {metrics['Recall']:.4f}, "
            f"F1 = {metrics['F1']:.4f} ")

    viz = Visualizer(train_loss, val_loss)
    viz.plot_training_history(config["name"])
    viz.plot_margins(model.predict_proba(X_test), y_test, config["name"])

    return {"description": config["name"], **metrics}


def main():
    loader = DataLoader(dataset_name="breast_cancer")
    X_train, X_test, y_train, y_test = loader.load_data()

    experiments = []
    for init in ["random", "random_multistart", "correlation"]:
        for opt in ["momentum", "fast"]:
            for sampling in ["margin", "random"]:
                exp_name = f"{init} {opt} {sampling}"
                experiments.append({
                    "name": exp_name,
                    "init": init,
                    "opt": opt,
                    "sampling": sampling
                })

    results = []
    for exp in experiments:
        res = execute_experiment(exp, X_train, y_train, X_test, y_test)
        results.append({
            "Method": res["description"],
            "Accuracy": res["Accuracy"],
            "Precision": res["Precision"],
            "Recall": res["Recall"],
            "F1": res["F1"]
        })

    results.append(evaluate_sklearn_baseline(X_train, y_train, X_test, y_test))

    df = pd.DataFrame(results)
    df = df[["Method", "Accuracy", "Precision", "Recall", "F1"]].copy()
    df[["Accuracy", "Precision", "Recall", "F1"]] = df[["Accuracy", "Precision", "Recall", "F1"]].round(4)

    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ")
    print("=" * 80)
    print(df.to_markdown())


if __name__ == "__main__":
    main()