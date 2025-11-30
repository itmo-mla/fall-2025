from model.model import BinaryLinearClassification
from module.margin import BinaryClassificationMargin
from data_load.data_load import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np

if __name__ == '__main__':
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 1. Корреляция
    model_corr = BinaryLinearClassification(
        loss_function="hinge",
        optimizer="sgd",
        evaluator="reccurent",
        margin=BinaryClassificationMargin(),
        epochs=100,
        batch_size=32,
        sampling_strategy="uniform",
        weight_init="correlation",
        learning_rate=0.002,
        gamma=0.9,
        coef=0.9
    )
    model_corr.fit(X_train, y_train)
    model_corr.plot_loss()
    print("Accuracy correlation", model_corr.score(X_test, y_test))
    model_corr.margin.visualize(model_corr.margin.calculate(model_corr.forward(np.c_[-np.ones((len(X), 1)), X], model_corr.weights), y))

    # Мультистарт
    model_multistart = BinaryLinearClassification(
        loss_function="hinge",
        optimizer="sgd",
        evaluator="reccurent",
        margin=BinaryClassificationMargin(),
        epochs=100,
        batch_size=32,
        sampling_strategy="uniform",
        weight_init="random",
        learning_rate=0.002,
        gamma=0.9,
        coef=0.9
    )
    model_multistart.fit(X_train, y_train, n_starts=5)
    model_multistart.plot_loss()
    print("Accuracy multistart SGD", model_multistart.score(X_test, y_test))
    model_multistart.margin.visualize(model_multistart.margin.calculate(model_multistart.forward(np.c_[-np.ones((len(X), 1)), X], model_multistart.weights), y))

    # Мультистарт + NAG
    model_multistart = BinaryLinearClassification(
        loss_function="hinge",
        optimizer="nag",
        evaluator="reccurent",
        margin=BinaryClassificationMargin(),
        epochs=100,
        batch_size=32,
        sampling_strategy="uniform",
        weight_init="random",
        learning_rate=0.002,
        gamma=0.9,
        coef=0.9
    )
    model_multistart.fit(X_train, y_train, n_starts=5)
    model_multistart.plot_loss()
    print("Accuracy multistart NAG", model_multistart.score(X_test, y_test))
    model_multistart.margin.visualize(model_multistart.margin.calculate(model_multistart.forward(np.c_[-np.ones((len(X), 1)), X], model_multistart.weights), y))

    # Предъявение по маржину
    model_hard_sampling = BinaryLinearClassification(
        loss_function="hinge",
        optimizer="sgd",
        evaluator="reccurent",
        margin=BinaryClassificationMargin(),
        epochs=100,
        batch_size=32,
        sampling_strategy="hard",
        weight_init="random",
        learning_rate=0.002,
        gamma=0.9,
        coef=0.9
    )
    model_hard_sampling.fit(X_train, y_train)
    model_hard_sampling.plot_loss()
    print("Accuracy margin sampling", model_hard_sampling.score(X_test, y_test))
    model_hard_sampling.margin.visualize(model_hard_sampling.margin.calculate(model_hard_sampling.forward(np.c_[-np.ones((len(X), 1)), X], model_hard_sampling.weights), y))

    # случайное предъявление
    model_uniform_sampling = BinaryLinearClassification(
        loss_function="hinge",
        optimizer="sgd",
        evaluator="reccurent",
        margin=BinaryClassificationMargin(),
        epochs=100,
        batch_size=32,
        sampling_strategy="uniform",
        weight_init="random",
        learning_rate=0.002,
        gamma=0.9,
        coef=0.9
    )
    model_uniform_sampling.fit(X_train, y_train)
    model_uniform_sampling.plot_loss()
    print("Accuracy uniform sampling", model_uniform_sampling.score(X_test, y_test))
    model_uniform_sampling.margin.visualize(model_uniform_sampling.margin.calculate(model_uniform_sampling.forward(np.c_[-np.ones((len(X), 1)), X], model_uniform_sampling.weights), y))

    # эталон
    baseline_model = SGDClassifier(
        loss='hinge',
        max_iter=100,
        learning_rate='constant',
        eta0=0.002,
        random_state=42,
        tol=1e-3,
        shuffle=True
    )

    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)

    print("Accuracy (sklearn SGDClassifier):", accuracy_score(y_test, y_pred_baseline))
