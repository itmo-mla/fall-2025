from model.model import BinaryLinearClassification
from module.margin import BinaryClassificationMargin
from data_load.data_load import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == '__main__':
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    my_model = BinaryLinearClassification(
        loss_function="hinge",
        optimizer="nag",
        evaluator="reccurent",
        margin=BinaryClassificationMargin(),
        epochs=100,
        batch_size=16,
        learning_rate=0.001,
        gamma=0.9,
        coef=0.9
    )

    my_model = my_model.fit(X_train, y_train)

    my_model.plot_loss()
    print("Accuracy", my_model.score(X_test, y_test))