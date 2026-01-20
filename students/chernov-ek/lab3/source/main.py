from pathlib import Path

from sklearn.metrics import classification_report
from sklearn.svm import SVC

from utils import get_samples, vis_solutions
from svm_model import CustomSVM


def main():
    # Загрузка выборок
    X_train, X_test, y_train, y_test = get_samples()

    # Инициализация линейного классификатора с kernel trick
    custom_model = CustomSVM(kernel_name="poly", C=1, degree=2)
    custom_model.fit(X_train, y_train)
    y_pred = custom_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Инициализация эталонного классификатора
    model_sk = SVC(kernel="poly", C=1, degree=2)
    model_sk.fit(X_train, y_train)
    y_pred_sk = model_sk.predict(X_test)
    print(classification_report(y_test, y_pred_sk))

    # Визуализация решений и сохранение результатов
    save_path = Path("../assets")
    models = [("CustomSVM", custom_model, y_pred), ("Sklearn SVM", model_sk, y_pred_sk)]
    vis_solutions(models, X_train, X_test, y_train, y_test, save_path)


if __name__ == "__main__":
    main()
