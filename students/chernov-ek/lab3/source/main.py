from utils import get_samples
from svm_model import CustomSVM


def main():
    # Загрузка выборок
    X_train, X_test, y_train, y_test = get_samples()

    # Инициализация линейного классификатора с kernel trick
    model = CustomSVM(kernel_name="poly")

    # Визуализация решений и сохранение результатов
    pass


if __name__ == "__main__":
    main()
