import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler


def _train_test_split(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not (0 < test_size < 1):
        raise ValueError("test_size должен быть в интервале (0, 1)")
    
    n = len(x)
    if len(y) != n:
        raise ValueError("x и y должны иметь одинаковую длину")
    if n == 0:
        raise ValueError("Нельзя делить пустой датасет")
    
    rng = np.random.default_rng(random_state)
    indices = np.arange(n)
    rng.shuffle(indices)
    
    n_test = int(round(n * test_size))
    n_test = max(1, min(n_test, n - 1))
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    x_test = x.iloc[test_idx]
    y_test = y.iloc[test_idx]
    x_train = x.iloc[train_idx]
    y_train = y.iloc[train_idx]
    
    return x_train, x_test, y_train, y_test


def load_and_preprocess_data():
    print("Загрузка датасета Diabetes из sklearn...")
    
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data
    y = diabetes.target
    
    print(f"Загружено {len(X)} объектов")
    print(f"Количество признаков: {X.shape[1]}")
    print(f"\nОписание признаков:")
    print(f"  - age: возраст")
    print(f"  - sex: пол")
    print(f"  - bmi: индекс массы тела")
    print(f"  - bp: среднее артериальное давление")
    print(f"  - s1-s6: шесть измерений сыворотки крови")
    print(f"\nЦелевая переменная: прогрессирование диабета через год после baseline")
    print(f"  Диапазон: [{y.min():.1f}, {y.max():.1f}], среднее: {y.mean():.1f}")
    
    X_train, X_test, y_train, y_test = _train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )
    
    print(f"\nОбучающая выборка: {len(X_train)} объектов")
    print(f"Тестовая выборка: {len(X_test)} объектов")
    
    print("Стандартизация признаков...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Финальная размерность признаков: {X_train_scaled.shape[1]}")
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f"\nРазмеры данных:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
