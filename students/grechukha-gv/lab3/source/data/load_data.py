import io
import zipfile

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _train_val_test_split(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    val_size: float,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    if not (0 < test_size < 1):
        raise ValueError("test_size должен быть в интервале (0, 1)")
    if not (0 < val_size < 1):
        raise ValueError("val_size должен быть в интервале (0, 1)")
    if test_size + val_size >= 1:
        raise ValueError("Сумма test_size и val_size должна быть меньше 1")

    n = len(x)
    if len(y) != n:
        raise ValueError("x и y должны иметь одинаковую длину")
    if n == 0:
        raise ValueError("Нельзя делить пустой датасет")

    rng = np.random.default_rng(random_state)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_test = max(1, min(n_test, n - 2))
    n_val = max(1, min(n_val, n - n_test - 1))

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    x_test = x.iloc[test_idx]
    y_test = y.iloc[test_idx]
    x_val = x.iloc[val_idx]
    y_val = y.iloc[val_idx]
    x_train = x.iloc[train_idx]
    y_train = y.iloc[train_idx]

    return x_train, x_val, x_test, y_train, y_val, y_test


def load_and_preprocess_data():
    print("Загрузка датасета Bank Marketing...")
    url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
    response = requests.get(url)

    with zipfile.ZipFile(io.BytesIO(response.content)) as outer_zip:
        with outer_zip.open('bank-additional.zip') as inner_zip_file:
            inner_zip_data = inner_zip_file.read()
            with zipfile.ZipFile(io.BytesIO(inner_zip_data)) as inner_zip:
                with inner_zip.open('bank-additional/bank-additional-full.csv') as f:
                    bank_marketing = pd.read_csv(f, sep=';')

    print(f"Загружено {len(bank_marketing)} объектов")

    # Преобразуем целевую переменную в {-1, +1}
    bank_marketing['y'] = bank_marketing['y'].map({'yes': 1, 'no': -1}).astype(int)
    bank_marketing = bank_marketing.drop('duration', axis=1)

    X = bank_marketing.drop('y', axis=1)
    y = bank_marketing['y']

    # Делим 70% train, 15% val, 15% test
    X_train, X_val, X_test, y_train, y_val, y_test = _train_val_test_split(
        X,
        y,
        test_size=0.15,
        val_size=0.15,
        random_state=42,
    )

    numerical_columns = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

    # Преобразуем категориальные признаки
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
    X_val_encoded = encoder.transform(X_val[categorical_columns])
    X_test_encoded = encoder.transform(X_test[categorical_columns])

    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_columns, index=X_train.index)
    X_val_encoded_df = pd.DataFrame(X_val_encoded, columns=encoded_columns, index=X_val.index)
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)

    X_train_final = pd.concat([X_train[numerical_columns], X_train_encoded_df], axis=1)
    X_val_final = pd.concat([X_val[numerical_columns], X_val_encoded_df], axis=1)
    X_test_final = pd.concat([X_test[numerical_columns], X_test_encoded_df], axis=1)

    # Нормализуем численные признаки
    scaler = StandardScaler()
    X_train_final[numerical_columns] = scaler.fit_transform(X_train_final[numerical_columns])
    X_val_final[numerical_columns] = scaler.transform(X_val_final[numerical_columns])
    X_test_final[numerical_columns] = scaler.transform(X_test_final[numerical_columns])

    print(f"Обучающая выборка: {len(X_train_final)} объектов, {X_train_final.shape[1]} признаков")
    print(f"Валидационная выборка: {len(X_val_final)} объектов")
    print(f"Тестовая выборка: {len(X_test_final)} объектов")
    
    return X_train_final.values, X_val_final.values, X_test_final.values, y_train.values, y_val.values, y_test.values
