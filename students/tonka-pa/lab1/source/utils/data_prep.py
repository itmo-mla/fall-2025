'''
Dataset link: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success.zip
'''
import requests
import zipfile
import io
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------------------------------------------------------

SEED = 18092025

# -------------------------------------------------------------------------

def download_dataset(download_path: str) -> str:
    url = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"

    response = requests.get(url)
    response.raise_for_status()

    saveto_path = Path(download_path)
    saveto_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        for file in z.namelist():
            target_path = saveto_path / file
            if target_path.exists():
                print(f"Skipping existing file: {target_path}")
            else:
                z.extract(file, saveto_path)
    return target_path.as_posix()


def col_names_transform(col_name: str) -> str:
    res_name = col_name.strip().replace("\t", "").replace(' ', '_').lower()
    return res_name

def read_data(path_to_file: str, delim: str = ',') -> pd.DataFrame:
    dataset_path = Path(path_to_file)
    assert dataset_path.exists() 
    df = pd.read_csv(dataset_path, delimiter=delim) # '../datasets/data.csv'
    return df

def preprocess_data(df: pd.DataFrame, verbose: bool = False) -> tuple[tuple[list, list], tuple[list, list], list[str]]:
    df.columns = map(col_names_transform, df.columns.values)
    X, y = df.drop(columns=['target']), df['target']
    if verbose:
        print(f"X shape: {X.shape},\t y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, shuffle=True, stratify=y)
    y_train.shape, y_test.shape
    if verbose:
        print(f"y_train length: {y_train.shape[0]},\t y_test length: {y_test.shape[0]}")

    std_scaler = StandardScaler()
    X_train_scaled = std_scaler.fit_transform(X_train)
    X_test_scaled  = std_scaler.transform(X_test)
    if verbose:
        print("Sample of scaled data: \n", X_train_scaled[0, :5])

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train.values)
    y_test_enc  = label_encoder.transform(y_test.values)
    if verbose:
        print("Encoding compliance example\n", y_train[:5].values, y_train_enc[:5], label_encoder.classes_)

    return X_train_scaled, y_train_enc, X_test_scaled, y_test_enc, label_encoder.classes_


if __name__ == '__main__':
    pass