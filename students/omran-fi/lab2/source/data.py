import os
import pandas as pd

KAGGLE_DATASET = "gkalpolukcu/knn-algorithm-dataset"
CSV_NAME = "KNNAlgorithmDataset.csv"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_dataset_to_data_dir(data_dir: str = "data") -> str:
    """
    Downloads dataset via kagglehub, then copies CSV into local ./data folder.
    Returns local path: data/KNNAlgorithmDataset.csv

    Requires kagglehub + Kaggle credentials.
    """
    import kagglehub

    ensure_dir(data_dir)
    local_csv_path = os.path.join(data_dir, CSV_NAME)

    if os.path.exists(local_csv_path):
        return local_csv_path

    downloaded_dir = kagglehub.dataset_download(KAGGLE_DATASET)
    src_csv_path = os.path.join(downloaded_dir, CSV_NAME)

    if not os.path.exists(src_csv_path):
        raise FileNotFoundError(f"CSV not found after download: {src_csv_path}")

    with open(src_csv_path, "rb") as fsrc, open(local_csv_path, "wb") as fdst:
        fdst.write(fsrc.read())

    return local_csv_path


def load_breast_cancer_df(data_dir: str = "data") -> pd.DataFrame:
    csv_path = download_dataset_to_data_dir(data_dir=data_dir)
    df = pd.read_csv(csv_path)

    # cleanup
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    return df
