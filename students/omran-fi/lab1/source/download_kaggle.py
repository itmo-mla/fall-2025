from pathlib import Path
import zipfile


DATASET_ID = "sonialikhan/heart-attack-analysis-and-prediction-dataset"


def download_heart_dataset(data_dir: str = "data") -> str:
    """
    Downloads Kaggle dataset into data_dir and returns path to heart.csv.

    Requires Kaggle API configured with kaggle.json in ~/.kaggle/kaggle.json.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    csv_path = data_path / "heart.csv"
    if csv_path.exists():
        return str(csv_path)

    api = KaggleApi()
    api.authenticate()

    # Download as zip (Kaggle API always downloads zip for dataset files)
    api.dataset_download_files(DATASET_ID, path=str(data_path), unzip=False)

    # Find zip
    zips = list(data_path.glob("*.zip"))
    if not zips:
        raise FileNotFoundError("No .zip file downloaded. Check Kaggle API setup/auth.")
    zip_file = zips[0]

    # Unzip
    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(str(data_path))

    # Remove zip (optional)
    try:
        zip_file.unlink()
    except OSError:
        pass

    # Prefer heart.csv
    if csv_path.exists():
        return str(csv_path)

    # Otherwise, pick first csv found
    csvs = list(data_path.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError("No CSV files found after unzip.")
    return str(csvs[0])
