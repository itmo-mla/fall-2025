import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from kagglehub import KaggleDatasetAdapter

DATASET_PATH_STRING = "zeeshier/weather-forecast-dataset"
DATASET_FILENAME = "weather_forecast_data.csv"


def load_dataset_raw() -> pd.DataFrame:

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        DATASET_PATH_STRING,
        DATASET_FILENAME,
    )
    
    return df

def load_dataset(
    test_size: float = 0.3,
    random_state: int = 42,
):
    df = load_dataset_raw()
    df["Rain"] = df["Rain"].map({"no rain": 0, "rain": 1})

    X = df.drop(columns=["Rain"])
    y = df["Rain"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train.values, X_test.values, y_train.values, y_test.values, df