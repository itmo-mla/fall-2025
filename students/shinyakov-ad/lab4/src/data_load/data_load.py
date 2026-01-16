import pandas as pd
from sklearn.datasets import load_diabetes

DATASET_PATH_STRING = "akshaydattatraykhare/diabetes-dataset"
DATASET_FILENAME = "diabets.csv"

def load_raw_dataframe() -> pd.DataFrame:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    return X, y

