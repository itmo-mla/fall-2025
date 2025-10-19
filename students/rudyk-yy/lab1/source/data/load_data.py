import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "student_performance_updated_1000.csv"


def import_dataset():
    df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "haseebindata/student-performance-predictions",
  file_path)
    return df
