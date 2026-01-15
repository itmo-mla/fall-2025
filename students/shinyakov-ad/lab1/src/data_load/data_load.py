import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "The_Cancer_data_1500_V2.csv"

def load_dataset():
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "rabieelkharoua/cancer-prediction-dataset",
        file_path)
    
    X = (df.drop(columns=['Diagnosis'])).to_numpy()
    y = df['Diagnosis'].replace({0: -1, 1: 1}).to_numpy()
    
    return X, y