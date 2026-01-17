import kagglehub

from data.dataset import Dataset


class ObesityDataset(Dataset):
    def load(self):
        self.df = kagglehub.dataset_load(
            kagglehub.KaggleDatasetAdapter.PANDAS,
            'ruchikakumbhar/obesity-prediction',
            'Obesity prediction.csv'
        )

    def preprocess(self):
        self.df = self.df.dropna().reset_index(drop=True)
        self.df['target'] = -1
        self.df.loc[self.df['Obesity'].str.startswith('Obesity'), 'target'] = 1
        self.df.drop('Obesity', axis=1, inplace=True)

        self.df['Gender'] = self.df['Gender'].map({"Female": 0, "Male": 1})

        yes_no_cols = ["family_history", "FAVC", "SMOKE", "SCC"]
        for col in yes_no_cols:
            self.df[col] = self.df[col].map({"no": 0, "yes": 1})

        stages_encode = {'no': 0.0, 'Sometimes': 0.5, 'Frequently': 0.8, 'Always': 1.0}
        for col in ['CAEC', 'CALC']:
            self.df[col] = self.df[col].map(stages_encode)

        top_value = self.df['MTRANS'].mode().values[0]
        for val in self.df['MTRANS'].unique():
            if val != top_value:
                self.df[f'MTRANS_{val}'] = (self.df['MTRANS'] == val).astype(int)
        self.df.drop('MTRANS', axis=1, inplace=True)

