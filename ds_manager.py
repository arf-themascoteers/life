import pandas as pd
from sklearn.model_selection import train_test_split


class DSManager:
    def __init__(self, name):
        self.name = name
        dataset_path = f"data/{name}.csv"
        df = pd.read_csv(dataset_path)
        df = df.sample(frac=1).reset_index(drop=True)
        self.data = df.to_numpy()

    def get_name(self):
        return self.name

    def get_k_folds(self):
        folds = 20
        for i in range(folds):
            seed = 40 + i
            yield train_test_split(self.data[:,0:-1], self.data[:,-1], test_size=0.95, random_state=seed, stratify=self.data[:, -1])

    def get_bs_train_data(self):
        return self.data

    def __repr__(self):
        return self.get_name()


