import pandas as pd
from sklearn.model_selection import train_test_split


class DSManager:
    def __init__(self, name, test=False):
        self.name = name
        self.test = test
        dataset_path = f"data/{name}.csv"
        df = pd.read_csv(dataset_path)
        frac = 1
        if self.test:
            frac = 0.5
        df = df.sample(frac=frac).reset_index(drop=True)
        self.data = df.to_numpy()

    def get_name(self):
        return self.name

    def get_k_folds(self):
        folds = 20
        for i in range(folds):
            seed = 40 + i
            yield self.get_a_fold(seed)

    def get_a_fold(self, seed=50):
        return train_test_split(self.data[:,0:-1], self.data[:,-1], test_size=0.95, random_state=seed, stratify=self.data[:, -1])

    def get_bs_train_data(self):
        return self.data

    def get_train_x_y(self):
        return self.get_train_x(), self.get_train_y()

    def get_train_x(self):
        return self.data[:,0:-1]

    def get_train_y(self):
        return self.data[:, -1]

    def __repr__(self):
        return self.get_name()


