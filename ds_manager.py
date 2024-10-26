import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import numpy as np


class DSManager:
    def __init__(self, name, test=False):
        self.name = name
        self.test = test
        dataset_path = f"data/{name}.csv"
        df = pd.read_csv(dataset_path)
        frac = 1
        if self.test:
            frac = 1
        df = df.sample(frac=frac).reset_index(drop=True)
        self.data = df.to_numpy()
        if name == "lucas_texture_4_r":
            X = self.data[:, :-1]
            y = self.data[:, -1]
            rus = RandomUnderSampler(sampling_strategy={0: 173, 1: 173, 2: 173, 3: 173})
            X_resampled, y_resampled = rus.fit_resample(X, y)
            self.data = np.column_stack((X_resampled, y_resampled))
            print(len(self.data))
        if name == "lucas_lc0_s_r":
            X = self.data[:, :-1]
            y = self.data[:, -1]
            rus = RandomUnderSampler(sampling_strategy={0: 599, 1: 599, 2: 599, 3: 599, 4:599})
            X_resampled, y_resampled = rus.fit_resample(X, y)
            self.data = np.column_stack((X_resampled, y_resampled))
            print(len(self.data))



    def get_name(self):
        return self.name

    def get_k_folds(self):
        folds = 20
        if self.test:
            folds = 5
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


