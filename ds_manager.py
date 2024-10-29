import pandas as pd
from sklearn.model_selection import train_test_split


class DSManager:
    def __init__(self, name, test=False,split="bsnet"):
        self.name = name
        self.test = test
        self.split = split
        dataset_path = f"data/{name}.csv"
        df = pd.read_csv(dataset_path)
        frac = 1
        if self.test:
            frac = 1
        df = df.sample(frac=frac).reset_index(drop=True)
        self.data = df.to_numpy()
        if self.split=="bsnet":
            self.bs_train_data = self.data
            self.svm_evaluate_data = self.data
        else:
            if self.is_classification():
                self.bs_train_data, self.svm_evaluate_data = train_test_split(self.data, test_size=0.90, stratify=self.data[:, -1])
            else:
                self.bs_train_data, self.svm_evaluate_data = train_test_split(self.data, test_size=0.90)

    def is_classification(self):
        return DSManager.is_dataset_classification(self.name)

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
        if self.is_classification():
            return train_test_split(self.svm_evaluate_data[:,0:-1], self.svm_evaluate_data[:,-1], test_size=0.95, random_state=seed, stratify=self.svm_evaluate_data[:, -1])
        return train_test_split(self.svm_evaluate_data[:, 0:-1], self.svm_evaluate_data[:, -1], test_size=0.95, random_state=seed)

    def get_bs_train_x_y(self):
        return self.get_bs_train_x(), self.get_bs_train_y()

    def get_bs_train_x(self):
        return self.bs_train_data[:,0:-1]

    def get_bs_train_y(self):
        return self.bs_train_data[:, -1]

    def __repr__(self):
        return self.get_name()

    @staticmethod
    def is_dataset_classification(name):
        if name == "lucas_r":
            return False
        return True


