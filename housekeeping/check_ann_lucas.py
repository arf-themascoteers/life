from algorithm import Algorithm
import torch
import torch.nn as nn
import numpy as np
from train_test_evaluator import evaluate_split
from ds_manager import DSManager


import torch

def r2_score_torch(y, y_pred):
    ss_tot = torch.sum((y - torch.mean(y)) ** 2)
    ss_res = torch.sum((y - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


class ANN(nn.Module):
    def __init__(self, target_size, class_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = target_size
        self.class_size = class_size
        self.linear = nn.Sequential(
            nn.Linear(self.target_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.class_size)
        )
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    def forward(self, X):
        outputs = self.linear(X)
        return outputs.reshape(-1)

    def get_indices(self):
        return torch.sigmoid(self.indices)


class Algorithm_bsdr():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        self.criterion = torch.nn.MSELoss()
        self.dataset = DSManager("lucas_r")
        self.lr = 0.0005
        self.ann = ANN(20, 1)
        self.ann.to(self.device)
        self.original_feature_size = self.dataset.get_bs_train_x().shape[1]
        self.total_epoch = 500
        idx = [i for i in range(100, 4100, 200)]
        X = self.dataset.get_bs_train_x()[:,idx]
        self.X_train = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(self.dataset.get_bs_train_y(), dtype=torch.float32).to(self.device)

    def get_selected_indices(self):
        self.ann.train()
        self.write_columns()
        optimizer = torch.optim.Adam(self.ann.parameters(), lr=self.lr, weight_decay=self.lr/10)
        for epoch in range(self.total_epoch):
            optimizer.zero_grad()
            y_hat = self.ann(self.X_train)
            loss = self.criterion(self.y_train, y_hat)
            loss.backward()
            optimizer.step()
            r2 = r2_score_torch(self.y_train, y_hat)
            self.report(epoch, loss.item(), r2)

    def write_columns(self):
        columns = ["epoch","loss","r2"]
        print("".join([str(i).ljust(20) for i in columns]))

    def report(self, epoch, loss, r2):
        cells = [epoch, loss, r2.item()]
        cells = [round(item, 5) if isinstance(item, float) else item for item in cells]
        print("".join([str(i).ljust(20) for i in cells]))



if __name__ == "__main__":
    alg = Algorithm_bsdr()
    alg.get_selected_indices()
