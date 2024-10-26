from algorithm import Algorithm
import torch
import torch.nn as nn
import numpy as np
from train_test_evaluator import evaluate_split


class LinearInterpolationModule(nn.Module):
    def __init__(self, y_points, device):
        super(LinearInterpolationModule, self).__init__()
        self.device = device
        self.y_points = y_points.to(device)

    def forward(self, x_new_):
        x_new = x_new_.to(self.device)
        batch_size, num_points = self.y_points.shape
        x_points = torch.linspace(0, 1, num_points).to(self.device).expand(batch_size, -1).contiguous()
        x_new_expanded = x_new.unsqueeze(0).expand(batch_size, -1).contiguous()
        idxs = torch.searchsorted(x_points, x_new_expanded, right=True)
        idxs = idxs - 1
        idxs = idxs.clamp(min=0, max=num_points - 2)
        x1 = torch.gather(x_points, 1, idxs)
        x2 = torch.gather(x_points, 1, idxs + 1)
        y1 = torch.gather(self.y_points, 1, idxs)
        y2 = torch.gather(self.y_points, 1, idxs + 1)
        weights = (x_new_expanded - x1) / (x2 - x1)
        y_interpolated = y1 + weights * (y2 - y1)
        return y_interpolated


class ANN(nn.Module):
    def __init__(self, target_size, class_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = target_size
        self.class_size = class_size

        init_vals = torch.linspace(0.001, 0.99, self.target_size + 2)
        self.indices = nn.Parameter(
            torch.tensor([ANN.inverse_sigmoid_torch(init_vals[i + 1]) for i in range(self.target_size)],
                         requires_grad=True).to(self.device))
        self.linear = nn.Sequential(
            nn.Linear(self.target_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.class_size)
        )
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    @staticmethod
    def inverse_sigmoid_torch(x):
        return -torch.log(1.0 / x - 1.0)

    def forward(self, linterp):
        outputs = linterp(self.get_indices())
        soc_hat = self.linear(outputs)
        if self.class_size == 1:
            soc_hat = soc_hat.reshape(-1)
        return soc_hat

    def get_indices(self):
        return torch.sigmoid(self.indices)


class Algorithm_bsdr(Algorithm):
    def __init__(self, target_size, dataset, tag, reporter, verbose, test):
        super().__init__(target_size, dataset, tag, reporter, verbose, test)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        self.verbose = verbose
        self.target_size = target_size
        if dataset.is_classification():
            self.criterion = torch.nn.CrossEntropyLoss()
            self.class_size = len(np.unique(self.dataset.get_train_y()))
        else:
            self.criterion = torch.nn.MSELoss()
            self.class_size = 1
        self.lr = 0.001
        self.ann = ANN(self.target_size, self.class_size)
        self.ann.to(self.device)
        self.original_feature_size = self.dataset.get_train_x().shape[1]
        self.total_epoch = 500
        self.X_train = torch.tensor(self.dataset.get_train_x(), dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(self.dataset.get_train_y(), dtype=torch.int32).to(self.device)

    def get_selected_indices(self):
        self.ann.train()
        self.write_columns()
        optimizer = torch.optim.Adam(self.ann.parameters(), lr=self.lr, weight_decay=self.lr/10)
        linterp = LinearInterpolationModule(self.X_train, self.device)
        if self.dataset.is_classification():
            y = self.y_train.type(torch.LongTensor).to(self.device)
        else:
            y = self.y_train
        for epoch in range(self.total_epoch):
            optimizer.zero_grad()
            y_hat = self.ann(linterp)
            if not self.dataset.is_classification():
                y_hat = y_hat.reshape(-1)
            loss = self.criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            self.report(epoch, loss.item())
        self.set_selected_indices(self.get_indices())
        self.set_weights([1]*self.target_size)
        return self, self.get_indices()

    def write_columns(self):
        if not self.verbose:
            return
        selected_bands = self.get_indices()
        columns = ["epoch","loss","oa","aa","k"] + [f"band_{index+1}" for index in range(len(selected_bands))]
        print("".join([str(i).ljust(20) for i in columns]))

    def report(self, epoch, loss):
        if not self.verbose:
            return
        if epoch%10 != 0:
            return

        oa, aa, k = evaluate_split(*self.dataset.get_a_fold(), self)
        bands = self.get_indices()
        self.reporter.report_epoch_bsdr(epoch, loss, oa, aa, k, bands)
        cells = [epoch, loss, oa, aa, k] + bands
        cells = [round(item, 5) if isinstance(item, float) else item for item in cells]
        print("".join([str(i).ljust(20) for i in cells]))

    def get_indices(self):
        indices = torch.round(self.ann.get_indices() * self.original_feature_size ).to(torch.int64).tolist()
        return list(dict.fromkeys(indices))

    def transform(self, X):
        return X[:,self.get_indices()]

    def is_cacheable(self):
        return False