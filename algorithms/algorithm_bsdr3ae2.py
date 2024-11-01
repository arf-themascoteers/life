from algorithm import Algorithm
import torch
import torch.nn as nn
import numpy as np
from train_test_evaluator import evaluate_split
import matplotlib.pyplot as plt


def r2_score_torch(y, y_pred):
    ss_tot = torch.sum((y - torch.mean(y)) ** 2)
    ss_res = torch.sum((y - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return round(r2.item(),5)


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
    def __init__(self, original_size, target_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.original_size = original_size
        self.target_size = target_size

        init_vals = torch.linspace(0.001, 0.99, self.target_size + 2)
        self.indices = nn.Parameter(
            torch.tensor([ANN.inverse_sigmoid_torch(init_vals[i + 1]) for i in range(self.target_size)],
                         requires_grad=True).to(self.device))
        self.linear = nn.Sequential(
            nn.Linear(self.target_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.original_size),
            nn.BatchNorm1d(self.original_size),
            nn.Sigmoid()
        )
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    @staticmethod
    def inverse_sigmoid_torch(x):
        return -torch.log(1.0 / x - 1.0)

    def forward(self, linterp):
        outputs = linterp(self.get_indices())
        soc_hat = self.linear(outputs)
        return soc_hat

    def get_indices(self):
        return torch.sigmoid(self.indices)


class Algorithm_bsdr3ae2(Algorithm):
    def __init__(self, target_size, dataset, tag, reporter, verbose, test, props):
        super().__init__(target_size, dataset, tag, reporter, verbose, test, props)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        self.verbose = verbose
        self.target_size = target_size
        self.classification = dataset.is_classification()
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.lr = 0.001
        self.total_epoch = 500


        self.original_feature_size = self.dataset.get_bs_train_x().shape[1]
        self.ann = ANN(self.original_feature_size, self.target_size)
        self.ann.to(self.device)


        self.X_train = torch.tensor(self.dataset.get_bs_train_x(), dtype=torch.float32).to(self.device)

    def get_selected_indices(self):
        self.ann.train()
        self.write_columns()
        optimizer = torch.optim.Adam(self.ann.parameters(), lr=self.lr, weight_decay=self.lr/10)
        linterp = LinearInterpolationModule(self.X_train, self.device)

        for epoch in range(self.total_epoch):
            optimizer.zero_grad()
            X_hat = self.ann(linterp)

            sample_input = self.X_train[10].detach().cpu().numpy()
            sample_output = X_hat[10].detach().cpu().numpy()

            # if epoch % 10 == 0:
            #     fig, (ax1, ax2) = plt.subplots(1, 2)
            #     ax1.plot(sample_input)
            #     ax2.plot(sample_output)
            #     plt.savefig(f"saved_graphics/b{epoch}.png")
            #     plt.close()

            loss = self.criterion(X_hat, self.X_train)
            loss.backward()
            optimizer.step()
            r2_train = 0
            self.report(epoch, loss.item(), r2_train)
        self.set_selected_indices(self.get_indices())
        self.set_weights([1]*self.target_size)
        return self, self.get_indices()

    def write_columns(self):
        if not self.verbose:
            return
        m1 = "oa"
        m2 = "aa"
        m3 = "k"
        if not self.classification:
            m1 = "r2"
            m2 = "rmse"
            m3 = "r2_train"
        columns = ["epoch","loss",m1,m2,m3] + [f"band_{index+1}" for index in range(self.target_size)]
        print("".join([str(i).ljust(20) for i in columns]))

    def report(self, epoch, loss, r2_train):
        if not self.verbose:
            return
        if epoch%10 != 0:
            return


        m1,m2,m3 = evaluate_split(*self.dataset.get_a_fold(), self, classification=self.classification)
        if not self.classification:
            m3 = r2_train

        bands = self.get_indices()
        self.reporter.report_epoch_bsdr(epoch, loss, m1, m2, m3, bands)
        cells = [epoch, loss, m1,m2,m3] + bands
        cells = [round(item, 5) if isinstance(item, float) else item for item in cells]
        print("".join([str(i).ljust(20) for i in cells]))

    def get_indices(self):
        indices = torch.round(self.ann.get_indices() * self.original_feature_size ).to(torch.int64).tolist()
        return list(dict.fromkeys(indices))

    def transform(self, X):
        return X[:,self.get_indices()]

    def is_cacheable(self):
        return False