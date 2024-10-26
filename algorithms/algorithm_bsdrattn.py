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


class Sparse(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.last_k = 0

    def forward(self, X, epoch,l0_norm):
        self.last_k = self.get_k(epoch,l0_norm)
        X = torch.where(torch.abs(X) < self.last_k, 0, X)
        return X

    def get_k(self, epoch,l0_norm):
        l0_norm_threshold = 40
        start = 250
        maximum = 1
        end = 500
        minimum = 0

        if self.dataset == "indian_pines":
            l0_norm_threshold = 50

        if l0_norm <= l0_norm_threshold:
            return self.last_k


        if epoch < start:
            return minimum
        elif epoch > end:
            return maximum
        else:
            return (epoch - start) * (maximum / (end - start))


class ANN(nn.Module):
    def __init__(self, dataset, target_size, class_size, shortlist):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.target_size = target_size
        self.class_size = class_size
        self.shortlist = shortlist
        init_vals = torch.linspace(0.001, 0.99, self.shortlist + 2)
        self.indices = nn.Parameter(
            torch.tensor([ANN.inverse_sigmoid_torch(init_vals[i + 1]) for i in range(self.shortlist)],
                         requires_grad=True).to(self.device))
        self.weighter = nn.Sequential(
            nn.Linear(self.shortlist, 512),
            nn.ReLU(),
            nn.Linear(512, self.shortlist)
        )
        self.classnet = nn.Sequential(
            nn.Linear(self.shortlist, 300),
            nn.ReLU(),
            nn.BatchNorm1d(300),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, self.class_size),
        )
        self.sparse = Sparse(self.dataset)
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    @staticmethod
    def inverse_sigmoid_torch(x):
        return -torch.log(1.0 / x - 1.0)

    def forward(self, linterp, epoch, l0_norm):
        outputs = linterp(self.get_indices())
        channel_weights = self.weighter(outputs)
        channel_weights = torch.abs(channel_weights)
        channel_weights = torch.mean(channel_weights, dim=0)
        sparse_weights = self.sparse(channel_weights, epoch, l0_norm)
        reweight_out = outputs * sparse_weights
        output = self.classnet(reweight_out)
        return channel_weights, sparse_weights, output

    def get_indices(self):
        return torch.sigmoid(self.indices)


class Algorithm_bsdrattn(Algorithm):
    def __init__(self, target_size, dataset, tag, reporter, verbose, test):
        super().__init__(target_size, dataset, tag, reporter, verbose, test)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        self.verbose = verbose
        self.target_size = target_size
        self.shortlist = self.target_size*3
        self.class_size = len(np.unique(self.dataset.get_train_y()))
        self.lr = 0.001
        self.ann = ANN(dataset.get_name(), self.target_size, self.class_size, self.shortlist)
        self.ann.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.original_feature_size = self.dataset.get_train_x().shape[1]
        self.total_epoch = 500
        self.X_train = torch.tensor(self.dataset.get_train_x(), dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(self.dataset.get_train_y(), dtype=torch.int32).to(self.device)

    def get_selected_indices(self):
        self.ann.train()
        self.write_columns()
        optimizer = torch.optim.Adam(self.ann.parameters(), lr=self.lr, weight_decay=self.lr/10)
        linterp = LinearInterpolationModule(self.X_train, self.device)
        y = self.y_train.type(torch.LongTensor).to(self.device)
        l0_norm = self.shortlist
        sparse_weights = None
        for epoch in range(self.total_epoch):
            optimizer.zero_grad()
            if sparse_weights is None:
                l0_norm = self.shortlist
            else:
                l0_norm = torch.norm(sparse_weights, p=0).item()
            channel_weights, sparse_weights, y_hat = self.ann(linterp, epoch, l0_norm)
            deciding_weights = channel_weights
            mean_weight, all_bands, selected_bands = self.get_weights_indices(deciding_weights)

            self.set_all_indices(all_bands)
            self.set_selected_indices(selected_bands)
            self.set_weights(mean_weight)

            mse_loss = self.criterion(y_hat, y)
            l1_loss = self.l1_loss(channel_weights)
            lambda_value = self.get_lambda(l0_norm)
            loss = mse_loss + lambda_value * l1_loss

            loss.backward()
            optimizer.step()
            self.report(epoch, loss.item())

        return self, self.get_indices()

    def get_weights_indices(self, deciding_weights):
        mean_weights = deciding_weights
        if len(mean_weights.shape) > 1:
            mean_weights = torch.mean(mean_weights, dim=0)

        corrected_weights = mean_weights
        if torch.any(corrected_weights < 0):
            corrected_weights = torch.abs(corrected_weights)

        band_indx = (torch.argsort(corrected_weights, descending=True)).tolist()
        return mean_weights, band_indx, band_indx[: self.target_size]

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

    def l1_loss(self, channel_weights):
        return torch.norm(channel_weights, p=1) / torch.numel(channel_weights)

    def get_lambda(self, l0_norm):
        l0_norm_threshold = 40
        if self.dataset == "indian_pines":
            l0_norm_threshold = 50
        if l0_norm <= l0_norm_threshold:
            return 0
        m = 0.001
        if self.dataset.get_name() == "salinas":
            m = 0.08
        elif self.dataset.get_name() == "indian_pines":
            m = 0.01
        return m

