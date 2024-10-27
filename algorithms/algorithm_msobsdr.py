from algorithm import Algorithm
import torch
import torch.nn as nn
import numpy as np
from train_test_evaluator import evaluate_split
from torch.jit import fork, wait
import math
from itertools import accumulate


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


class Agent(nn.Module):
    def __init__(self, target_size, class_size, classification, offset, start, r1, r2, last=False, var = 0, lin=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = target_size
        self.class_size = class_size
        self.classification = classification
        self.last = last
        self.var = var
        self.lin = lin
        self.r1 = torch.tensor(r1, dtype=torch.float32)
        self.r2 = torch.tensor(r2, dtype=torch.float32)
        #print(r1,r2)
        init_vals = torch.linspace(0.001, 0.99, self.target_size + 2)
        displacement = offset*start
        init_vals[1:-1] = init_vals[1:-1] + displacement
        init_vals = torch.clamp(init_vals, max=0.99)
        self.raw_indices = nn.Parameter(
            torch.tensor([Agent.inverse_sigmoid_torch(init_vals[i + 1]) for i in range(self.target_size)],
                         requires_grad=True).to(self.device))
        self.linear = nn.Sequential(
            nn.Linear(self.target_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.class_size)
        )
        # print(torch.min(self.indices))
        # print(torch.max(self.indices))
        if self.classification:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

    @staticmethod
    def inverse_sigmoid_torch(x):
        return -torch.log(1.0 / x - 1.0)

    def forward(self, linterp, y):
        zero = torch.tensor(0,dtype=torch.float32).to(self.device)
        indices = self.get_indices()
        outputs = linterp(indices)
        soc_hat = self.linear(outputs)
        if self.class_size == 1:
            soc_hat = soc_hat.reshape(-1)
        loss = self.criterion(soc_hat,y)
        if self.lin:
            return soc_hat, loss, zero,zero,zero

        norm = torch.norm(indices, p=2)
        r1_loss = torch.relu(self.r1 - norm)
        r_loss = r1_loss
        if not self.last:
            r2_loss = torch.relu(norm - self.r2)
            r_loss = r_loss + r2_loss

        var_loss = zero
        if self.var !=0 :
            var_loss = (1/torch.var(self.raw_indices))*self.var

        successive_loss = torch.sum(torch.stack([torch.relu(self.raw_indices[i] - self.raw_indices[i + 1]) for i in range(len(self.raw_indices) - 1)]))
        return soc_hat, loss, r_loss, successive_loss, var_loss

    def get_indices(self):
        return torch.sigmoid(self.raw_indices)


class ANN(nn.Module):
    def __init__(self, target_size, class_size, original_size, classification):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = target_size
        self.class_size = class_size
        self.original_size = original_size
        self.num_agents = 10
        self.classification = classification

        band_unit = 1 / original_size

        var_agents = [0,0,5,10,0,0,20,40,0,0]
        linear_spaced_agent = [4]

        rs = [0]+self.get_rs()
        self.agents = nn.ModuleList(
            [Agent(self.target_size, self.class_size, self.classification, band_unit, i, rs[i], rs[i+1], last=(i==self.num_agents-1),
                   var=var_agents[i], lin=(i in linear_spaced_agent)) for i in range(self.num_agents)]
        )

        self.best = 0
        self.print_weights()

    def print_weights(self):
        for i in range(self.num_agents):
            print("\t".join([str(round(i.item(),5)) for i in self.agents[i].get_indices()]))
            print("\t".join([str(round(i.item(), 5)) for i in self.agents[i].get_indices()*self.original_size]))

    def get_rs(self):
        r_values = [1/math.pow(1+1,2) for i in range(self.num_agents)]
        s = sum(r_values)
        r_values = [r / s for r in r_values]
        r_values = list(accumulate(r_values))
        r_values = [r*math.sqrt(self.target_size) for r in r_values]
        return r_values

    def forward(self, linterp, y):
        futures = [fork(linear, linterp, y) for linear in self.agents]
        y_preds, losses, r_losses, successive_loss, var_loss = zip(*[wait(future) for future in futures])
        losses = torch.stack(losses)
        r_losses = torch.stack(r_losses)
        successive_loss = torch.stack(successive_loss)
        var_loss = torch.stack(var_loss)
        self.best = torch.argmin(losses)
        output = y_preds[self.best]
        loss = torch.sum(losses) + 40*torch.sum(r_losses) + 40*torch.sum(successive_loss) + torch.sum(var_loss)

        ls = [str(round(l.item(),5)) for l in losses]
        ls = "\t".join(ls)

        rs = [str(round(l.item(),5)) for l in r_losses]
        rs = "\t".join(rs)

        ss = [str(round(l.item(),5)) for l in successive_loss]
        ss = "\t".join(ss)

        vs = [str(round(l.item(),5)) for l in var_loss]
        vs = "\t".join(vs)

        s = ls + "\t\t" + rs + "\t\t" + ss + "\t\t" + vs

        agent_bands = [a.get_indices()*self.original_size for a in self.agents]
        band_strs = []
        for ab in agent_bands:
            band_strs.append("\t".join([str(round(b.item())) for b in ab]))

        band_strs = "\t\t".join(band_strs)

        s= s + "\t\t" + band_strs

        print(s)

        return output, loss

    def get_ann_indices(self):
        return self.agents[self.best].get_indices()


class Algorithm_msobsdr(Algorithm):
    def __init__(self, target_size, dataset, tag, reporter, verbose, test):
        super().__init__(target_size, dataset, tag, reporter, verbose, test)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        self.verbose = verbose
        self.target_size = target_size
        self.classification = dataset.is_classification()
        if self.classification:
            self.class_size = len(np.unique(self.dataset.get_bs_train_y()))
            self.lr = 0.01
            self.total_epoch = 1000
        else:
            self.class_size = 1
            self.lr = 0.001
            self.total_epoch = 1000
        self.original_feature_size = self.dataset.get_bs_train_x().shape[1]
        self.ann = ANN(self.target_size, self.class_size, self.original_feature_size, self.classification)
        self.ann.to(self.device)


        self.X_train = torch.tensor(self.dataset.get_bs_train_x(), dtype=torch.float32).to(self.device)
        ytype = torch.float32
        if self.classification:
            ytype = torch.int32
        self.y_train = torch.tensor(self.dataset.get_bs_train_y(), dtype=ytype).to(self.device)

    def get_selected_indices(self):
        self.ann.train()
        self.write_columns()
        optimizer = torch.optim.Adam(self.ann.parameters(), lr=self.lr, weight_decay=self.lr/10)
        linterp = LinearInterpolationModule(self.X_train, self.device)
        if self.classification:
            y = self.y_train.type(torch.LongTensor).to(self.device)
        else:
            y = self.y_train
        for epoch in range(self.total_epoch):
            optimizer.zero_grad()
            y_hat, loss = self.ann(linterp, y)
            loss.backward()
            optimizer.step()
            r2_train = 0
            if not self.classification:
                r2_train = r2_score_torch(y, y_hat)
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
        indices = torch.round(self.ann.get_ann_indices() * self.original_feature_size).to(torch.int64).tolist()
        return list(dict.fromkeys(indices))

    def transform(self, X):
        return X[:,self.get_indices()]

    def is_cacheable(self):
        return False