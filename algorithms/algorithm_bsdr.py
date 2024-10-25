from algorithm import Algorithm
from algorithms.bsdr.bsdr import BSDR
import numpy as np
import torch


class AlgorithmBSDR(Algorithm):
    def __init__(self, target_size, dataset, tag, reporter, verbose, test):
        super().__init__(target_size, dataset, tag, reporter, verbose, test)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        self.verbose = verbose
        self.bsdr = BSDR(self.target_size, len(np.unique(self.dataset.get_train_y())))

    def get_selected_indices(self):
        self.bsdr.fit(self.splits.train_x, self.splits.train_y, self.splits.validation_x, self.splits.validation_y)
        return self.bsdr, self.bsdr.get_indices()

    def get_name(self):
        return "bsdr"