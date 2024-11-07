from algorithm import Algorithm
from auswahl import SPA, VIP
import numpy as np


class Algorithm_spa(Algorithm):
    def __init__(self, target_size: int, dataset, tag, reporter, verbose, test, props):
        super().__init__(target_size, dataset, tag, reporter, verbose, test, props)

    def get_selected_indices(self):
        vip = VIP()
        selector = SPA(n_features_to_select=self.target_size)
        vip.fit(self.dataset.get_bs_train_x(), self.dataset.get_bs_train_y())
        mask = vip.vips_ > 0.3
        selector.fit(self.dataset.get_bs_train_x(), self.dataset.get_bs_train_y(), mask=mask)
        support = selector.get_support(indices=True)
        self.set_selected_indices(support)
        self.set_weights(np.zeros(self.dataset.get_bs_train_x().shape[1]))
        return selector, support

    def is_cacheable(self):
        return False
