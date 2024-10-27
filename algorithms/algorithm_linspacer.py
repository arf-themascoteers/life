from algorithm import Algorithm
import torch


class Algorithm_linspacer(Algorithm):
    def __init__(self, target_size:int, dataset, tag, reporter, verbose, test):
        super().__init__(target_size, dataset, tag, reporter, verbose, test)
        self.indices = None

    def get_selected_indices(self):
        original_size = self.dataset.get_bs_train_x().shape[1]
        indices = self.get_points(0, original_size-1, self.target_size,1)
        self.indices = torch.round(indices).long().tolist()
        self.set_selected_indices(self.indices)
        self.set_weights([1]*self.target_size)
        return self, self.indices

    def transform(self, X):
        return X[:,self.indices]

    def is_cacheable(self):
        return False

    def get_points(self, low, up, target_size, group_size):
        if group_size == 1:
            split = (up - low) / (2 * target_size)
            start = low + split
            end = up - split
            return torch.linspace(start, end, target_size)

        anchors = torch.linspace(low, up, target_size + 1)
        all_points = []
        for i in range(target_size):
            points = torch.linspace(anchors[i], anchors[i + 1], group_size)
            for p in points:
                all_points.append(p)

        all_points = torch.stack(all_points)
        return all_points
