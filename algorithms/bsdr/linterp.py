import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class LinearInterpolationModule(nn.Module):
    def __init__(self, y_points, device='cpu'):
        super(LinearInterpolationModule, self).__init__()
        self.device = device
        self.y_points = y_points.to(device)

    def forward(self, x_new_):
        x_new = x_new_.to(self.device)
        batch_size, num_points = self.y_points.shape
        x_points = torch.linspace(0,1, num_points).to(self.device).expand(batch_size, -1).contiguous()
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


if __name__ == "__main__":
    y_points = torch.tensor([[0, 2, 4], [0, 2, 4]], dtype=torch.float32)

    y_points_ = y_points[0].tolist()
    x_points_ = torch.linspace(0, 1, len(y_points_)).tolist()
    print(x_points_)
    plt.scatter(x_points_, y_points_)

    interp = LinearInterpolationModule(y_points, device='cuda')
    x_new = torch.tensor([1.5, 0.25, 0,0.5,1,-1], dtype=torch.float32, device='cuda')
    y_new = interp(x_new)
    print(y_new)
    y_new = y_new[0].tolist()
    x_new = x_new.tolist()

    plt.scatter(x_new, y_new, c="red", marker=".")
    plt.show()
