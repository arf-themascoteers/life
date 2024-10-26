import matplotlib.pyplot as plt
from algorithms.algorithm_bsdr import LinearInterpolationModule
import torch


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
