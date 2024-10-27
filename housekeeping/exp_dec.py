# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define the range of x values
# x = np.linspace(0, 5, 100)
#
# # Define the exponential decay function
# y = np.exp(-x)
#
# # Plot the function
# plt.plot(x, y, label="f(x) = e^(-x)")
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.title("Exponential Decay Function")
# plt.legend()
# plt.grid(True)
# plt.show()


import torch

v1 = torch.tensor([0.0, 0.1, 0.2])
direction = torch.tensor([0.1, 0.2, 0.3]) - v1
direction = direction / torch.norm(direction) * 0.5  # Normalize to 0.5 distance
v3 = v1 + direction