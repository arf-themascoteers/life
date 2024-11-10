import numpy as np
import matplotlib.pyplot as plt


def generate_linear_sequence(start, end, N):
    points = np.linspace(start, end, 2 * N)
    starts = points[:N]
    ends = points[N:]
    return starts, ends


def plot_linear_sequence(starts, ends):
    full_sequence = np.concatenate((starts, ends))
    y_values = [0] * len(full_sequence)

    for i, x in enumerate(full_sequence):
        label = f'start_{i + 1}' if i < len(starts) else f'end_{i - len(starts) + 1}'
        plt.plot(x, 0, 'o', markersize=8)
        plt.annotate(label, (x, 0.01), textcoords="offset points", xytext=(-10, 10), ha='center')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.show()


starts, ends = generate_linear_sequence(0.001, 0.99, 5)
plot_linear_sequence(starts, ends)
