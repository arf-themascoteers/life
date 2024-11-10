import numpy as np

def entropy(values):
    probs = values / np.sum(values)
    return -np.sum(probs * np.log(probs + 1e-10))

arrays = [
    np.array([1, 1, 1, 1]),
    np.array([10, 1, 1, 1]),
    np.array([100, 0, 0, 0]),
    np.array([0.25, 0.25, 0.25, 0.25]),
]

for i, arr in enumerate(arrays):
    print(f"Array {i+1}: {arr}, Entropy: {entropy(arr):.4f}")
