import torch

delta = 0.005
min_vector = torch.tensor([i * delta for i in range(10)], dtype=torch.float32)
max_vector = torch.tensor([1 - (9 - i) * delta for i in range(10)], dtype=torch.float32)

num_vectors = 5
weights = torch.tensor([0.1**(i / (num_vectors - 1)) for i in range(num_vectors)], dtype=torch.float32).unsqueeze(1)
vectors = min_vector + weights * (max_vector - min_vector)

print(vectors)
