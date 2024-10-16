import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(10, 10, device=device, requires_grad=True)
y = torch.randn(10, 10, device=device)

model = torch.nn.Linear(10, 10).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

output = model(x)
loss = torch.nn.functional.mse_loss(output, y)
loss.backward()

optimizer.step()

print("CUDA Available:", torch.cuda.is_available())
print("Loss after backpropagation:", loss.item())
