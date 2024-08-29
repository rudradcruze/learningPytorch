import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(x)

print(x.view(3, 2))
print(x.permute(1, 0))