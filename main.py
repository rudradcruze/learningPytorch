from turtledemo.penrose import start

import torch

# print(torch.__version__)


# ================================ #
#      Initializing Tensor         #
# ================================ #

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1,2,3],
                         [4,5,6]], dtype=torch.float32,
                         device=device, requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Other common initialization methods
X = torch.empty(size = (3,3))
# print(X)
X = torch.zeros((3, 3))
# print(X)
X = torch.rand((3, 3))
X = torch.ones((3, 3))
X = torch.eye(5, 5)
X = torch.arange(start=0, end=5, step=1)
X = torch.linspace(start=0.1, end=1, steps=10)
X = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
X = torch.empty(size=(1, 5)).uniform_(0, 1)
X = torch.diag(torch.ones(5))

# How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(5)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())
print(tensor.double())

# Array to tensor conversion and vice-versa
import numpy as np
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_back_array = tensor.numpy()

print(tensor)
print(np_back_array)