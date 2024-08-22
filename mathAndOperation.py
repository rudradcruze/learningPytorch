import torch

from main import tensor

# ================================ #
#       Math And Operation         #
# ================================ #


x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)

z2 = torch.add(x, y)
z3 = x + y

# Subtraction
z4 = x - y

# Division
z5 = torch.true_divide(x, y)

# implace operation
# t = torch.zeros(3)
t = torch.ones(3)
t.add_(x)
t += x # t = t + x (not going to work)

# Exponentiation
p = x.pow(2)
p = x ** 2

# Simple comparison
z  = x > 0
z = x < 0

# Matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) # 2x3
x4 = x1.mm(x2)

# Matrix exponentiation
matrix_exp = torch.rand(5, 5)
print(matrix_exp)

print(matrix_exp.matrix_power(3))

# Element wize multiplication
z = x * y
print(z)

# dot product
z = torch.dot(x, y)
print(z)

# Batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2) ## (batch, n, p)

# Example of Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2
z = x1 ** x2

print(z)

# print(out_bmm)