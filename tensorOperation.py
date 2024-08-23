import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6, 32, 53, 12])

sumx = torch.sum(x, dim=0)
print(sumx)

value, indices = torch.max(x, dim=0)
print(value)
print(indices)

value, indices = torch.min(x, dim=0)
print(value)
print(indices)

absx = torch.abs(x)
print(absx)

x = torch.tensor([1, 2, 3, 4, 353, 32])
z = torch.argmax(x, dim=0)
print(z)

z = torch.argmin(x, dim=0)
print(z)

z = torch.mean(x.float())
print(z)

zz = torch.eq(x, y)
print(zz)

sortValues, indexes = torch.sort(y, dim=0, descending=False)
print(sortValues)
print(indexes)

z = torch.clamp(x, min=0)
print(z)

z = torch.clamp(x, min=2)
print(z)

z = torch.tensor([1,0,1,1,0,1], dtype=torch.bool)
print(z)

yy = torch.any(z)
print(yy)

z = torch.tensor([1,0,1,1,0,1], dtype=torch.bool)
yy = torch.all(z)
print(yy)
