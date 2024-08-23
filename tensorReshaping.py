
## Useful Operation

print(torch.where(x>5, x, x*2))

print(torch.tensor([1,2,3,0,1,1,12,2,24,4,4,5,5]).unique())

print(x.ndimension()) # 5x5x5

print(x.numel())44

## Tensor Reshaping

x = torch.arange(9)

x_3x3 = x.view(3, 3)

print(x_3x3)

x_3x3 = x.reshape(3, 3)
print(x_3x3)

y = x_3x3.t()
print(y)

print(y.contiguous().view(9))

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))

print(torch.cat((x1, x2), dim=0))
print(torch.cat((x1, x2), dim=0).shape)
print(torch.cat((x1, x2), dim=1))
print(torch.cat((x1, x2), dim=1).shape)

z = x1.view(-1)
print(z)
print(z.shape)

batch_size = 64
x = torch.rand((batch_size, 2, 5))
z = x.view(batch_size, -1)
# print(z)
print(z.shape)

z = x.permute(0, 2, 1)
# print(z)
print(z.shape)

x = torch.arange(10)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1) # 1x1x10
print(x)
print(x.shape)

z = x.squeeze(1)
print(z.shape)
