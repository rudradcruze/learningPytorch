## Tensor Indexing

batchSize = 10
features = 25

x = torch.rand((batchSize, features))

print(x[0].shape)

print(x[:, 0].shape)

x[0, 0] = 100
print(x[0])

## Fancy Indexing

x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand(3,5)

rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])

print(x[rows, cols])
 print(x[rows, cols].shape)

x = torch.arange(10)
print(x[(x<2) | (x>8)])
print(x[(x<2) & (x>8)])
print(x[x.remainder(2) == 0])