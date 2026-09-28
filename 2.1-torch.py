import torch

x = torch.arange(12)
print(x)
print(x.shape)
X =x.reshape(3,4)
print(X)

print(torch.zeros(2,3,4))
print(torch.randn(3,4))

print(X.sum())
y = torch.arange(4)
print(X + y)
print(X.sum(0))    # 按列求和：每一列的 3 个数加起来 → 输出 4 个数
print(X.sum(1))    # 按行求和：每一行的 4 个数加起来 → 输出 3 个数
