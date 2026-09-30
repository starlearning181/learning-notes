import torch
import random
torch.manual_seed(42)
random.seed(42)
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声，共 num_examples 个样本"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print(features.shape)
print(labels.shape)
print(features[0])
print(labels[0])
def data_iter(batch_size, features, labels):
    """把数据打乱后，按 batch_size 一批一批地吐出来"""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X.shape)
    print(y.shape)
    break
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
print(w.shape)
print(b.shape)
def linreg(X, w, b):
    """模型：给定输入和参数，算出预测值"""
    return X @ w + b          # (10,2) @ (2,1) = (10,1)，正好跟 labels 同形状
def squared_loss(y_hat, y):
    """损失：平方误差再除以 2（除以 2 纯粹是为了求导后更简洁）"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
def sgd(params, lr, batch_size):
    """优化器：随机梯度下降——沿着梯度反方向，把参数挪一小步"""
    with torch.no_grad():
       for p in params:
            p -= lr * p.grad / batch_size
            p.grad.zero_()
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
print(f'w 的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b 的估计误差: {true_b - b}')
