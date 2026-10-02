import torch
import random
torch.manual_seed(42)
random.seed(42)
#第一部分
def synthetic_data(w, b, num_examples):
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
#第二部分
def data_iter(batch_size,features,labels):  #定义数据类型synthetic_data和data_iter是不同的数据类型吗？
    num_examples=len(features)   #定义样本长度为features的长度1000
    indices=list(range(num_examples))  #这行什么意思，indices有什么作用
    random.shuffle(indices)   #随机打乱indices的数据
    for i in range(0,num_examples,batch_size):  #这句代码是什么意思？括号里面的内容都分别代表什么意思？for循环的结构是什么？和C语言的语法用处一样吗？
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])   #这行再详细解释一下
        yield features[batch_indices],labels[batch_indices]   #这一行也是，详细解释一下，和上一行一起形成什么作用？
batch_size=10   #将样本数据分为10个一批？
for X,y in data_iter(batch_size,features,labels):
    print(X.shape)  #X的形状为什么是10行1列
    print(y.shape)  #y的形状为什么是10行1列
    break   #为什么break后只看了一批样本
#第三部分
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
print(w.shape)   #32行定义的w的形状是2行1列吗？
print(b.shape)   #b的元素是一维数字0？
#第四部分
def linreg(X,w,b):
    return X@w+b

def squared_loss(y_hat,y):
    return(y_hat-y.reshape(y_hat.shape))**2/2

def sgd(params,lr,batch_size):
    with torch.no_grad():
        for p in params:
            p-=lr*p.grad/batch_size
            p.grad.zero_()
#第五部分
lr=0.03
num_epochs=3
net=linreg
loss=squared_loss
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l=loss(net(X,w,b),y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l=loss(net(features,w,b),labels)
        print(f'epoch{epoch+1},loss{float(train_l.mean()):f}')
#第六部分
with torch.no_grad():
    print(f'w 的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b 的估计误差: {true_b - b}')
