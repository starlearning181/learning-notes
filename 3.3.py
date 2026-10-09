import torch
from torch import nn
from torch.utils import data

#第一部分 造数据
def synthetic_data(w,b,num_examples):
    X=torch.normal(0,1,(num_examples,len(w)))
    y=X@w+b
    y+=torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))

true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000)

#第二部分 读数据
def load_array(data_arrays,batch_size,is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
batch_size=10
data_iter=load_array((features,labels),batch_size)
print('一个批次的 X 形状:', next(iter(data_iter))[0].shape)

#第三部分 定义模型：nn.Sequential+nn.Linear
net=nn.Sequential(nn.Linear(2,1))

#第四部分 初始化参数
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

#损失函数 nn.MSELoss
loss=nn.MSELoss()

#第六部分 优化器:torch.optim.SGD
trainer=torch.optim.SGD(net.parameters(),lr=0.03)

#第七部分 训练循环
num_epoch=3
for epoch in range(num_epoch):
    for X,y in data_iter:
        I=loss(net(X),y)
        trainer.zero_grad()
        I.backward()
        trainer.step()
    I=loss(net(features),labels)
    print(f'epoch{epoch+1},loss{I:f}')

#第八部分 验收
w=net[0].weight.data
print('w 的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b 的估计误差：', true_b - b)
