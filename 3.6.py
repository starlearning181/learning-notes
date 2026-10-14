import os
import torch
from d2l import torch as d2l

batch_size=256
_HERE=os.path.dirname(os.path.abspath(__file__))
_ROOT=os.path.normpath(os.path.join(_HERE,"data"))
import torchvision
from torchvision import transforms
trans=transforms.ToTensor()
mnist_train=torchvision.datasets.FashionMNIST(root=_ROOT,train=True,transform=trans,download=True)
mnist_test=torchvision.datasets.FashionMNIST(root=_ROOT,train=False,transform=trans,download=True)
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=d2l.get_dataloader_workers())
test_iter=torch.utils.data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=d2l.get_dataloader_workers())
print("[1] 数据就绪：训练 60000 张 / 测试 10000 张，每批", batch_size)
num_inputs=748
num_output=10

W=torch.normal(0,0.01,size=(784,10),requires_grad=True)
b=torch.zeros(10,requires_grad=True)
print("[2] 参数初始化：W", tuple(W.shape), " b", tuple(b.shape), " 共 7850 个参数")

#softmax函数
def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)
    return X_exp/partition
X=torch.normal(0,1,(2,5))
X_prob=softmax(X)
print("[3] softmax 示例（形状", tuple(X_prob.shape), "):")
print("    第一行 =", [round(v, 4) for v in X_prob[0].tolist()])
print("    每行和 =", [round(v, 4) for v in X_prob.sum(1).tolist()], "← 必须都是 1.0")

#定义模型net：把“算得分”和“软化”两步串起来
def net(X):
    return softmax(torch.matmul(X.reshape((-1,W.shape[0])),W)+b)

#交叉损失熵
y=torch.tensor([0,2])
y_hat=torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
print("[5] 按真实标签挑出预测概率 y_hat[[0,1],[0,2]] =", y_hat[[0, 1], y].tolist())

def cross_entropy(y_hat,y):
    return-torch.log(y_hat[range(len(y_hat)),y])
print("     交叉熵损失 =", [round(v, 4) for v in cross_entropy(y_hat, y).tolist()],"← -log(0.1)=2.3026, -log(0.5)=0.6931")

#精度accuracy：只看“预测对不对”，不看概率高低
def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=y_hat.argmax(axis=1)
    cmp=(y_hat.type(y.dtype)==y)
    return float(cmp.type(y.dtype).sum())

#Accumulator累加器
class Accumulator:
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[a+float(b)for a,b in zip(self.data,args)]
    def reset(self):
        self.data=[0,0]*len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]

acc=Accumulator(2)
acc.add(3,5)
acc.add(1,5)
print("[7] Accumulator 演示：add(3,5) add(1,5) 后 =", acc.data, "← 逐位累加")

#在测试集上评估精度
def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric=Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]
print("[8] 训练前测试精度 =", round(evaluate_accuracy(net, test_iter), 4),"← 10 类瞎猜 ≈ 0.1")

#训练第一轮
def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric=Accumulator(3)
    for X,y in train_iter:
        y_hat=net(X)
        I=loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            I.mean().backward()
            updater.step()
        else:
            I.sum().backward()
            updater(X.shape[0])
        metric.add(float(I.sum()),accuracy(y_hat,y),y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]

#定制优化器
def updater(batch_size):
    return d2l.sgd([W,b],0.1,batch_size)
lr=0.1

#训练主循环
def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    for epoch in range(num_epochs):
        train_metrics=train_epoch_ch3(net,train_iter,loss,updater)
        test_acc=evaluate_accuracy(net,test_iter)
        print("    epoch %2d | 训练损失 %.4f | 训练精度 %.4f | 测试精度 %.4f"% (epoch + 1, train_metrics[0], train_metrics[1], test_acc))
    train_loss,train_acc=train_metrics
    assert train_loss<0.5,train_loss
    assert train_acc<=1 and train_acc>0.7,train_acc
    assert test_acc<=1 and test_acc>0.7,test_acc
num_epoch=10
print("[11] 开始训练(10 轮，每轮过完 60000 张）")
train_ch3(net,train_iter,test_iter,cross_entropy,num_epoch,updater)

#预测，看啊可能模型实际认出了什么
def get_fashion_mnist_labels(labels):
    text_labels=['t_shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return[text_labels[int(i)]for i in labels]

def predict_ch3(net,test_iter,n=6):
    for X,y in test_iter:
        break
    trues=get_fashion_mnist_labels(y)
    preds=get_fashion_mnist_labels(net(X).argmax(axis=1))
    print("[12] 前 6 张的真实标签 vs 模型预测：")
    for i in range(n):
        flag="OK"if trues[i]==preds[i] else"X"
        print("     真实: %-12s 预测: %-12s %s" % (trues[i], preds[i], flag))
predict_ch3(net, test_iter)
