import torch
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

#第一部分  trans = transforms.ToTensor()
trans=transforms.ToTensor()

#第二部分 下载并加载数据集
mnist_train=FashionMNIST(root="data",train=True,transform=trans,download=True)
mnist_test=FashionMNIST(root="data",train=False,transform=trans,download=True)

#第三部分 数据集大小
print("训练集:", len(mnist_train), " 测试集:", len(mnist_test))

#第四部分
img,label=mnist_train[0]
print("图像形状:", img.shape, " dtype:", img.dtype)
print("标签编号:", label)
print("像素范围:", float(img.min()), "~", float(img.max()))
raw=FashionMNIST(root="data",train=True,transform=None,download=False)
img2,label2=raw[0]
print("不加 transform 时:", type(img2).__name__, img2.size, img2.mode)

#第五部分 编号 → 文字标签
def get_fashion_mnist_labels(labels):
   text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
   return[text_labels[int(i)]for i in labels]
print("\n编号 0~9 对应的衣服:", get_fashion_mnist_labels(range(10)))
print("官方类别名:", mnist_train.classes)

#第六部分画图

#第七部分DateLoader
batch_size=256
train_iter=data.DataLoader(mnist_train,batch_size,shuffle=True)
X,y=next(iter(train_iter))
print("\n一批 X:", tuple(X.shape), " 一批 y:", tuple(y.shape))
print("这批 y 前 10 个:", y[:10].tolist())
import time
t=time.time()
for X,y in train_iter:
    continue
print(f"读完一遍训练集: {time.time()-t:.2f} 秒")
X18,_=next(iter(data.DataLoader(mnist_train,batch_size=18)))
print("batch_size=18 时 X:", tuple(X18.shape),"-> reshape(18,28,28):", tuple(X18.reshape(18,28,28).shape))
