"""
d2l 3.5 图像分类数据集（Fashion-MNIST）—— 逐行注释
==================================================
这一节代码很短，但**每行都有讲究**。它干的事只有一件：
    把 7 万张 28×28 的服装灰度图，准备好成「可以喂给模型」的批次。

为什么第 3 章要专门插一节讲数据集？
    因为 3.6「softmax 从零实现」和 3.7「简洁实现」都要用它 ——
    这一节是它们共同的「弹药库」。

运行方式（WSL）：
   conda activate dl
   cd ~/code/learning-notes/notes
   python 3.5-逐行注释.py

【数字核对说明】
  ✅ 本文件所有输出均已**实跑核准**（CPU 版 torch 2.14.0 + torchvision 0.29.1，
     数据已下载并通过官方 md5 校验）。下面标「实测」的就是真实跑出来的值。
  ⚠️ 唯一会变的是 Part 7 里 y[:10] 那行 —— 它跟 shuffle 打乱有关，每次不同。
"""

# ============================================================
# Part 0 · 导包
# ============================================================
import torch
from torch.utils import data              # DataLoader 在这里
from torchvision import transforms        # transforms.ToTensor 在这里
from torchvision.datasets import FashionMNIST   # 数据集本身
# 书里还用 from d2l import torch as d2l 拿 show_images 画图（需要 matplotlib）
# 本文件为了「不依赖 d2l 包也能跑」，把画图部分单独放到 Part 6 说明


# ============================================================
# Part 1 · trans = transforms.ToTensor()  —— 一行干三件事
# ============================================================
# 这是全节最重要的一个零件。它把「原始图片」变成「模型能吃的张量」：
#   ① 类型：PIL 图片对象  ->  torch.Tensor
#   ② 形状：灰度图 (28,28) -> (1,28,28)
#            注意「通道维」在最前面 —— 灰度图通道数是 1，彩色是 3
#            （这就是为什么待会 X 的形状是 (18,1,28,28) 而不是 (18,28,28)）
#   ③ 数值：整数 0~255  ->  浮点 0.0~1.0（除以 255）
#            这一步**非常关键**：神经网络对 0~255 这种大数很敏感，
#            归一化到 0~1 之后训练才稳。这也是后面归一化章节的起点。
trans = transforms.ToTensor()


# ============================================================
# Part 2 · 下载并加载数据集
# ============================================================
# 四个参数分别管：放在哪、要哪一半、怎么变换、要不要联网下载
#   root      : 数据存到哪个目录（首次会在这儿建 FashionMNIST/）
#   train=True: 取训练集；False 取测试集
#   transform : 每取出一张图，就自动套一遍这个变换（就是上面的 ToTensor）
#   download  : 本地没找到就联网下载（有缓存后再跑不会重复下）
#               ★ 已下过一次后它会自动跳过，不用手动改 False
mnist_train = FashionMNIST(root="../data", train=True,  transform=trans, download=True)
mnist_test  = FashionMNIST(root="../data", train=False, transform=trans, download=True)

# ★ 训练集/测试集为什么要分开？
#   训练集用来「学习参数」，测试集只在最后用来「验收」。
#   如果拿测试集训练，就像考前看过答案 —— 成绩虚高，没有意义。

# ★ 首次运行会下载约 30MB 到 ../data/FashionMNIST/raw/。
#   国内直连官方源只有 7.5KB/s（要下 1 小时），**先开代理再跑**，几秒就好：
#      export http_proxy=http://$(ip route show default | awk '{print $3}'):10808
#      export https_proxy=$http_proxy
#   （网关 IP 每次重启 WSL 会变，所以用命令现场取，别记死数字）


# ============================================================
# Part 3 · 数据集有多大？（实测：固定不变）
# ============================================================
print("训练集:", len(mnist_train), " 测试集:", len(mnist_test))
# 实测输出：训练集: 60000  测试集: 10000
# 读法：`len(数据集)` = 有多少张图。这个数字跟 3.2 里 num_examples=1000 是同一个角色。


# ============================================================
# Part 4 · 取一张图看看：dataset[i] 返回的是「一个元组」
# ============================================================
# ★ 这是新手第一个容易困惑的点：
#     mnist_train[0]     取到的是 (图像张量, 标签) —— 两样东西打包
#     mnist_train[0][0]  才是图像本身
#     mnist_train[0][1]  才是标签
img, label = mnist_train[0]        # 用「拆包」一次拿两样，可读性更好
print("图像形状:", img.shape, " dtype:", img.dtype)
print("标签编号:", label)
# 实测输出：图像形状: torch.Size([1, 28, 28])  dtype: torch.float32
#           标签编号: 9   （第 0 张图恰好是 ankle boot 短靴）
#           dtype 是 float32：因为 ToTensor 给的就是 float32（正好是 torch 默认）

# 验证 Part 1 说的三件事：
print("像素范围:", float(img.min()), "~", float(img.max()))
# 实测输出：像素范围: 0.0 ~ 1.0
#   （原始是 0~255 整数，被除以 255 了 —— ToTensor 的第③件事）

# 顺手看看「不套 transform」是什么样子 —— 理解 ToTensor 到底做了什么
raw = FashionMNIST(root="../data", train=True, transform=None, download=False)
img2, label2 = raw[0]
print("不加 transform 时:", type(img2).__name__, img2.size, img2.mode)
# 实测输出：不加 transform 时: Image (28, 28) L
#   读法：PIL.Image 对象、28×28、模式 L（Luminance 灰度）。
#   → 这就是「没用 ToTensor 时的原始形态」，对比一下就懂 ToTensor 的价值了。
#   ★ 对比：同一张图，加 transform 是 tensor(1,28,28)，不加是 PIL.Image(28,28)
#     —— 两样东西差了「通道维」和「数值范围」，全赖 ToTensor 一行之功。


# ============================================================
# Part 5 · 编号 → 文字标签（含一处「教材 vs 官方」的名称差异）
# ============================================================
def get_fashion_mnist_labels(labels):
    """把数字编号翻译成人能看懂的衣服名字"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 用法：模型预测出编号 9，你才知道它说的是「ankle boot（短靴）」
print("\n编号 0~9 对应的衣服:", get_fashion_mnist_labels(range(10)))
# 输出：['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

# ★ 一个能拿去面试的小细节（已实测验证）：
#   torchvision 官方给这 10 类起的名字其实更长一点，
#   打印 mnist_train.classes 得到（这是实测输出）：
#     ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#   d2l 书里做了简化（"t-shirt" 而不是 "T-shirt/top"），且首字母都小写了。
#   顺序完全一致，所以只影响显示，不影响训练 —— 编号 9 永远是同一类。
print("官方类别名:", mnist_train.classes)


# ============================================================
# Part 6 · 画图（书里用 d2l.show_images，底层是 matplotlib）
# ============================================================
# 书里的两行：
#     X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
#     d2l.show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
#
# 逐段读：
#   data.DataLoader(mnist_train, batch_size=18)  造一个「批次搬运工」，每批 18 张
#   iter(...)                                    拿到迭代器
#   next(...)                                    取出第一批（第一个 18 张）
#   X.reshape(18, 28, 28)                        ★ 把 (18,1,28,28) 拍成 (18,28,28)
#                                                因为 matplotlib 不认「通道维在最前」的 1
#   show_images(X, 2, 9, titles=...)             2 行 9 列 = 18 张，标题是衣服名字
#
# 实测确认这行 reshape 的形状变化（见 Part 7 最后一行）：
#     (18,1,28,28)  --reshape-->  (18,28,28)   ← 那个 1 被拍掉了
# d2l.show_images 的作用就是把上面这段画成图。本次不跑（本机没装 matplotlib），
# 你在 WSL 里跑书上的代码就能看到 18 张服装图。


# ============================================================
# Part 7 · DataLoader：为什么它比手写循环香
# ============================================================
batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)
X, y = next(iter(train_iter))
print("\n一批 X:", tuple(X.shape), " 一批 y:", tuple(y.shape))
# 实测输出：一批 X: (256, 1, 28, 28)  一批 y: (256,)
#   读法：256 张图，每张 1 通道 28×28；对应 256 个标签（一维）。
print("这批 y 前 10 个:", y[:10].tolist())
# 实测示例：[3, 5, 9, 5, 7, 2, 8, 5, 3, 2]
#   ★ 这行每次跑都不一样 —— 见下面 shuffle 的说明

# ★ shuffle=True 是什么：
#   每个 epoch 开始前把样本顺序打乱。不打乱的话，模型会「背顺序」——
#   比如数据恰好按类别排好，模型可能学出「前 6000 张都是鞋」这种假规律。
#   这也是为什么上面 y[:10] 每次跑出来都不一样。

# 顺便感受一下「读一遍数据集」要多久（书里有这段计时）：
import time
t = time.time()
for X, y in train_iter:
    continue                     # continue = 什么都不做，只是把数据全过一遍
print(f"读完一遍训练集: {time.time()-t:.2f} 秒")
# 实测输出：读完一遍训练集: 5.86 秒
#   （这个数受机器和硬盘影响，你的会不同；重点感受「读数据本身也要时间」，
#     这就是为什么训练时要看「数据加载是不是瓶颈」）

X18, _ = next(iter(data.DataLoader(mnist_train, batch_size=18)))
print("batch_size=18 时 X:", tuple(X18.shape),
      "-> reshape(18,28,28):", tuple(X18.reshape(18,28,28).shape))
# 实测输出：batch_size=18 时 X: (18, 1, 28, 28) -> reshape(18,28,28): (18, 28, 28)


# ============================================================
# Part 8 · 这一节和 3.6 / 3.7 的关系
# ============================================================
# 3.6、3.7 里会出现这一句：
#     train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 它其实是**把本节这一堆代码打包成了一个函数**：
#     ToTensor + 下载两个数据集 + 各套一个 DataLoader + 返回两个迭代器
# 所以只要你今天看懂了本文件，3.6 开头那行就是「老朋友换了个马甲」。


# ============================================================
# Part 9 · 自测三问
# ============================================================
# Q1 为什么灰度图的形状是 (1,28,28) 而不是 (28,28)？
#    A: 深度学习统一按「通道维在最前」组织：灰度 1 通道，彩色 3 通道。
#       留着这个 1，图像的张量格式就统一了，代码不用分情况写。
#
# Q2 mnist_train[0] 和 mnist_train[0][0] 分别是什么？
#    A: [0] 取到样本 = (图像, 标签) 这个元组；
#       [0][0] 才是图像张量，[0][1] 是标签。
#
# Q3 ToTensor 把整数 0~255 变成 0.0~1.0，为什么必须这么做？
#    A: 让数值落在小范围、量级一致，梯度更新才稳定；
#       不归一化时大像素值会把梯度推得乱跳，训练难收敛。
