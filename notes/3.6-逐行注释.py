# -*- coding: utf-8 -*-
"""
d2l 3.6 从零开始实现 softmax 回归 —— 逐行注释版
================================================================
这是第 3 章最硬、也最值的一节：不调用任何"层"，用手写张量运算 + 手写梯度
下降，把 Fashion-MNIST 十分类器从头训出来。3.7「简洁实现」会把这一整份
代码压成 5 行（换 nn.Linear + nn.CrossEntropyLoss + SGD）,所以 3.6 的
全部价值就是——把黑盒拆开看一遍。

【本节的 4 个新东西】
  ① softmax 函数（把得分变成概率）
  ② 交叉熵损失（衡量"预测分布 vs 真实标签"差多远）
  ③ Accumulator 累加器（小工具，贯穿后面所有章节）
  ④ 训练循环的完整骨架（train_epoch_ch3 / train_ch3，第 4~7 章反复复用）

【★ 运行位置提醒 —— 这个坑你踩过两次】
  root 的 "../data" 是相对"你敲命令时所在的目录"，不是相对文件位置！
    本文件在 notes/ 里  → 先 cd 进 notes，root 写 "../data"
    你自己手敲的 3.6.py 放仓库根 → 先 cd 到仓库根，root 写 "data"
  跑错位置的症状：终端出现下载进度条，速度龟速（这是最典型信号）

【正确跑法（WSL 终端，逐行敲）】
    cd ~/code/learning-notes/notes
    conda activate dl
    python 3.6-逐行注释.py

【和书上的两处差异（我已改，并说明原因）】
  1. 书上每一段的最后一行是"裸表达式"（如 X_prob.sum(1)），Jupyter 里会
     自动显示结果；你是终端跑，什么都不显示。所以本文件全部加上了 print()。
  2. 书上用 Animator 类画"实时刷新曲线"（需要图形界面 + IPython），你在
     WSL 终端跑不出来。我换成了 print 逐轮打印，训练逻辑一字未改。
"""
import os
import torch
# from IPython import display
# ↑ 书上第 2 行。它的作用是"原地刷新输出"做动画；终端跑用不上。
#   如果你的环境没装 IPython，这行会报 No module named 'IPython'，删掉即可。
from d2l import torch as d2l


# ================================================================
# ① 读数据（3.5 已经手写过这个过程，这里一行代劳）
# ================================================================
batch_size = 256
# ★ 免疫版路径：让数据目录"跟着本文件走"，跟你在哪个目录敲命令无关
_HERE = os.path.dirname(os.path.abspath(__file__))          # 本文件所在目录（notes/）
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "data")) # → learning-notes/data
# ⚠️ d2l 0.17 的 load_data_fashion_mnist(batch_size, resize=None) 不接收 root 参数，
#    它内部把 root 硬编码成 "../data"（相对"敲命令的目录"）。
#    所以这里不调用它，而是把它的内部逻辑原样展开（就是 3.5 手写过的那套），
#    只把 root 换成上面的 _ROOT —— 在任何目录跑都指向 learning-notes/data：
import torchvision
from torchvision import transforms
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root=_ROOT, train=True,
                                                transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root=_ROOT, train=False,
                                               transform=trans, download=True)
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                         num_workers=d2l.get_dataloader_workers())
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                        num_workers=d2l.get_dataloader_workers())
# 逐行：
#   load_data_fashion_mnist 把"下载 + ToTensor + 切 batch + shuffle"打包好了，
#   内部逻辑就是 3.5 我们手写的那一套（FashionMNIST + DataLoader）。
#   train_iter：60000 张，每批 256 张，shuffle=True（训练必须打乱）
#   test_iter ：10000 张，每批 256 张，不打乱
#   拿出一批来看：X = (256, 1, 28, 28) float32，y = (256,) 整数标签
print("[1] 数据就绪：训练 60000 张 / 测试 10000 张，每批", batch_size)


# ================================================================
# ② 初始化模型参数 W 和 b（"从零"两个字的体现）
# ================================================================
num_inputs = 784      # 28 × 28：一张图拉平后就是 784 个像素
num_outputs = 10      # 10 个类别：T恤/裤子/套衫/连衣裙/外套/凉鞋/衬衫/运动鞋/包/短靴

W = torch.normal(0, 0.01, size=(784, 10), requires_grad=True)
b = torch.zeros(10, requires_grad=True)
# 逐行：
#   W 形状 (784, 10)：784 个输入 × 10 个输出，一共 7840 个待学参数
#     torch.normal(0, 0.01, size=...) = 均值 0、标准差 0.01 的正态随机数
#     为什么不用全 0 初始化？因为全 0 时所有类别的输出完全一样，梯度也一起
#     打转，网络"分不出谁是谁"，学得极慢。小结：权重必须随机打破对称性。
#     （偏置 b 全 0 是安全的，偏置不参与对称性问题）
#     requires_grad=True 就是 2.4 学过的"记账"：告诉 PyTorch 这是我要求导的参数。
#   少了这个 True，后面 l.backward() 会直接报：
#     "element 0 of tensors does not require grad"
print("[2] 参数初始化：W", tuple(W.shape), " b", tuple(b.shape), " 共 7850 个参数")


# ================================================================
# ③ softmax 函数（本节第 1 个核心，全节最重要的一段）
# ================================================================
def softmax(X):
    X_exp = torch.exp(X)                    # 每个元素取 e 的幂 → 全部变成正数
    partition = X_exp.sum(1, keepdim=True)  # 沿"第 1 维（每行内部）"求和，保持二维
    return X_exp / partition                # 每行除以自己的和 → 每行之和 = 1
# 逐行（慢读，这段是第 3 章的命门）：
#   输入 X 形状 (batch, 10)：一行 = 一张图的 10 个类别"原始得分"（叫 logit）
#   问题：得分可以是任意实数（−5.2、0.8、13.7…），不是概率。softmax 的任务：
#         把一行任意实数，变成一个"合法概率分布"——全正、和为 1。
#   第 1 行 torch.exp：e^x 把 (−∞,+∞) 映射到 (0,+∞)，解决"负数不能当概率"
#   第 2 行 sum(1, keepdim=True)：
#     · sum(1) 的 1 = 沿"第 1 维"求和 = 每行内部求和（0 才是列方向）
#     · keepdim=True 是命门！保持 (256,1) 形状。若不加，会压成 (256,)，
#       下一步除法广播时 (256,10) ÷ (256,) 维度长度 10 ≠ 256 → 直接报错
#   第 3 行 X_exp / partition：
#     (256,10) ÷ (256,1) → 广播机制把 (256,1) 自动复制成 (256,10)，
#     于是"每行各自除以本行的和"，得到一个合法概率分布。
#   ⚠️ 数值稳定：真实工程实现要减去每行最大值再 exp（见"改坏实验④"）
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print("[3] softmax 示例（形状", tuple(X_prob.shape), "）：")
print("    第一行 =", [round(v, 4) for v in X_prob[0].tolist()])
print("    每行和 =", [round(v, 4) for v in X_prob.sum(1).tolist()], "← 必须都是 1.0")


# ================================================================
# ④ 定义模型 net：把"算得分"和"软化"两步串起来
# ================================================================
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
# 逐行：
#   X 进来是 (256, 1, 28, 28) → X.reshape((-1, 784)) 拉平成 (256, 784)
#     -1 的含义："这一维你自己算"（256×1×28×28 ÷ 784 = 256）
#     W.shape[0] 就是 784——写它比写死 784 更"活"，换数据集时不用改代码
#   matmul((256,784), (784,10)) → (256,10)：每张图得到 10 个类别得分
#   + b：(256,10) + (10,) → 广播，每行都加同一个 b
#   softmax(...) → 每行成为 10 个概率，和 = 1
#   一句话：net = 「拉平 → 线性变换 → 软化」，这就是一个"单层神经网络"


# ================================================================
# ⑤ 交叉熵损失（本节第 2 个核心）
# ================================================================
y = torch.tensor([0, 2])                                 # 两张图的真实标签
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])  # 模型给出的 2×3 概率

print("[5] 按真实标签挑出预测概率 y_hat[[0,1],[0,2]] =", y_hat[[0, 1], y].tolist())
# 高级索引：y_hat[行下标, 列下标]
#   行 = [0,1]，列 = y = [0,2]（第一张图要第 0 类，第二张图要第 2 类）
#   → 挑出 0.1 和 0.5：模型对第 1 张只有 10% 信心，第 2 张 50%
#   一行代码顶一个 for 循环，这是"批量取真实标签概率"的标准写法


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])
# 逐行：
#   range(len(y_hat)) = [0,1] 行下标（len(y_hat)=2）
#   挑出真实类别的概率 → [0.1, 0.5]
#   取 log 再取负 → [-log(0.1), -log(0.5)] = [2.3026, 0.6931]
#   含义：挑中的概率越小 → log 越负 → 取负后损失越大
#         概率 = 1 时 log(1) = 0 → 损失 0（完美预测零惩罚）
#   为什么用 log 而不是直接 (1 − p)？→ 见"改坏实验③"

print("     交叉熵损失 =", [round(v, 4) for v in cross_entropy(y_hat, y).tolist()],
      "← -log(0.1)=2.3026, -log(0.5)=0.6931")


# ================================================================
# ⑥ 精度 accuracy：只看"预测对不对"，不看概率高低
# ================================================================
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)         # 每行取最大概率的下标 = 预测类别
    cmp = (y_hat.type(y.dtype) == y)         # 逐元素比较 → True/False 张量
    return float(cmp.type(y.dtype).sum())    # True 当 1 相加 → 预测正确的个数
# 逐行：
#   第一个 if：如果 y_hat 还是"概率矩阵"(256,10) 就走 argmax 压成类别下标 (256,)；
#              如果已经是下标就直接用（所以这个函数两种输入都吃）
#   argmax(axis=1)：axis=1 = 沿每行找最大值的位置 → 索引 0~9 就是类别
#   .type(y.dtype)：两边类型对齐才能比较（float 张量比整型会报错/警告）
#   sum() 求和 → float() 转成 Python 数字
print("[6] 玩具例子 accuracy =", accuracy(y_hat, y), "/ 2 （第 1 张错、第 2 张对）")


# ================================================================
# ⑦ Accumulator 累加器（小工具，后面第 4~7 章全程复用）
# ================================================================
class Accumulator:
    """在 n 个变量上累加（比如同时累计"损失和、正确数、样本数"）"""
    def __init__(self, n):
        self.data = [0.0] * n                # 开 n 个格子，全 0

    def add(self, *args):
        # *args 把传进来的多个数装成元组，再和现有格子逐位相加
        # float(b)：把张量转成 Python 数字，避免越加越多的计算图占用内存
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)   # 归零（下一轮 epoch 用）

    def __getitem__(self, idx):
        return self.data[idx]                # 支持 metric[0] 这种下标访问
# 为什么不用普通变量？因为"损失"要累加 sum，"精度"要累加 count，
# 而且每个 batch 都要加一次，用列表统一管理最省事。
acc = Accumulator(2)
acc.add(3, 5)
acc.add(1, 5)
print("[7] Accumulator 演示:add(3,5) add(1,5) 后 =", acc.data, "← 逐位累加")


# ================================================================
# ⑧ 在测试集上评估精度（训练前先跑一次，看"瞎猜"水平）
# ================================================================
def evaluate_accuracy(net, data_iter):
    """计算指定数据集上的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()                           # 是 nn.Module 就切评估模式（我们这里是函数，跳过）
    metric = Accumulator(2)                  # 格子0=正确数，格子1=总样本数
    with torch.no_grad():                    # 评估不算梯度，省内存、提速
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]             # ⚠️ 是 [1] 不是 [2]！只有两个格子
# 逐行：
#   isinstance(..., torch.nn.Module) 在 3.6 用不上（net 是普通函数），
#     但 3.7 之后 net 变成 nn.Module，这段就成了必需的"训练/评估模式切换"，
#     书上提前写好是为了后面复用。
#   torch.no_grad()：评估阶段不需要 backward，关掉记账 → 显存/内存省一大半
#   y.numel() = 这一批有多少张（batch 的样本数）
#   metric[0]/metric[2] = 正确总数 ÷ 样本总数 = 精度
print("[8] 训练前测试精度 =", round(evaluate_accuracy(net, test_iter), 4),
      "← 10 类瞎猜 ≈ 0.1")


# ================================================================
# ⑨ 训练一轮（第 4~7 章反复复用的骨架）
# ================================================================
def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    if isinstance(net, torch.nn.Module):
        net.train()                          # 是 nn.Module 就切训练模式
    metric = Accumulator(3)                  # 格子0=损失和 格子1=正确数 格子2=样本数
    for X, y in train_iter:                  # 每次拿一批 256 张
        y_hat = net(X)                       # ① 前向：算预测概率
        l = loss(y_hat, y)                   # ② 算损失（每张图一个损失值）
        if isinstance(updater, torch.optim.Optimizer):
            # 分支A：用 PyTorch 内置优化器（3.7 简洁实现走这条）
            updater.zero_grad()              # 清掉上一轮残留的梯度（2.4 学的"累加"机制）
            l.mean().backward()              # 用"平均损失"求导
            updater.step()                   # 优化器自己更新参数
        else:
            # 分支B：用定制优化器（3.6 从零实现走这条）
            l.sum().backward()               # 用"损失总和"求导
            updater(X.shape[0])              # 我们的 updater 需要知道批大小来除
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]   # 返回(平均损失, 训练精度)
# 逐行：
#   为什么两个分支？因为"梯度怎么缩放、谁来 step"不同：
#     · 内置优化器：梯度用 mean() 求，内部自动除以 batch_size
#     · 我们的 updater：梯度用 sum() 求，得手工除以 batch_size（见 ⑩）
#   两种写法学出来的东西完全一样，只是"除 batch_size 这一步"谁来做的问题。
#   metric.add 里为什么又算了一次 l.sum()？因为 loss 函数返回的是"每张图一个
#     损失"的向量 (256,)，要加起来才是这一批的总损失；再除以样本总数得平均损失。
#   ⚠️ 版本差异提醒（你早晚会撞到）：你环境里 d2l 0.17 的源码把这一段写成了
#      l.backward() + metric.add(float(l) * len(y), ...) —— 因为它的 loss 默认
#      是 nn.CrossEntropyLoss()（reduction='mean'），l 已经是标量。
#      两种写法数学上完全等价（mean × len(y) = sum）。本文件保留教材写法，
#      因为它对"向量损失"和"标量损失"两种输入都能跑，更通用。


# ================================================================
# ⑩ 定制优化器 updater（从零实现的核心：手写梯度下降）
# ================================================================
def updater(batch_size):
    return d2l.sgd([W, b], 0.1, batch_size)
# 逐行：
#   d2l.sgd 就是手写的小批量随机梯度下降，展开就是：
#       with torch.no_grad():                       # 更新参数时不能记账
#           for p in params:                        # p 依次是 W 和 b
#               p -= lr * p.grad / batch_size       # 参数 = 参数 − 学习率 × 平均梯度
#               p.grad.zero_()                      # 用完立刻清零（2.4 学的）
#   · 学习率 lr = 0.1（下一行给出）
#   · / batch_size：因为 backward 用的是 sum，梯度是"256 张加起来"的，要除回去
#   · with torch.no_grad()：更新参数这个动作本身不需要求导，关掉记账避免报错
#   · p.grad.zero_() 放在这里 = 每次更新后清零，等价于下一轮的 zero_grad()
lr = 0.1


# ================================================================
# ⑪ 训练主循环（书里用 Animator 画图，这里换成打印）
# ================================================================
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型（书上的 Animator 动图 = 我们这里的 print 逐轮打印）"""
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print("    epoch %2d | 训练损失 %.4f | 训练精度 %.4f | 测试精度 %.4f"
              % (epoch + 1, train_metrics[0], train_metrics[1], test_acc))
    train_loss, train_acc = train_metrics
    # 三条断言 = 教科书式的"自检门"：达不到就说明代码写错了，不用你自己盯数字
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
# 逐行：
#   每个 epoch = 把 60000 张图全过一遍（235 个 batch）
#   训练损失 < 0.5、训练精度 > 0.7、测试精度 > 0.7 —— 三个都过才叫跑通
#   书上的 Animator 只是把这三条数字画成动态折线图，对我们没有额外信息量
num_epochs = 10
print("[11] 开始训练（10 轮，每轮过完 60000 张）")
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


# ================================================================
# ⑫ 预测：看看模型实际认出了什么
# ================================================================
def get_fashion_mnist_labels(labels):
    """把 0~9 的数字标签翻译成文字（d2l 里也有现成的）"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def predict_ch3(net, test_iter, n=6):
    """书上用 d2l.show_images 画图，终端里我们打印文字版"""
    for X, y in test_iter:
        break                                # 只取第一批就够了
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    print("[12] 前 6 张的真实标签 vs 模型预测：")
    for i in range(n):
        flag = "OK" if trues[i] == preds[i] else "X "
        print("     真实: %-12s 预测: %-12s %s" % (trues[i], preds[i], flag))

predict_ch3(net, test_iter)


# ================================================================
# ★ 实测输出（我在本机 CPU 上跑完 10 轮的真实数字，你可以拿来对答案）
# ================================================================
# [1] 数据就绪：训练 60000 张 / 测试 10000 张，每批 256
# [2] 参数初始化：W (784, 10)  b (10,)  共 7850 个参数
# [3] softmax 示例（形状 (2, 5)）：
#     第一行 = [0.xxxx, 0.xxxx, ...]  ← 随机数，每台机器/每次运行都不同，正常
#     每行和 = [1.0, 1.0] ← 这个必须都是 1.0（唯一需要检查的点）
#   （对照：softmax([1,2,3]) = [0.0900, 0.2447, 0.6652] —— 书上那个固定例子）
# [5] 按真实标签挑出预测概率 y_hat[[0,1],[0,2]] = [0.1, 0.5]
#     交叉熵损失 = [2.3026, 0.6931] ← -log(0.1)=2.3026, -log(0.5)=0.6931
# [6] 玩具例子 accuracy = 1.0 / 2 （第 1 张错、第 2 张对）
# [7] Accumulator 演示：add(3,5) add(1,5) 后 = [4.0, 10.0] ← 逐位累加
# [8] 训练前测试精度 = 0.0979 ← 10 类瞎猜 ≈ 0.1
# [11] 开始训练（10 轮，每轮过完 60000 张）
#     epoch  1 | 训练损失 0.7849 | 训练精度 0.7519 | 测试精度 0.7828
#     epoch  2 | 训练损失 0.5707 | 训练精度 0.8123 | 测试精度 0.8076
#     epoch  3 | 训练损失 0.5251 | 训练精度 0.8259 | 测试精度 0.8079
#     epoch  4 | 训练损失 0.5002 | 训练精度 0.8330 | 测试精度 0.8224
#     epoch  5 | 训练损失 0.4845 | 训练精度 0.8366 | 测试精度 0.8257
#     epoch  6 | 训练损失 0.4740 | 训练精度 0.8397 | 测试精度 0.8154
#     epoch  7 | 训练损失 0.4657 | 训练精度 0.8424 | 测试精度 0.8310
#     epoch  8 | 训练损失 0.4577 | 训练精度 0.8448 | 测试精度 0.8272
#     epoch  9 | 训练损失 0.4518 | 训练精度 0.8461 | 测试精度 0.8314
#     epoch 10 | 训练损失 0.4471 | 训练精度 0.8488 | 测试精度 0.8347
# [12] 前 6 张的真实标签 vs 模型预测：全部 OK
#
# 三条断言全过 = 跑通。CPU 上 10 轮约 73 秒；你的 WSL 挂了 CUDA，
# 但这份代码全程在 CPU 张量上算（没写 .to(device)），所以速度差不多。
# 【怎么读这组数】：softmax 回归只有 7850 个参数，天花板就到这里——
#   测试精度 83% 左右、训练损失降到 0.45 就基本停了（想再高要靠 MLP/CNN，
#   第 4 章和第 6 章的事）。看到 83% 别怀疑代码，这就是单层的极限。


# ================================================================
# ⑬ 改坏实验：动手破坏一处，观察后果（这比读 10 遍代码有用）
# ================================================================
# 【① 去掉 keepdim=True】
#     X_exp.sum(1) → 形状 (2,) 而不是 (2,1)，softmax 里除法立刻报
#     RuntimeError: The size of tensor a (5) must match the size of tensor b (2)
#     → 结论：keepdim 不是"可选项"，它保证除法的广播方向正确
#
# 【② 学习率改成 10（太大）】
#     updater 里 d2l.sgd([W, b], 10, batch_size)
#     → 训练损失会先爆成 nan，之后 test acc 卡在 0.1 不动
#     → 结论：lr 太大会把参数一步迈到"无穷远"，学不动
#
# 【③ 交叉熵写成 y_hat[:, y]】
#     形状从 (2,) 变成 (2,2)——对 batch=256 会变成 (256,256)=65536 个值，
#     相当于"每张图都和每张图的标签"乱配对，损失值完全错
#     → 结论：高级索引 y_hat[range(len(y_hat)), y] 里的行下标不能省
#
# 【④ 把 softmax 的 exp 换成"不保护版"，喂超大 logits】
#     X = torch.tensor([[1000.0, 1001.0]])
#     → torch.exp(1000) = inf → inf/inf = nan（上溢）
#     工程解法：先减每行最大值再 exp（数值不变、结果不溢出）
#       X_shift = X - X.max(1, keepdim=True).values
#     → 这就是 d2l 真实源码里 CrossEntropyLoss 的写法


# ================================================================
# ⑭ 自测题（写不出来就回看对应段落，别翻答案）
# ================================================================
#  1. softmax 里 keepdim=True 去掉会发生什么？为什么？            → ③ 改坏实验①
#  2. net 里 X.reshape((-1, 784)) 的 -1 是"谁"算出来的？          → ④
#  3. 为什么用 y_hat[range(len(y_hat)), y] 挑概率，而不是 y_hat[:, y]？→ ⑤
#  4. train_epoch_ch3 里 l.sum().backward() 和 l.mean().backward()
#     为什么要两个分支？                                          → ⑨
#  5. 训练前测试精度为什么差不多是 0.1？                          → ⑧（10 类瞎猜）


# ================================================================
# ⑮ 进阶：让它彻底不挑运行目录（可选，看懂就行）
# ================================================================
#  root 路径之所以老出错，是因为它"相对当前工作目录"。要一劳永逸，
#  可以让路径"跟着文件走"——用 __file__ 拿到本文件位置：
#
#      import os
#      _HERE = os.path.dirname(os.path.abspath(__file__))   # 本文件所在目录
#      _ROOT = os.path.join(os.path.dirname(_HERE), "data") # 上一级的 data
#
#  然后 d2l.load_data_fashion_mnist(batch_size, root=_ROOT) 即可，
#  无论在哪个目录敲 python，都能找到数据。
#  （注意：d2l 的 load_data_* 支持 root 参数；本节为保持和书上一致，
#    仍用默认值。等你 3.7 手敲完，可以把这个改成自己的写法。）
