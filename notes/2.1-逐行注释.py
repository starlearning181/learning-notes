# ============================================================
# d2l 2.1 数据操作 —— 逐行注释版
# 读法：一行代码 + 一行注释 + 一行"会输出什么"
# 有疑问就改一个地方再跑，看结果变了没有（改坏它 = 学得最快）
# ============================================================

# 【导入】import = 导入；torch = PyTorch 工具箱的名字。
# 意思：把 PyTorch 这个工具箱搬进当前文件，后面才能用它的工具。
# 以后每个深度学习文件的第一行几乎都是它。
import torch

# 【造数据】arange = array + range（一串连续的数）。
# torch.arange(12) = 让工具箱造出 0~11 共 12 个数，然后把结果"还回来"。
# 前面写 x = ，意思是"用一个叫 x 的名字接住它"（= 是装进抽屉，不是数学等号）。
x = torch.arange(12)

# 【打印】print = 把东西显示到终端。
print(x)          # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
                  # tensor = PyTorch 的张量，可以理解成"装数字的容器"

# 【问形状】shape = 形状，回答"几行几列"。
print(x.shape)    # torch.Size([12]) —— 括号里只有一个数 = 一排 12 个（1 维）

# 【改摆法】reshape = re + shape（重新摆形状）。
# 12 个数从"一排"摆成"3 行 4 列"，数字一个没多、一个没少。
# 注意：x 本身没变，结果装进了新名字 X（大写）。Python 严格区分大小写！
X = x.reshape(3, 4)
print(X)          # [[ 0,  1,  2,  3],
                  #  [ 4,  5,  6,  7],
                  #  [ 8,  9, 10, 11]]

# 【全 0 张量】zeros(2, 3, 4) = 造 2 层、每层 3 行 4 列，全是 0。
print(torch.zeros(2, 3, 4))   # 形状 (2,3,4)，3 维张量

# 【随机数】randn = random normal（标准正态分布随机数）。
print(torch.randn(3, 4))      # 每次跑都不一样，正常

# 【全加】sum() 不带参数 = 把所有数加起来。
print(X.sum())    # 0+1+2+...+11 = 66

# 【自己加自己】对应位置相加，形状不变。
print(X + X)      # [[0,2,4,6],[8,10,12,14],[16,18,20,22]]

# ------------------------------------------------------------
# 【广播】下面这条曾经报错，先看清"错的怎么写"：
# print(X + x)   # ❌ RuntimeError: The size of tensor a (4) must match the size of tensor b (12)
#   X 是 (3,4)，x 是 (12,) —— 从最右边的维度开始对齐：4 和 12 对不上 → 拒绝相加
# 正确写法：让另一个张量"最后一维"等于 X 的列数（4）：
y = torch.arange(4)   # 一排 4 个数
print(X + y)          # ✅ y 被自动复制 3 份，加到每一行上
                      # [[0,2,4,6],[4,6,8,10],[8,10,12,14]]
# 口诀：从最右维开始对齐，每一维要么相等、要么有一方是 1、要么缺失
# ------------------------------------------------------------

# 【按轴求和】dim=0 是竖轴（跨行），dim=1 是横轴（跨列）。
# 被压掉的那根轴会消失。
print(X.sum(0))   # 竖着加：每一列的 3 个数相加 → 4 个数：[12, 15, 18, 21]
print(X.sum(1))   # 横着加：每一行的 4 个数相加 → 3 个数：[6, 22, 38]

# ------------------------------------------------------------
# 【拼接】cat = concatenate（拼接），dim 决定往哪个方向接。
# 规则：除了被拼接的那一维，其他维度必须完全相同（这里不享受广播待遇）。
print(torch.cat((X, X), dim=0))   # 上下摞 → 6 行 4 列
print(torch.cat((X, X), dim=1))   # 左右拼 → 3 行 8 列

# 想把 y（一排 4 个数）也拼进来？先把它升成二维：
y2 = y.reshape(1, 4)              # (4,) → (1,4)：从"纸条"变成"1 行 4 列的表格"
print(torch.cat((X, y2), dim=0))  # ✅ 现在两个都是 2 维 → 4 行 4 列

# 两个踩过的坑，留着当教材：
# print(torch.cat((X, y), dim=0))  # ❌ Tensors must have same number of dimensions: got 2 and 1
# print(torch.cat(X, y, dim=0))    # ❌ TypeError: argument 'tensors' must be tuple of Tensors
#   → cat 的第一个参数要"一叠"张量，所以要写成双括号 (X, y)
# ------------------------------------------------------------

# 【逐元素比较】得到 True/False 组成的张量，形状跟 X 一样。
print(X == y)     # [[True, True, True, True],
                  #  [False, False, False, False],
                  #  [False, False, False, False]]

# 【和 numpy 互转】numpy 是另一个"数字表格"库，pandas 和很多老代码都靠它。
A = X.numpy()               # 张量 → numpy 数组
B = torch.from_numpy(A)     # numpy 数组 → 张量（from = 从…来）
print(type(A), type(B))     # <class 'numpy.ndarray'> <class 'torch.Tensor'>

# 【标量转换】item() = 把里面那一个数取出来，变成普通 Python 数字。
print(X.sum(), X.sum().item())   # tensor(66) 66 —— 左边是张量，右边是纯数字
