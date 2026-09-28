# ============================================================
# d2l 2.2 数据预处理（pandas）—— 逐行注释版
# 这一版和书上顺序完全一致：先独热编码，再填缺失值
# 运行：python notes/2.2-逐行注释.py
# ============================================================

# 【导入 pandas】处理"表格"（内存里的 Excel），惯例简写成 pd。
import pandas as pd
# 【导入 torch】负责把表格变成神经网络能吃的张量。
import torch

# 【造一张"脏表"】字典 → 表：键是列名，值是这一列的数据。
# None 在 pandas 里会自动变成 NaN（Not a Number）= 这里缺数据。
# 真实数据（CSV / Excel）永远有缺失值和文字列，这一节就是教你把它们收拾干净。
data = {"NumRooms": [2, 4, None, None],
        "Alley": ["Pave", None, None, None],
        "Price": [127500, 106000, 178000, 140000]}

# 【字典 → 表格】DataFrame = pandas 的表格类型。
data = pd.DataFrame(data)

# 【切表】iloc = 按位置取数据（i = integer，整数下标）。
#   [:, 0:2] = 所有行、第 0 到第 1 列  → 特征表 inputs（模型看的东西）
#   [:, 2]   = 所有行、第 2 列        → 标签 targets（模型要猜的答案）
# 特征和标签必须分开，这是机器学习的铁律：X 进模型，y 用来对答案。
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]

print("=== ① 原始 inputs（有 NaN） ===")
print(inputs)
#    NumRooms Alley
# 0       2.0  Pave
# 1       4.0   NaN
# 2       NaN   NaN
# 3       NaN   NaN

# 【独热编码】get_dummies = 把"文字列"拆成若干列 0/1。
#   dummy_na=True → 额外给"缺失值"单独开一列（Alley_nan），
#   因为"这条巷子没记录"本身也是一条信息，不该丢。
#   结果：Alley 列消失，多出 Alley_Pave、Alley_nan 两列（True/False）。
inputs = pd.get_dummies(inputs, dummy_na=True)

print("=== ② get_dummies 之后（文字变 0/1） ===")
print(inputs)
#    NumRooms  Alley_Pave  Alley_nan
# 0       2.0        True      False
# 1       4.0       False       True
# 2       NaN       False       True
# 3       NaN       False       True

# 【填缺失值】fillna = fill N/A（填掉空值）；mean() = 求每一列的平均值。
# 整句意思：哪一格空着，就用"它所在那一列的均值"补上。
# 为什么用均值？不填就只能删掉整行，而数据本来就少，删了就没了。
inputs = inputs.fillna(inputs.mean())

print("=== ③ fillna 之后（空值被均值替换） ===")
print(inputs)
# NumRooms 的均值 = (2+4)/2 = 3 → 两个 NaN 变成 3.0
# Alley_Pave / Alley_nan 这两列没有空值，所以保持不变（新版 pandas）
# 书上显示成 0.25 / 0.75，是旧版 pandas 的显示差异，本质一样

# 【表格 → 张量】这一步才是整节的目的：
#   to_numpy() 先去 pandas 化，变成 numpy 数组；
#   dtype=float 强制转成小数 —— 神经网络只吃数字，不吃文字和 True/False；
#   torch.tensor(...) 再包成张量，就能喂给模型了。
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))

print("=== ④ 转成张量 ===")
print(X)   # 4 行 3 列的特征；True→1.0，False→0.0
print(y)   # tensor([127500., 106000., 178000., 140000.]) 要预测的价格

# ------------------------------------------------------------
# 【附：我最早给的那一版为什么跑不动】
# 那一版里写了 targets = pd.DataFrame(data) ——把整张表（含文字列 Alley）
# 当成标签 y。于是最后 targets.to_numpy(dtype=float) 撞上 "Pave"，
# 报 ValueError: could not convert string to float: 'Pave'
# 正确做法就是上面这句：iloc 切出 Price 那一列当 y。
# ------------------------------------------------------------
