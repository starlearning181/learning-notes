import torch

#第一部分 原始输出
o=torch.tensor([[1.0,2.0,3.0],[3.0,2.0,1.0]])
print("原始输出o(2个样本*3个类别):\n",o)

#第二部分 softmax 三步：取指数 → 求和 → 相除  公式：softmax(o)ᵢ = exp(oᵢ) / Σⱼ exp(oⱼ)
exp_o=torch.exp(o)
print("\n① 取指数 exp(o):\n", exp_o)
partition=exp_o.sum(axis=1,keepdim=True)
print("\n② 每行求和（keepdims=True 保住 (2,1) 形状）:\n", partition)
y_hat=exp_o/partition
print("\n③ 相除 = softmax 输出:\n", y_hat)
print("   每行之和:", y_hat.sum(axis=1).tolist(), " <-- 全 1 才叫概率分布")
print("   最小值  :", float(y_hat.min()), " <-- 非负")

#第三部分 交叉熵
y=torch.tensor([2,0])
print("\n真实标签 y =", y.tolist(), "（整数编号）")
pick = y_hat[torch.arange(2), y]
print("   正确类的预测概率:", pick.tolist())
ce = -torch.log(pick)
print("   交叉熵 -log(p)   :", ce.tolist())
print("   该批平均 loss    :", float(ce.mean()))

#第四部分 独热编码
onehot=torch.nn.functional.one_hot(y,num_classes=3)
print("\n⑤ 独热编码 one_hot(y, num_classes=3):\n", onehot.float())
