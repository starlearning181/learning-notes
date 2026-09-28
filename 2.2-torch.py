import pandas as pd
import torch
data = {"NumRooms": [2, 4, None, None],
        "Alley": ["Pave", None, None, None],
        "Price": [127500, 106000, 178000, 140000]}
data = pd.DataFrame(data)
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]

print("=== ① 原始 inputs ===")
print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True)
print("=== ② get_dummies 之后 ===")
print(inputs)
inputs = inputs.fillna(inputs.mean())
print("=== ③ fillna 之后 ===")
print(inputs)
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
print("=== ④ 张量 ===")
print(X)
print(y)
