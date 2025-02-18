import torch

# A.pthとB.pthのモデルをロード
model_A = torch.load("myfolder/arwkv-cje5-init.pth")
model_B = torch.load("myfolder/Outputs/arwkv5cjetest/rwkv-0.pth")

# 差分合計を計算し表示
diff_sum = {}
total_diff = 0.0

for name, param_A in model_A.items():
    if name in model_B:
        param_B = model_B[name]
        diff = torch.abs(param_A - param_B).sum().item()
        diff_sum[name] = diff
        total_diff += diff

# 各パラメータの差分合計を表示
for name, diff in diff_sum.items():
    print(f"{name}: {diff}")

# 全体の差分合計を表示
print(f"Total difference: {total_diff}")