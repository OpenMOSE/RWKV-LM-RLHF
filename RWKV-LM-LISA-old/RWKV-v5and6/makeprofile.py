import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--num_layer", default=24, type=int)
parser.add_argument("--startvalue", default=1, type=float)
parser.add_argument("--endvalue", default=1, type=float)
parser.add_argument("--centroidpos", default=12, type=int)
parser.add_argument("--centroidvalue", default=0.5, type=float)
parser.add_argument("--outputfile", default="custom_lp_profile.csv", type=str)

args = parser.parse_args()


# パラメータ設定
num_points = args.num_layer  # 点の数
start_value = args.startvalue  # 開始点の値
end_value = args.endvalue   # 終了点の値
centroid_pos = args.centroidpos  # 重心の位置（1から32の間の値）
centroid_value = args.centroidvalue  # 重心の値

# 開始点、重心、終了点を用いて基本的なデータポイントを作成
x_points = np.array([1, centroid_pos, num_points])
y_points = np.array([start_value, centroid_value, end_value])

# スプライン補完
f = interp1d(x_points, y_points, kind='quadratic')  # 二次のスプライン補完を使用

# 補完を行う点
x_new = np.linspace(1, num_points, num_points)

# 補完されたデータを取得
y_new = f(x_new)

# 結果をプロット
plt.figure(figsize=(8, 4))
plt.plot(x_points, y_points, 'o', label='Original data points')
plt.plot(x_new, y_new, '-', label='Interpolated curve')
plt.xlabel('Position')
plt.ylabel('Value')
plt.title('Curve Interpolation with Centroid')
plt.legend()
plt.grid(True)
#plt.show()

# CSVファイルに保存
data_to_save = np.column_stack((x_new, y_new))
np.savetxt(args.outputfile, data_to_save, delimiter=",", header="Position,Value", comments="")

# 保存完了のメッセージを表示
print(f'データがCSVファイルに保存されました: {args.outputfile}')