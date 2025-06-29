import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyedflib  # 用于读取EDF文件
import pandas as pd

# 1. 读取EDF文件并提取信号
def read_edf_file(file_path, channel_indices=[0, 1, 2]):
    """
    读取EDF文件并提取指定通道的信号

    参数:
    - file_path: EDF文件路径
    - channel_indices: 要提取的通道索引列表，默认为前三个通道

    返回:
    - 包含指定通道信号的NumPy数组
    """
    try:
        with pyedflib.EdfReader(file_path) as f:
            n_channels = f.signals_in_file
            signal_labels = f.getSignalLabels()

            # 检查请求的通道索引是否有效
            for idx in channel_indices:
                if idx < 0 or idx >= n_channels:
                    raise ValueError(f"通道索引 {idx} 超出范围 (0-{n_channels - 1})")

            # 提取指定通道的信号
            signals = np.zeros((f.getNSamples()[0], len(channel_indices)))
            for i, idx in enumerate(channel_indices):
                signals[:, i] = f.readSignal(idx)

            print(f"成功读取EDF文件: {file_path}")
            print(f"提取的通道: {[signal_labels[idx] for idx in channel_indices]}")
            return signals

    except Exception as e:
        print(f"读取EDF文件时出错: {e}")
        return None


# 读取EDF文件（替换为你的文件路径）
#coords1 = read_edf_file(r'D:\Lujing\EKNO20231212 (2)\dataRaw\EGGDatasets\chb24_02.edf')

#anom_coords = read_edf_file(r'D:\Lujing\EKNO20231212 (2)\dataRaw\EGGDatasets\chb24_01.edf')
#coords = pd.read_csv(r"D:\Lujing\EKNO20231212 (2)\output\CAFUC-abnormal2\bigdata_test.csv")
#coords1 = pd.read_csv(r"D:\Lujing\EKNO20231212 (2)\output\RectangeDatasets\bigdata_test.csv")
coords1 = pd.read_csv(r"D:\Lujing\EKNO20231212 (2)\output\CAFUC-abnormal3\bigdata_test.csv")

print("coords1 形状:", coords1.shape)  # 确认是否为 (2000, 3)

# 2. 转换为NumPy数组（若需要）
coords1 = coords1.to_numpy()  # 转换为 (2000, 3) 的数组
#coords = coords.to_numpy()  # 转换为 (2000, 3) 的数组
# 筛掉第二列 < -60000 的点
mask1 = coords1[:, 1] >= -60000
coords1 = coords1[mask1]

anom_coords = coords1.copy()
# 第一维同时满足：>= -60000 且 <= 10000
mask2 = (anom_coords[:, 0] >= -60000) &  (anom_coords[:, 0] < 100000)

# # 第二维同时满足：>= -60000 且 <= 10000
mask3 = (anom_coords[:, 1] >= -60000) &  (anom_coords[:, 1] < 100000)

# # 使用逻辑与(&)组合两个掩码
combined_mask = mask2 & mask3
#
# # 应用复合掩码筛选数据
anom_coords = anom_coords[combined_mask]

coords1 = coords1[combined_mask]
# 2. 生成带异常的副本

start = 3067
end = 3167
rows = 100# 在前 1000 个点注入噪声
cols = coords1.shape[1]
noise = (np.random.rand(rows, cols) * 8 - 4)
anom_coords[start:end] += noise

# 3. 绘图对比
fig = plt.figure(figsize=(12, 5))

# ——— 左：原始轨迹 ——————————————————
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(coords1[:, 0], coords1[:, 1], coords1[:, 2],
         lw=0.7, color='blue', label='Original')
ax1.scatter(coords1[0,0], coords1[0,1], coords1[0,2],
            c='green', s=80, marker='o', label='Start')
ax1.scatter(coords1[-1,0],coords1[-1,1],coords1[-1,2],
            c='red',   s=80, marker='X', label='End')
ax1.set_title("Normal Trajectory")
ax1.set_xlabel("Dim 1"); ax1.set_ylabel("Dim 2"); ax1.set_zlabel("Dim 3")
ax1.legend()
ax1.set_box_aspect((1,1,1))

# ——— 右：带异常的轨迹 —————————————————
ax2 = fig.add_subplot(1, 2, 2, projection='3d')



# 恢复段
# ax2.plot(anom_coords[:, 0], anom_coords[:, 1], anom_coords[:, 2],
#          lw=0.7,color='blue')
# ax2.plot(anom_coords[:, 0], anom_coords[:, 1], anom_coords[:, 2],
#          lw=0.7,color='blue')

ax2.plot(anom_coords[0:start, 0], anom_coords[0:start, 1], anom_coords[0:start, 2],
         lw=0.7, color='blue')
ax2.plot(anom_coords[end:, 0], anom_coords[end:, 1], anom_coords[end:, 2],
         lw=0.7, color='blue', label='Normal')
# 异常段高亮
# ax2.plot(anom_coords[:, 0], anom_coords[:, 1], anom_coords[:, 2],
#          lw=0.7)
ax2.plot(anom_coords[start:end, 0], anom_coords[start:end, 1], anom_coords[start:end, 2],
         lw=0.7, color='red', label='Anomal')
ax2.scatter(anom_coords[0,0], anom_coords[0,1], anom_coords[0,2],
            c='green', s=80, marker='o', label='Start')
ax2.scatter(anom_coords[-1,0],anom_coords[-1,1],anom_coords[-1,2],
            c='black', s=80, marker='X', label='End')
ax2.set_title("Anomal Trajectory")
ax2.set_xlabel("Dim 1"); ax2.set_ylabel("Dim 2"); ax2.set_zlabel("Dim 3")
ax2.legend()
ax2.set_box_aspect((1,1,1))

plt.tight_layout()
plt.show()
