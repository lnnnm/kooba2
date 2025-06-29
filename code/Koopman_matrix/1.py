import numpy as np
import matplotlib.pyplot as plt

# 1. 读取 Koopman 算子（确保文件路径正确）
K = np.load("RectangeDatasets1-20.npy")

# 2. 特征值分解
eigvals, eigvecs = np.linalg.eig(K)

# 3. 每个模态的能量（模态向量的 L2 范数平方）
mode_energy = np.linalg.norm(eigvecs, axis=0) ** 2

# 4. 从大到小排序能量
sorted_indices = np.argsort(-mode_energy)
sorted_energy = mode_energy[sorted_indices]

# 5. 可视化前30个模态能量
plt.figure(figsize=(10, 4))
plt.plot(range(2, 6), sorted_energy[:4], 'bo-', linewidth=2, label="Mode Energy")
plt.title("Koopman Mode Energy Spectrum (Top 30)", fontsize=14)
plt.xlabel("Mode Index (sorted)", fontsize=12)
plt.ylabel("Energy", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
