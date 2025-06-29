import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置空间和时间轴
time = np.linspace(0, 600, 200)
space = np.linspace(0, 150, 100)
T, S = np.meshgrid(time, space)

# 定义一些伪造的 Koopman 模态函数（只是为了模拟图像结构）
def koopman_mode_2(S, T):
    return 10 * np.sin(0.1 * T) * np.cos(0.05 * S) + 5 * np.random.randn(*S.shape)

def koopman_mode_33(S, T):
    return 20 * np.exp(-0.002 * T) * np.sin(0.5 * S)

def koopman_mode_34(S, T):
    return 15 * np.exp(-0.003 * T) * np.cos(0.4 * S)

def koopman_mode_121(S, T):
    return 5 * np.exp(-0.01 * T) * np.sin(1.0 * S)

def koopman_mode_120(S, T):
    return 10 * np.exp(-0.02 * T) * np.cos(1.2 * S)

def koopman_mode_149(S, T):
    return 5 * np.exp(-0.03 * T) * np.sin(1.5 * S + T * 0.01)

# 模态函数列表
modes = [
    (koopman_mode_2, "Koopman Mode2"),
    (koopman_mode_33, "Koopman Mode33"),
    (koopman_mode_34, "Koopman Mode34"),
    (koopman_mode_121, "Koopman Mode121"),
    (koopman_mode_120, "Koopman Mode120"),
    (koopman_mode_149, "Koopman Mode149")
]

# 创建 3x2 的子图
fig = plt.figure(figsize=(18, 10))
for i, (mode_func, title) in enumerate(modes, 1):
    ax = fig.add_subplot(2, 3, i, projection='3d')
    Z = mode_func(S, T)
    surf = ax.plot_surface(T, S, Z, cmap='viridis', linewidth=0, antialiased=False)
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Monitoring Station')
    ax.set_zlabel('Z')

plt.tight_layout()
plt.show()
