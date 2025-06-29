
import random
import sklearn.metrics
import numpy as np
import matplotlib
from numpy.linalg import norm
import AnomalyDetection
from scipy.linalg import eig
import seaborn as sns
import sklearn.metrics
import  torch
from myHankel import myHankelTensor, myHankelRerverse_Tensor
from myObserver import *
from myKoopman import *
from myDatasets import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler
import warnings
import numpy as np


# 设置全局字体为 SimHei（黑体）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

import numpy as np
import matplotlib.pyplot as plt

def plot_koopman_spectrum(eigvals, modes, title='Koopman 特征值'):
    real_parts = eigvals.real
    imag_parts = eigvals.imag
    mode_strength = np.linalg.norm(modes, axis=0)

    # 打印检查一下范围
    print(f"{title} 实部 ∈ [{real_parts.min():.2e}, {real_parts.max():.2e}], "
          f"虚部 ∈ [{imag_parts.min():.2e}, {imag_parts.max():.2e}]")

    fig, ax = plt.subplots(figsize=(5,5))
    sc = ax.scatter(real_parts, imag_parts,
                    c=mode_strength, cmap='viridis', s=80, edgecolors='k')
    cbar = plt.colorbar(sc, label='模态强度 ||Φ||')
    cbar.formatter.set_powerlimits((-8,-8))
    cbar.update_ticks()

    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    ax.set_title(title)
    ax.grid(True)

    # —— 下面开始设置坐标轴范围 —— #
    # 原始极值
    x_min, x_max = real_parts.min(), real_parts.max()
    y_min, y_max = imag_parts.min(), imag_parts.max()

    # 给 y 轴一个最小“展开量”，防止它和 x 一样平坦
    eps = max(1e-8, (x_max - x_min) * 1e-2)
    if y_max - y_min < eps:
        y_min -= eps
        y_max += eps

    # 给 x 轴也留一点边距
    x_margin = (x_max - x_min) * 0.1
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()




def plot_koopman_spectrum(eigvals, modes, title='Koopman 特征值'):
    """
    绘制 Koopman 模态谱（复平面）

    参数：
        eigvals: ndarray, 特征值（复数），形状 (n,)
        modes: ndarray, Koopman 模态矩阵，形状 (n_states, n_modes)
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # 提取实部、虚部、模态强度
    real_parts = eigvals.real
    imag_parts = eigvals.imag
    mode_strength = np.linalg.norm(modes, axis=0)  # 每个模态的强度（L2范数）

    rmax = max(1.0, np.max(np.abs(real_parts)), np.max(np.abs(imag_parts)))
    margin = 0.1 * rmax
    # 绘制单位圆（表示模长 = 1，振荡保留）
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', linewidth=1)
    # circle = plt.Circle((0, 0), 0.8, color='red', fill=False, linestyle='--', linewidth=1)
    ax.add_artist(circle)

    # 画模态点
    sc = ax.scatter(real_parts, imag_parts, c=mode_strength, cmap='viridis', s=80, edgecolors='k')


    #plt.colorbar(sc, label='模态强度 ||Φ||')
    plt.colorbar(sc, label='Modal Strength ||Φ||')
    ax.set_xlim(-rmax - margin, rmax + margin)
    ax.set_ylim(-rmax - margin, rmax + margin)
    ax.set_xlabel('Re(λ)', fontsize=12)
    ax.set_ylabel('Im(λ)', fontsize=12)
    #ax.set_xlabel('实部 Re(λ)', fontsize=12)
    #ax.set_ylabel('虚部 Im(λ)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True)
    ax.set_aspect('equal')
    plt.tight_layout()
    # plt.show()

def plotheatmap(title1, title2, matrix, dif_):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        cmap="coolwarm",
        cbar=True,
        xticklabels=False,  # 太密，关闭
        yticklabels=False,
        linewidths=0,
        square=True
    )
    plt.title(title1, fontsize=14)
    plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(dif_, annot=False, fmt=".3f", cmap="coolwarm", cbar=True)
    plt.title(title2)
    plt.xlabel("列")
    plt.ylabel("行")
    plt.tight_layout()
    # plt.show()

def kl_divergence(p, q):
    # 归一化
    p_normalized = p / np.sum(p)
    q_normalized = q / np.sum(q)
    # 计算KL散度（避免log(0)）
    mask = p_normalized > 0
    return np.sum(p_normalized[mask] * np.log(p_normalized[mask] / q_normalized[mask]))

def cosine_similarity(K1, K2):
    flat1, flat2 = K1.ravel(), K2.ravel()
    return np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2))


def manhattan_distance(matrix1, matrix2):
    """
    计算两个矩阵之间的曼哈顿距离（城市街区距离）
    公式：d = Σ|matrix1[i,j] - matrix2[i,j]|
    """
    # 确保矩阵形状一致
    if matrix1.shape != matrix2.shape:
        raise ValueError(f"矩阵形状不一致：{matrix1.shape} vs {matrix2.shape}")

    # 计算每个元素差值的绝对值之和
    return np.sum(np.abs(matrix1 - matrix2))

def corrected_cosine_similarity(vec_i, vec_j):
    """
    修正余弦相似度（先减去各自均值再计算余弦）
    """
    vec_i = np.asarray(vec_i, dtype=float)
    vec_j = np.asarray(vec_j, dtype=float)
    if vec_i.shape != vec_j.shape:
        raise ValueError(f"长度不一致：{vec_i.shape} vs {vec_j.shape}")
    dev_i = vec_i - vec_i.mean()
    dev_j = vec_j - vec_j.mean()
    num   = np.dot(dev_i, dev_j)
    denom = np.linalg.norm(dev_i) * np.linalg.norm(dev_j)
    return 0.0 if denom == 0 else num / denom

def compute_d_pred(Y_feat_np, Kx_np):
    """
    Y_feat_np: numpy array of shape (T', d)
    Kx_np: numpy array of shape (d, d)
    g(y)=identity on already-embedded features
    """
    Y = torch.from_numpy(Y_feat_np).float()
    Kx = torch.from_numpy(Kx_np).float()
    G = Y                        # (T', d)
    G_prev = G[:-1]              # (T'-1, d)
    G_true = G[1:]               # (T'-1, d)
    G_pred = (Kx @ G_prev.t()).t()  # (T'-1, d)
    se = (G_true - G_pred).pow(2).sum()
    Tprime = Y.shape[0]
    return torch.sqrt(se).item() / Tprime

num = 20
Anom_num = 21
dict_data = {
    0: "./Koopman_matrix/CAFUC-abnormal.npy",
    1: './Koopman_matrix/RosslerDatasets.npy',
    2: './Koopman_matrix/RosslerDatasets-1000-(-4,4).npy',
    3: './Koopman_matrix/RosslerDatasets-2000-(-4,4).npy',
    4: './Koopman_matrix/RosslerDatasets-1000.npy',
    5: './Koopman_matrix/RosslerDatasets-2000.npy',
    6: './Koopman_matrix/RosslerDatasets-10000-11000.npy',
    7: './Koopman_matrix/RosslerDatasets-10000-12000.npy',
    8: './Koopman_matrix/RosslerDatasets-10000-11000-(-4,4).npy',
    9: './Koopman_matrix/RosslerDatasets-10000-12000-(-4,4).npy',
    10: './Koopman_matrix/RosslerDatasets-15000-17000.npy',
    11: './Koopman_matrix/LorenzDatasets1-2.npy',
    12: './Koopman_matrix/LorenzAnmoDatasets-1000-(-4,4).npy',
    13: './Koopman_matrix/LorenzAnmoDatasets-2000-(-4,4).npy',
    14: './Koopman_matrix/LorenzAnmoDatasets-1000.npy',
    15: './Koopman_matrix/LorenzAnmoDatasets-2000.npy',
    16: './Koopman_matrix/LorenzAnmoDatasets-10000-11000.npy',
    17: './Koopman_matrix/LorenzAnmoDatasets-10000-12000.npy',
    18: './Koopman_matrix/LorenzAnmoDatasets-10000-11000-(-4,4).npy',
    19: './Koopman_matrix/LorenzAnmoDatasets-10000-12000-(-4,4).npy',
    20: "./Koopman_matrix/RectangeDatasets-0.npy",
    21: "./Koopman_matrix/CAFUC-abnormal2.npy",
    22: "./Koopman_matrix/EGG-normal.npy",
    23: "./Koopman_matrix/EGG-abnormal-real.npy"
}

Y_np = np.load(r"D:\Lujing\EKNO20231212 (2)\LorenzDatasets-anomaly.npy")
epsilon = 2
d_embed  = 4
H_Y_tf = myHankelTensor(Y_np, epsilon, d_embed, name='Y_embed')
H_Y_np = H_Y_tf.numpy().reshape(H_Y_tf.shape[0], -1)


matrix = np.load(dict_data[num])  #原始算子矩阵
data_list = []
for num_ in dict_data:
    array = np.load(dict_data[num_])
    data_list.append(array)

array_anom = np.load(dict_data[Anom_num]) # 异常矩阵
eigvals_normal, W_normal = eig(matrix)   #原始算子矩阵的特征值和模态
eigvals_anom, W_anom = eig(array_anom)   #异常算子矩阵的特征值和模态
dif = W_normal - W_anom
dif_ = matrix - array_anom
print(dif_.shape)
matrixname = {dict_data[num].split("/")[2].split(".")[0][:16]}
# np.save(f'./Koopman_matrix/{matrixname}-LorenzAnomDatasets_dif_noeig.npy', dif_)

#对矩阵进行处理
# rows, cols = dif_.shape
# for i in range(rows):
#     for j in range(cols):
#         value = dif_[i, j]
#         if value < 0.1 and value > -0.1:
#             dif_[i, j] = 0
res = norm(dif_, ord='fro')
hadamard_norm = np.max(np.abs(dif_))
print("F范数：", res)
# plotheatmap(f'{dict_data[7].split("/")[2].split(".")[0]}-ano', 'dif', array_anom, dif_)
plot_koopman_spectrum(eigvals_normal,W_normal,title='Normal Koopman Eigenvalue')
plot_koopman_spectrum(eigvals_anom,W_anom,title='Abnormal Koopman Eigenvalue')
# print(matrix)
# print("KL散度:", kl_divergence(matrix, array_anom))
print("余弦相似度:", cosine_similarity(matrix, array_anom))



sim = corrected_cosine_similarity(matrix, array_anom)

if isinstance(sim, np.ndarray):
    try:
        sim = sim.item()
    except Exception:
        # 如果不是 0-d array，就取第一个元素
        sim = float(sim.flat[0])

print(f"修正余弦相似度: {sim:.4f}")

d_pred = compute_d_pred(H_Y_np, matrix)
print(f" d_pred = {d_pred:.3e}")

# 1. 均方根误差（RMSE）
diff = matrix - array_anom
rmse = np.sqrt(np.mean(diff**2))
print(f"RMSE: {rmse:.3e}")

# 2. 绝对误差之和（SAE）
sae = np.sum(np.abs(diff))
print(f"SAE: {sae:.3e}")

manhattan = manhattan_distance(matrix, array_anom)
print(f"曼哈顿距离: {manhattan:.3e}")

# 4. 皮尔逊相关系数（对比两个矩阵摊平成向量后的相关性）
flat1 = matrix.ravel()
flat2 = array_anom.ravel()
pearson = np.corrcoef(flat1, flat2)[0,1]
print(f"Pearson Correlation: {pearson:.4f}")

# 5. 最大李亚普诺夫指数（用 λ 的连续谱估计）
#    Re(log λ) / Δt 近似对应连续时间增长率，取最大值作为 λ_max。
#    这里假设 Δt = 1；如果你的采样间隔不是 1，请替换 dt。
def filter_eigvals(eigvals, eps=1e-8):
    """严格筛选有效特征值（排除数值噪声）"""
    mask = np.logical_and(
        np.abs(eigvals) > eps,       # 排除接近0的特征值
        ~np.isclose(eigvals, 0+0j)   # 排除纯零（数值误差导致）
    )
    return eigvals[mask]
# def analyze_eigval_modulus(eigvals, title):
#     modulus = np.abs(eigvals)
#     print(f"=== {title} 特征值分析 ===")
#     print(f"模长范围: [{modulus.min():.4f}, {modulus.max():.4f}]")
#     print(f"模长>1的比例: {np.mean(modulus > 1):.2%}")
#     print(f"模长=1的比例: {np.mean(np.isclose(modulus, 1, atol=0.01)):.2%}")
#     print(f"模长<1的比例: {np.mean(modulus < 1):.2%}")
#
# analyze_eigval_modulus(eigvals_normal, "正常算子（矩形数据集）")
# analyze_eigval_modulus(eigvals_anom, "异常算子")
# 重新筛选特征值
normal_eigvals_filtered = filter_eigvals(eigvals_normal)
abnormal_eigvals_filtered = filter_eigvals(eigvals_anom)
#
# # 统计筛选前后的特征值数量
# print(f"正常算子原始特征值数量: {len(eigvals_normal)}, 筛选后: {len(normal_eigvals_filtered)}")
# print(f"异常算子原始特征值数量: {len(eigvals_anom)}, 筛选后: {len(abnormal_eigvals_filtered)}")

import numpy as np

# 假设 eigvals_abn 是你的异常算子特征值 ndarray
eigvals_nom = np.array(eigvals_normal)
eigvals_abn = np.array(eigvals_anom)  # 来自你的 np.linalg.eig 调用

# 1. 设定一个很小的阈值，过滤零值或数值噪声
eps = 1e-12
mask = np.abs(eigvals_abn) > eps
filtered = eigvals_abn[mask]

mask = np.abs(eigvals_nom) > eps
filtered1 = eigvals_nom[mask]

dif1 =abs(eigvals_nom - eigvals_abn )
# 2. 计算增长率 Re(log λ)/dt
dt = 0.01  # 你的采样间隔
growths = np.real(np.log(dif1)) / dt

# 3. 取最大值
if growths.size > 0:
    lyap_abn = np.max(growths)
    print(f"Max Lyapunov Exponent (Abnormal): {lyap_abn:.3e}")
else:
    print("No valid eigenvalues to compute Lyapunov exponent.")


plt.show()



