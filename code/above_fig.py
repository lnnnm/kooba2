import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体字体
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import norm
from scipy.linalg import eig
import torch
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from myHankel import myHankelTensor, myHankelRerverse_Tensor
import torch
import numpy as np
import scipy.linalg
import tensorflow as tf
from sklearn.decomposition import PCA


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
    ax.add_artist(circle)
    # 画模态点
    sc = ax.scatter(real_parts, imag_parts, c=mode_strength, cmap='viridis', s=80, edgecolors='k')
    plt.colorbar(sc, label='Modal Strength ||Φ||')
    ax.set_xlim(-rmax - margin, rmax + margin)
    ax.set_ylim(-rmax - margin, rmax + margin)
    ax.set_xlabel('Re(λ)', fontsize=12)
    ax.set_ylabel('Im(λ)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True)
    ax.set_aspect('equal')
    plt.tight_layout()

def plotheatmap(title1, title2, matrix, dif_):
    """
    绘制两个矩阵及差异的热力图
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="coolwarm", cbar=True, xticklabels=False, yticklabels=False, linewidths=0, square=True)
    plt.title(title1, fontsize=14)
    plt.tight_layout()
    plt.figure(figsize=(10, 8))
    sns.heatmap(dif_, annot=False, fmt=".3f", cmap="coolwarm", cbar=True)
    plt.title(title2)
    plt.xlabel("列")
    plt.ylabel("行")
    plt.tight_layout()

def kl_divergence(p, q):
    """
    计算两个分布的 KL 散度
    """
    p_normalized = p / np.sum(p)
    q_normalized = q / np.sum(q)
    mask = p_normalized > 0
    return np.sum(p_normalized[mask] * np.log(p_normalized[mask] / q_normalized[mask]))

def cosine_similarity(K1, K2):
    """
    计算两个矩阵的余弦相似度
    """
    flat1 = K1.ravel()
    flat2 = K2.ravel()
    return np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2))

def manhattan_distance(matrix1, matrix2):
    """
    计算两个矩阵之间的曼哈顿距离（城市街区距离）
    """
    if matrix1.shape != matrix2.shape:
        raise ValueError(f"矩阵形状不一致：{matrix1.shape} vs {matrix2.shape}")
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
    num = np.dot(dev_i, dev_j)
    denom = np.linalg.norm(dev_i) * np.linalg.norm(dev_j)
    return 0.0 if denom == 0 else num / denom

def compute_d_pred(Y_feat_np, Kx_np):
    """
    计算基于 Koopman 算子的预测误差（均方根）
    参数:
        Y_feat_np: numpy array (T', d)
        Kx_np: Koopman 算子矩阵 (d, d)
    """
    Y = torch.from_numpy(Y_feat_np).float()
    Kx = torch.from_numpy(Kx_np).float()
    G = Y
    G_prev = G[:-1]
    G_true = G[1:]
    G_pred = (Kx @ G_prev.t()).t()
    se = (G_true - G_pred).pow(2).sum()
    Tprime = Y.shape[0]
    return torch.sqrt(se).item() / Tprime

def load_dmd(path, r=200):
    """
    加载 .npy 快照矩阵并执行 DMD 分析
    返回:
      period      – 模态周期 (2π/|Im(log λ)|)
      amplitude   – 模态振幅
      growth_rate – 模态增长率 (Re(log λ))
    """
    Data = np.load(path)
    X1, X2 = Data[:, :-1], Data[:, 1:]
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    r_cut = min(X1.shape[1], r)
    Ur, Vr = U[:, :r_cut], Vh.conj().T[:, :r_cut]
    S_inv = np.diag(1.0 / S[:r_cut])
    Atilde = Ur.T @ X2 @ Vr @ S_inv
    eigs, W = np.linalg.eig(Atilde)
    log_e = np.log(eigs)
    with np.errstate(divide='ignore', invalid='ignore'):
        period = 2 * np.pi / np.abs(log_e.imag)
    growth_rate = log_e.real
    Phi = X2 @ Vr @ S_inv @ W
    amplitude = np.linalg.norm(Phi.real, axis=0)
    mask = np.isfinite(period)
    return period[mask], amplitude[mask], growth_rate[mask]

def compute_eigs(data_path, pca_dims):
    """
    加载 npy 数据 → Z-score 标准化 → PCA → DMD → 返回 eigs 和 log(eigs).
    """
    data = np.load(data_path)
    Z = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    Z = PCA(n_components=pca_dims).fit_transform(Z)
    X, X2 = Z.T[:, :-1], Z.T[:, 1:]
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    Ur, Vr = U[:, :len(S)], Vh.conj().T[:, :len(S)]
    Atilde = Ur.T @ X2 @ Vr @ np.linalg.inv(np.diag(S))
    eigs = np.linalg.eig(Atilde)[0]
    return eigs, np.log(eigs)

def classify(eigs):
    """
    根据 |λ| 给每个特征值打 Stable/Neutral/Unstable 标签并映射到颜色。
    """
    modulus = np.abs(eigs)
    labels = ['stable' if m < 0.98 else 'unstable' if m > 1.02 else 'neutral' for m in modulus]
    cmap = {'stable':'blue','neutral':'green','unstable':'red'}
    return [cmap[l] for l in labels]

def myHankelTensor2(inputdata, epsilon, d, name):
    X = np.asarray(inputdata)
    T, n = X.shape
    hankelNO = max(0, T - d - epsilon)
    mats = []

    for i in range(hankelNO):
        # 构造 Hankel 第一列和第一行
        c = np.vstack([X[i + j] for j in range(epsilon + 1)])         # shape (ε+1, n)
        r = np.hstack([X[i + epsilon + j] for j in range(epsilon + 1)]) # shape (n*(ε+1),)
        gamma = scipy.linalg.hankel(c, r).astype(np.float32)           # shape (ε+1, n*(ε+1))
        mats.append(gamma)

    if not mats:
        raise ValueError(f"{name}: 序列过短 (T={T})，无法用 ε={epsilon}, d={d} 构造 Hankel 块。")

    stacked = np.stack(mats, axis=0)  # shape (N, H, W)
    # print(f"{name} shape: {stacked.shape}")
    return tf.constant(stacked, dtype=tf.float32)

def run_koopman_analysis(normal_file, abnormal_file):
    """
    运行 Koopman 矩阵分析：加载正常与异常 Koopman 矩阵，
    计算特征值、模态，并绘制谱图；同时计算并输出所有相似度与误差度量。

    参数：
      normal_file   – 正常 Koopman 算子矩阵 (.npy 文件)
      abnormal_file – 异常 Koopman 算子矩阵 (.npy 文件)
    """
    # 1. 加载矩阵
    Kx_norm = np.load(normal_file)
    Kx_abn = np.load(abnormal_file)

    # 2. 计算特征值与模态
    eigvals_norm, W_norm = eig(Kx_norm)
    eigvals_abn, W_abn = eig(Kx_abn)

    # 3. 差异矩阵 & 基本指标
    diff = Kx_norm - Kx_abn
    # print(f"矩阵差异形状: {diff.shape}")
    frob_norm = norm(diff, ord='fro')
    hadamard_norm = np.max(np.abs(diff))
    print(f"Frobenius norm: {frob_norm:.3e}")
    # print(f"Hadamard 范数: {hadamard_norm:.3e}")

    # 4. 绘制 Koopman 特征值谱
    plot_koopman_spectrum(eigvals_norm, W_norm, title='Normal Koopman Eigenvalue')
    plot_koopman_spectrum(eigvals_abn, W_abn, title='Abnormal Koopman Eigenvalue')

    # 5. 相似度 & 误差度量
    cos_sim = cosine_similarity(Kx_norm, Kx_abn)
    corr_cos = corrected_cosine_similarity(Kx_norm, Kx_abn)
    # 将 numpy.ndarray 转为 float
    if isinstance(corr_cos, np.ndarray):
        try:
            corr_cos = corr_cos.item()
        except:
            corr_cos = float(corr_cos.flat[0])

    rmse = np.sqrt(np.mean(diff ** 2))
    sae = np.sum(np.abs(diff))
    manh = manhattan_distance(Kx_norm, Kx_abn)

    print(f"Cosine similarity: {cos_sim:.4f}")
    print(f"Modified cosine similarity: {corr_cos:.4f}")
    print(f"RMSE: {rmse:.3e}")
    print(f"SAE: {sae:.3e}")
    # print(f"曼哈顿距离: {manh:.3e}")

    # 6. 皮尔逊相关系数
    flat1 = Kx_norm.ravel()
    flat2 = Kx_abn.ravel()
    pearson = np.corrcoef(flat1, flat2)[0, 1]
    print(f"Pearson correlation coefficient: {pearson:.4f}")

    # 7. 最大李亚普诺夫指数（基于特征值差分增长率）
    eps = 1e-12
    vals_n = eigvals_norm[np.abs(eigvals_norm) > eps]
    vals_a = eigvals_abn[np.abs(eigvals_abn) > eps]
    dif_eig = np.abs(vals_n - vals_a)
    dt = 0.01  # 假设采样间隔
    with np.errstate(divide='ignore', invalid='ignore'):
        growths = np.real(np.log(dif_eig)) / dt
    if growths.size > 0:
        lyap = np.nanmax(growths)
        print(f"Max Lyapunov Exponent (Abnormal): {lyap:.3e}")
    else:
        print("There are no valid eigenvalues for computing the Lyapunov exponent.")

    # 8. 显示所有绘图
    plt.show()


def run_pca_dmd_analysis(normal_file, abnormal_file):
    """
    运行 PCA + DMD 分析：分别对正常与异常数据进行 PCA 降维，然后构造 DMD，
    并绘制非对数与对数特征值图。
    """
    pca_dims = 22
    # 计算特征值
    e_norm, log_norm = compute_eigs(normal_file, pca_dims)
    e_abn, log_abn = compute_eigs(abnormal_file, pca_dims)
    c_norm = classify(e_norm)
    c_abn = classify(e_abn)
    # 绘图：2行2列子图
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    theta = np.linspace(0, 2*np.pi, 300)
    # (0,0) 正常：非对数谱
    ax = axs[0, 0]
    ax.plot(np.cos(theta), np.sin(theta), '--', color='gray', linewidth=1)
    ax.scatter(e_norm.real, e_norm.imag, c=c_norm, s=50)
    ax.axhline(0, color='gray', linestyle=':')
    ax.axvline(0, color='gray', linestyle=':')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('Re(λ)'); ax.set_ylabel('Im(λ)')
    ax.set_title('Normal – Non-logarithm eigenvalues')
    # (0,1) 异常：非对数谱
    ax = axs[0, 1]
    ax.plot(np.cos(theta), np.sin(theta), '--', color='gray', linewidth=1)
    ax.scatter(e_abn.real, e_abn.imag, c=c_abn, s=50)
    ax.axhline(0, color='gray', linestyle=':')
    ax.axvline(0, color='gray', linestyle=':')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('Re(λ)'); ax.set_ylabel('Im(λ)')
    ax.set_title('Abnormal – Non-logarithm eigenvalues')
    # (1,0) 正常：对数谱
    ax = axs[1, 0]
    ax.scatter(log_norm.imag, log_norm.real, c=c_norm, s=50)
    ax.axhline(0, color='gray', linestyle=':')
    ax.axvline(0, color='gray', linestyle=':')
    ax.set_xlabel('Im(log λ)'); ax.set_ylabel('Re(log λ)')
    ax.set_title('Normal – Logarithmized eigenvalues')
    # (1,1) 异常：对数谱
    ax = axs[1, 1]
    ax.scatter(log_abn.imag, log_abn.real, c=c_abn, s=50)
    ax.axhline(0, color='gray', linestyle=':')
    ax.axvline(0, color='gray', linestyle=':')
    ax.set_xlabel('Im(log λ)'); ax.set_ylabel('Re(log λ)')
    ax.set_title('Abnormal – Logarithmized eigenvalues')
    # 添加图例
    legend_elems = [
        Line2D([0],[0], marker='o', color='w', label='stable',   markerfacecolor='blue',  markersize=8),
        Line2D([0],[0], marker='o', color='w', label='neutral',  markerfacecolor='green', markersize=8),
        Line2D([0],[0], marker='o', color='w', label='unstable', markerfacecolor='red',   markersize=8),
    ]
    axs[1, 1].legend(handles=legend_elems, loc='lower right')
    plt.tight_layout()
    plt.show()

def run_dmd_feature_analysis(normal_file, abnormal_file):
    """
    运行 DMD 特征分析：对正常与异常数据进行 DMD 计算，绘制模态周期 vs 振幅和振幅 vs 增长率。
    """
    # 计算 DMD 特征
    per_n, amp_n, gr_n = load_dmd(normal_file)
    per_a, amp_a, gr_a = load_dmd(abnormal_file)
    # 绘图：2行2列子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
    # (a) 正常数据: 周期 vs 振幅
    ax = axes[0, 0]
    ax.scatter(per_n, amp_n, s=40, c='C0', label='Normal')
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Amplitude')
    ax.set_title('(a) Normal: Modal Period vs. Amplitude')
    ax.grid(True)
    # (b) 异常数据: 周期 vs 振幅
    ax = axes[0, 1]
    ax.scatter(per_a, amp_a, s=40, c='C3', marker='x', label='Abnormal')
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Amplitude')
    ax.set_title('(b) Abnormal: Modal Period vs. Amplitude')
    ax.grid(True)
    # (c) 正常数据: 振幅 vs 增长率
    ax = axes[1, 0]
    ax.scatter(amp_n, gr_n, s=40, c='C0', label='Normal')
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Growth Rate')
    ax.set_title('(c) Normal: Growth Rate vs. Amplitude')
    ax.grid(True)
    # (d) 异常数据: 振幅 vs 增长率
    ax = axes[1, 1]
    ax.scatter(amp_a, gr_a, s=40, c='C3', marker='x', label='Abnormal')
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Growth Rate')
    ax.set_title('(d) Abnormal: Growth Rate vs. Amplitude')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def run_koopman_d_pred(normal_file, abnormal_file, epsilon=1, d=2):
    """
    流程：Hankel 嵌入 → PCA 降维 → 最小二乘估计 Koopman 矩阵 → 计算 d_pred
    """
    # 读取原始序列
    X_np = np.load(normal_file)
    Y_np = np.load(abnormal_file)
    # 嵌入
    H_X_tf = myHankelTensor2(X_np, epsilon, d, name='X_embed')  # shape (T', H, W)
    H_Y_tf = myHankelTensor2(Y_np, epsilon, d, name='Y_embed')  # shape (T', H, W)

    # 步骤2：展平为 (T', H*W)
    H_X_flat = H_X_tf.numpy().reshape(H_X_tf.shape[0], -1)
    H_Y_flat = H_Y_tf.numpy().reshape(H_Y_tf.shape[0], -1)

    # 步骤3：PCA 降维到 D = d * n
    n = X_np.shape[1]
    k = d * n
    pca = PCA(n_components=k)
    H_X_reduced = pca.fit_transform(H_X_flat)  # shape (T', k)
    H_Y_reduced = pca.transform(H_Y_flat)  # shape (T', k)

    # 步骤4：最小二乘估计 Koopman 算子 matrix (k x k)
    G_prev = H_X_reduced[:-1]
    G_next = H_X_reduced[1:]
    matrix = np.linalg.lstsq(G_prev, G_next, rcond=None)[0].T

    # 步骤5：计算公式异常分数
    d_pred = compute_d_pred(H_Y_reduced, matrix)
    print(f"d_pred = {d_pred:.3e}")

