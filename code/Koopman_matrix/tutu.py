# above_fig.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eig
from numpy.linalg import norm
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
import warnings

warnings.filterwarnings("ignore", message="Input data condition number")

def plot_koopman_spectrum(eigvals, modes, title='Koopman 特征值'):
    """
    绘制 Koopman 模态谱（复平面）
    eigvals: 特征值 (复数数组)
    modes: Koopman 模态矩阵 (n_states, n_modes)
    """
    real_parts = eigvals.real
    imag_parts = eigvals.imag
    mode_strength = np.linalg.norm(modes, axis=0)
    rmax = max(1.0, np.max(np.abs(real_parts)), np.max(np.abs(imag_parts)))
    margin = 0.1 * rmax
    # 单位圆
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', linewidth=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_artist(circle)
    sc = ax.scatter(real_parts, imag_parts, c=mode_strength, cmap='viridis', s=80, edgecolors='k')
    plt.colorbar(sc, label='Modal Strength ||Φ||')
    ax.set_xlim(-rmax - margin, rmax + margin)
    ax.set_ylim(-rmax - margin, rmax + margin)
    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()

def cosine_similarity(matrix1, matrix2):
    """计算两个矩阵的余弦相似度"""
    flat1, flat2 = matrix1.ravel(), matrix2.ravel()
    return np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2))

def manhattan_distance(matrix1, matrix2):
    """计算两个矩阵的曼哈顿距离"""
    if matrix1.shape != matrix2.shape:
        raise ValueError(f"矩阵形状不一致: {matrix1.shape} vs {matrix2.shape}")
    return np.sum(np.abs(matrix1 - matrix2))

def run_koopman_analysis(normal_file, abnormal_file):
    """
    生成正常 vs 异常 Koopman 矩阵特征值谱图，并输出误差指标
    """
    # 载入 Koopman 算子矩阵
    Kx_norm = np.load(normal_file)
    Kx_abn = np.load(abnormal_file)
    # 计算特征值和模态
    eigvals_norm, W_norm = eig(Kx_norm)
    eigvals_abn, W_abn = eig(Kx_abn)
    # 绘制特征值谱
    plot_koopman_spectrum(eigvals_norm, W_norm, title='Normal Koopman Eigenvalue')
    plot_koopman_spectrum(eigvals_abn, W_abn, title='Abnormal Koopman Eigenvalue')
    # 计算误差指标
    diff = Kx_norm - Kx_abn
    frob_norm = norm(diff, ord='fro')
    cos_sim = cosine_similarity(Kx_norm, Kx_abn)
    rmse = np.sqrt(np.mean(diff**2))
    sae = np.sum(np.abs(diff))
    manh = manhattan_distance(Kx_norm, Kx_abn)
    # 输出指标
    print(f"Frobenius 范数: {frob_norm:.3e}")
    print(f"余弦相似度: {cos_sim:.4f}")
    print(f"RMSE: {rmse:.3e}")
    print(f"SAE: {sae:.3e}")
    print(f"曼哈顿距离: {manh:.3e}")
    plt.show()

def compute_eigs(data_path, pca_dims):
    """加载 npy 数据 → 标准化 → PCA → DMD → 返回 特征值 和 对数特征值"""
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
    """根据特征值模长分类：模长<0.98: stable, >1.02: unstable, else neutral"""
    modulus = np.abs(eigs)
    labels = []
    for m in modulus:
        if m < 0.98:
            labels.append('stable')
        elif m > 1.02:
            labels.append('unstable')
        else:
            labels.append('neutral')
    cmap = {'stable':'blue','neutral':'green','unstable':'red'}
    return [cmap[l] for l in labels]

def run_pca_dmd_analysis(normal_file, abnormal_file):
    """
    PCA降维后进行DMD分析，生成正常 vs 异常 特征值分布图及分类
    """
    pca_dims = 22
    # 计算特征值及其对数
    e_norm, log_norm = compute_eigs(normal_file, pca_dims)
    e_abn, log_abn = compute_eigs(abnormal_file, pca_dims)
    c_norm = classify(e_norm)
    c_abn = classify(e_abn)
    # 绘制散点图
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    theta = np.linspace(0, 2*np.pi, 300)
    # 正常: 非对数谱
    ax = axs[0,0]
    ax.plot(np.cos(theta), np.sin(theta), '--', color='gray', linewidth=1)
    ax.scatter(e_norm.real, e_norm.imag, c=c_norm, s=50)
    ax.axhline(0, color='gray', linestyle=':')
    ax.axvline(0, color='gray', linestyle=':')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('Re(λ)'); ax.set_ylabel('Im(λ)')
    ax.set_title('Normal – Non-logarithm eigenvalues')
    # 异常: 非对数谱
    ax = axs[0,1]
    ax.plot(np.cos(theta), np.sin(theta), '--', color='gray', linewidth=1)
    ax.scatter(e_abn.real, e_abn.imag, c=c_abn, s=50)
    ax.axhline(0, color='gray', linestyle=':')
    ax.axvline(0, color='gray', linestyle=':')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('Re(λ)'); ax.set_ylabel('Im(λ)')
    ax.set_title('Abnormal – Non-logarithm eigenvalues')
    # 正常: 对数谱
    ax = axs[1,0]
    ax.scatter(log_norm.imag, log_norm.real, c=c_norm, s=50)
    ax.axhline(0, color='gray', linestyle=':')
    ax.axvline(0, color='gray', linestyle=':')
    ax.set_xlabel('Im(log λ)'); ax.set_ylabel('Re(log λ)')
    ax.set_title('Normal – Logarithmized eigenvalues')
    # 异常: 对数谱
    ax = axs[1,1]
    ax.scatter(log_abn.imag, log_abn.real, c=c_abn, s=50)
    ax.axhline(0, color='gray', linestyle=':')
    ax.axvline(0, color='gray', linestyle=':')
    ax.set_xlabel('Im(log λ)'); ax.set_ylabel('Re(log λ)')
    ax.set_title('Abnormal – Logarithmized eigenvalues')
    # 图例
    legend_elems = [
        Line2D([0],[0], marker='o', color='w', label='stable', markerfacecolor='blue', markersize=8),
        Line2D([0],[0], marker='o', color='w', label='neutral', markerfacecolor='green', markersize=8),
        Line2D([0],[0], marker='o', color='w', label='unstable', markerfacecolor='red', markersize=8),
    ]
    axs[1,1].legend(handles=legend_elems, loc='lower right')
    plt.tight_layout()
    plt.show()

def load_dmd(path, r=200):
    """
    载入数据并执行DMD，返回周期、幅值、增长率
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

def run_dmd_feature_analysis(normal_file, abnormal_file):
    """
    执行DMD分析，绘制正常 vs 异常 模态的周期-幅值 和 增长率-幅值 关系图
    """
    per_n, amp_n, gr_n = load_dmd(normal_file)
    per_a, amp_a, gr_a = load_dmd(abnormal_file)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
    # (a) 正常: 周期 vs 幅值
    ax = axes[0, 0]
    ax.scatter(per_n, amp_n, s=40, c='C0', label='Normal')
    ax.set_xlabel('周期 (秒)')
    ax.set_ylabel('幅值')
    ax.set_title('(a) 正常: 周期 vs 幅值')
    ax.grid(True)
    # (b) 异常: 周期 vs 幅值
    ax = axes[0, 1]
    ax.scatter(per_a, amp_a, s=40, c='C3', marker='x', label='Abnormal')
    ax.set_xlabel('周期 (秒)')
    ax.set_ylabel('幅值')
    ax.set_title('(b) 异常: 周期 vs 幅值')
    ax.grid(True)
    # (c) 正常: 增长率 vs 幅值
    ax = axes[1, 0]
    ax.scatter(amp_n, gr_n, s=40, c='C0', label='Normal')
    ax.set_xlabel('幅值')
    ax.set_ylabel('增长率')
    ax.set_title('(c) 正常: 增长率 vs 幅值')
    ax.grid(True)
    # (d) 异常: 增长率 vs 幅值
    ax = axes[1, 1]
    ax.scatter(amp_a, gr_a, s=40, c='C3', marker='x', label='Abnormal')
    ax.set_xlabel('幅值')
    ax.set_ylabel('增长率')
    ax.set_title('(d) 异常: 增长率 vs 幅值')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
