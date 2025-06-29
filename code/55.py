import torch
import numpy as np
import scipy.linalg
import tensorflow as tf
from sklearn.decomposition import PCA


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
    print(f"{name} shape: {stacked.shape}")
    return tf.constant(stacked, dtype=tf.float32)


def compute_d_pred(H_feat, Kx):
    """
    H_feat: numpy array, shape (T', D)
    Kx:     numpy array, shape (D, D)
    返回 d_pred: float
    """
    Y = torch.from_numpy(H_feat).float()
    K = torch.from_numpy(Kx).float()
    G_prev = Y[:-1]                       # (T'-1, D)
    G_true = Y[1:]                        # (T'-1, D)
    G_pred = (K @ G_prev.t()).t()         # (T'-1, D)
    se = (G_true - G_pred).pow(2).sum()   # ∑‖…‖²
    Tprime = Y.shape[0]
    return torch.sqrt(se).item() / Tprime


if __name__ == "__main__":
    # 参数设置
    epsilon = 1
    d = 2

    # 载入数据
    X_np = np.load(r"D:\Lujing\EKNO20231212 (2)\EGG_normal.npy")
    Y_np = np.load(r"D:\Lujing\EKNO20231212 (2)\EGG_abnormal.npy")

    # 步骤1：Hankel 嵌入
    H_X_tf = myHankelTensor(X_np, epsilon, d, name='X_embed')  # shape (T', H, W)
    H_Y_tf = myHankelTensor(Y_np, epsilon, d, name='Y_embed')  # shape (T', H, W)

    # 步骤2：展平为 (T', H*W)
    H_X_flat = H_X_tf.numpy().reshape(H_X_tf.shape[0], -1)
    H_Y_flat = H_Y_tf.numpy().reshape(H_Y_tf.shape[0], -1)

    # 步骤3：PCA 降维到 D = d * n
    n = X_np.shape[1]
    k = d * n
    pca = PCA(n_components=k)
    H_X_reduced = pca.fit_transform(H_X_flat)  # shape (T', k)
    H_Y_reduced = pca.transform(H_Y_flat)      # shape (T', k)

    # 步骤4：最小二乘估计 Koopman 算子 matrix (k x k)
    G_prev = H_X_reduced[:-1]
    G_next = H_X_reduced[1:]
    matrix = np.linalg.lstsq(G_prev, G_next, rcond=None)[0].T

    # 步骤5：计算公式异常分数
    d_pred = compute_d_pred(H_Y_reduced, matrix)
    print(f"d_pred = {d_pred:.3e}")
