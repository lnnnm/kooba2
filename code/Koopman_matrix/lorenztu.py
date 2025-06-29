import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

DATA_DIR = r'D:\Lujing\EKNO20231212 (2)\Koopman_matrix'
normal_path   = os.path.join(DATA_DIR, 'LorenzDatasets1-2.npy')
abn_path      = os.path.join(DATA_DIR, 'RosslerDatasets.npy')

normal_data   = np.load(normal_path)
abnormal_data = np.load(abn_path)

# ===== 检查 abnormal_data 大小 =====
N_abn = abnormal_data.shape[0]
print(f"abnormal_data has {N_abn} samples.")

# 你原来想画 10000–12000，那么先保证这段在 [0, N_abn] 范围内
anom_start, anom_end = 0, 69
if anom_start >= N_abn:
    raise IndexError(f"Requested start index {anom_start} ≥ data length {N_abn}")
# 确保不越界
anom_end = min(anom_end, N_abn)

print(f"Plotting abnormal_data[{anom_start}:{anom_end}] ...")

# 切片
seg = abnormal_data[anom_start:anom_end, :3]  # shape 会是 (anom_end-anom_start, 3)

# —— 如果这时 seg 还是空，就说明你的 anom_start 写错了 ——
if seg.size == 0:
    raise ValueError("After slicing, no data was selected. 请检查 anom_start/anom_end。")

# —— 积分正常数据（和之前一样） ——————————————————
def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x,y,z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y - beta*z]

initial_normal = normal_data[0, :3]
t0, t1, steps = 0.0, 40.0, 20000
t_eval = np.linspace(t0, t1, steps)
sol_norm = solve_ivp(lorenz, (t0, t1), initial_normal,
                     t_eval=t_eval, method='RK45',
                     atol=1e-9, rtol=1e-6)

# —— 开始画图 ——————————————————————————————
fig = plt.figure(figsize=(12, 5))

# 子图1：正常轨迹
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(sol_norm.y[0], sol_norm.y[1], sol_norm.y[2], lw=0.5, label='Normal')
ax1.scatter(sol_norm.y[0,0], sol_norm.y[1,0], sol_norm.y[2,0],
            c='green', s=50, marker='o', label='Start')
ax1.scatter(sol_norm.y[0,-1], sol_norm.y[1,-1], sol_norm.y[2,-1],
            c='black', s=50, marker='X', label='End')
ax1.set_box_aspect((1,1,1))
ax1.set_title("Normal Data Trajectory")
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
ax1.legend()

# 子图2：直接画异常采样
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot(seg[:,0], seg[:,1], seg[:,2],
         color='red', lw=0.5, label='Abnormal Segment')
ax2.scatter(seg[0,0], seg[0,1], seg[0,2],
            c='green', s=50, marker='o', label='Start')
ax2.scatter(seg[-1,0], seg[-1,1], seg[-1,2],
            c='black', s=50, marker='X', label='End')
ax2.set_box_aspect((1,1,1))
ax2.set_title("Anomalous Data (Direct Plot)")
ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
ax2.legend()

plt.tight_layout()
plt.show()
