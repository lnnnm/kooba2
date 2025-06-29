import numpy as np
import matplotlib.pyplot as plt

def load_dmd(path, r=200):
    """
    Load a .npy snapshot matrix and perform DMD.
    Returns:
      period      – modal periods (2π/|Im(log λ)|)
      amplitude   – modal amplitudes
      growth_rate – modal growth rates (Re(log λ))
    """
    Data = np.load(path)
    X1, X2 = Data[:, :-1], Data[:, 1:]
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    r_cut = min(X1.shape[1], r)
    Ur, Vr = U[:, :r_cut], Vh.conj().T[:, :r_cut]
    S_inv = np.diag(1.0 / S[:r_cut])
    Atilde = Ur.T @ X2 @ Vr @ S_inv

    eigs, W = np.linalg.eig(Atilde)
    log_e   = np.log(eigs)

    # periods
    with np.errstate(divide='ignore', invalid='ignore'):
        period = 2 * np.pi / np.abs(log_e.imag)
    # growth rates
    growth_rate = log_e.real
    # amplitudes
    Phi       = X2 @ Vr @ S_inv @ W
    amplitude = np.linalg.norm(Phi.real, axis=0)

    mask = np.isfinite(period)
    return period[mask], amplitude[mask], growth_rate[mask]

# ——— USER CONFIGURATION: paths to your normal & abnormal .npy files ———
normal_file   = r"D:\Lujing\EKNO20231212 (2)\Koopman_matrix\RectangeDatasets-0.npy"
abnormal_file = r"D:\Lujing\EKNO20231212 (2)\Koopman_matrix\CAFUC-abnormal3.npy"


# ————————————————————————————————————————————————————————————
# ——

# Compute DMD features for each dataset
per_n, amp_n, gr_n = load_dmd(normal_file)
per_a, amp_a, gr_a = load_dmd(abnormal_file)

# Create a 2×2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)

# (a) Normal data: Period vs. Amplitude
ax = axes[0, 0]
ax.scatter(per_n, amp_n, s=40, c='C0', label='Normal')
ax.set_xlabel('Seconds')
ax.set_ylabel('Amplitude')
ax.set_title('(a) Normal: Modal Period vs. Amplitude')
ax.grid(True)

# (b) Abnormal data: Period vs. Amplitude
ax = axes[0, 1]
ax.scatter(per_a, amp_a, s=40, c='C3', marker='x', label='Abnormal')
ax.set_xlabel('Seconds')
ax.set_ylabel('Amplitude')
ax.set_title('(b) Abnormal: Modal Period vs. Amplitude')
ax.grid(True)

# (c) Normal data: Amplitude vs. Growth Rate
ax = axes[1, 0]
ax.scatter(amp_n, gr_n, s=40, c='C0', label='Normal')
ax.set_xlabel('Amplitude')
ax.set_ylabel('Growth Rate')
ax.set_title('(c) Normal: Growth Rate vs. Amplitude')
ax.grid(True)

# (d) Abnormal data: Amplitude vs. Growth Rate
ax = axes[1, 1]
ax.scatter(amp_a, gr_a, s=40, c='C3', marker='x', label='Abnormal')
ax.set_xlabel('Amplitude')
ax.set_ylabel('Growth Rate')
ax.set_title('(d) Abnormal: Growth Rate vs. Amplitude')
ax.grid(True)

plt.tight_layout()
plt.show()
