# demo.py
from above_fig import run_koopman_analysis, run_pca_dmd_analysis, run_dmd_feature_analysis
from above_fig import run_koopman_d_pred

# 文件路径
normal_file = "./Koopman_matrix/RectangeDatasets-0.npy"
abnormal_file = "./Koopman_matrix/CAFUC-abnormal2.npy"

original_normal = "D:\Lujing\EKNO20231212 (2)\EGG_normal.npy"
original_abnormal = "D:\Lujing\EKNO20231212 (2)\EGG_abnormal.npy"

# Generate Koopman eigenvalue spectrum comparison chart and output error index
run_koopman_analysis(normal_file, abnormal_file)
run_koopman_d_pred(original_normal, original_abnormal, epsilon=1, d=2)
# Generate Koopman eigenvalue distribution plot and mark stable/neutral/unstable modes
run_pca_dmd_analysis(normal_file, abnormal_file)

# Generate period-amplitude and growth-amplitude scatter plots of normal vs. abnormal modes
run_dmd_feature_analysis(normal_file, abnormal_file)

