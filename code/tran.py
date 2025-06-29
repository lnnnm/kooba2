import numpy as np
from scipy.io import savemat

# 1. 读入 .npy 文件
arr = np.load(r'D:\Lujing\EKNO20231212 (2)\Koopman_matrix\CAFUC-abnormal3.npy',
              allow_pickle=False)

# 2. 存成 .mat 文件，变量名这里叫 'data'
savemat(r'CAFUC-abnormal3.mat',
        {'data': arr})
