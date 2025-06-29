# save_observer_info.py
import numpy as np
import pickle
from myHankel import myHankelTensor
from myObserver import myObserver

# —— 参数 ——  
epsilon = 2        # 和 EKNO 中一致
d_state = 3        # Lorenz 系统维度
dt      = 1

# —— 1. 载入并标准化“正常”数据 ——  
normal = np.load('Lorenz_normal.npy')    # 确保 path 正确
μ = normal.mean(axis=0, keepdims=True)
σ = normal.std(axis=0, keepdims=True) + 1e-8
norm_z = (normal - μ) / σ

# —— 2. Hankel 嵌入 ——  
# 返回 shape = ((ε+1)*d_state, (ε+1)*d_state, num_snapshots)
Gamma_train_Tensor = myHankelTensor(norm_z, epsilon, d_state, name="Gamma_train")

# —— 3. 用 myObserver 训练 g 函数 ——  
model_observer = myObserver(Gamma_train_Tensor, dt)

# —— 4. 提取并保存 特征、输出特征名、系数 ——  
features    = model_observer.feature_names
poly        = model_observer.model.steps[0][1]
output_feats= poly.get_feature_names(features)
linreg      = model_observer.model.steps[-1][1]
coef        = linreg.coef_

# 存磁盘
with open('features.pkl',     'wb') as f: pickle.dump(features,    f)
with open('output_feats.pkl', 'wb') as f: pickle.dump(output_feats,f)
with open('coef.pkl',         'wb') as f: pickle.dump(coef,        f)

print("✅ Saved features.pkl, output_feats.pkl, coef.pkl")
