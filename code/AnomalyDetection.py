import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns

def detect_koopman_anomalies(
        U_base,
        U_current_list,
        method='percentile',  # 'statistical' or 'percentile'
        k=3,
        percentile=95,
        distance_metric='frobenius',  # 可扩展为 'spectral' 或 'cosine'
        plot=True
):
    """
    - U_base: 基准 Koopman 算子 (ndarray)
    - U_current_list: 当前算子列表 (List[ndarray])
    - method: 阈值设定方法，'statistical' 或 'percentile'
    - k: 若使用 statistical，则为 k * std 的倍数
    - percentile: 若使用 percentile，则为多少分位数
    - distance_metric: 距离度量方法，'frobenius'
    - plot: 是否绘图显示异常结果
    - scores: 每个算子的异常分数
    - threshold: 检测使用的阈值
    - flags: 异常标记（True表示异常）
    """

    def compute_distance(U1, U2, metric='frobenius'):
        if metric == 'frobenius':
            U1, U2 = np.asarray(U1), np.asarray(U2)

            if U1.shape != U2.shape:
                raise ValueError("U1 and U2 must have the same shape.")

            # 若是2维：直接计算 Frobenius
            if U1.ndim == 2:
                return norm(U1 - U2, ord='fro')

            # 若是3维及以上：使用欧几里得范数直接计算
            elif U1.ndim > 2:
                return np.array([
                    norm(U1 - U2)
                ])
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")

    # Step 1: 计算所有 Koopman 算子的差异分数
    score_m = np.array([
        float(compute_distance(U_cur, U_base, metric=distance_metric))
        for U_cur in U_current_list
    ])
    print(score_m)
    # # 对可能为向量的 score 做平均，保证每个 score 是一个标量
    # score_m = np.array([
    #     np.mean(score) if isinstance(score, (list, np.ndarray)) else score
    #     for score in scores
    # ])

    # Step 2: 设定阈值
    if method == 'statistical':
        mean = np.mean(score_m)
        std = np.std(score_m)
        threshold = mean + k * std
    elif method == 'percentile':
        threshold = np.percentile(score_m, percentile)
    else:
        raise ValueError(f"Unsupported threshold method: {method}")

    # Step 3: 标记异常
    flags = score_m > threshold

    # # Step 4: 可视化
    # if plot:
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(score_m, label="Anomaly Score", marker='o')
    #     plt.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold = {threshold:.2f}")
    #     plt.xlabel("Index")
    #     plt.ylabel("Score")
    #     plt.title("Koopman Anomaly Detection")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()

    # Step 4: 可视化 - 热力图
    if plot:
        plt.figure(figsize=(12, 1.5))
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        sns.heatmap(
            score_m[np.newaxis, :],  # 增加一个维度使其变成 2D
            cmap=cmap,
            cbar=True,
            xticklabels=np.arange(len(score_m)),
            yticklabels=["Score"],
            linewidths=0.5,
            linecolor='gray',
            annot=True,
            fmt=".2f"
        )
        plt.axhline(y=0, color='black', linewidth=1)
        plt.title(f"Koopman Anomaly Scores Heatmap (Threshold = {threshold:.2f})")
        plt.tight_layout()
        plt.show()


    # Step 5: 打分
    grades = []
    for score in score_m:
        scalar_score = np.mean(score) if isinstance(score, (np.ndarray, list)) else score
        diff = scalar_score - threshold
        grades.append(abs(diff))
        # print(threshold)
        # if diff <= 0:
        #     grades.append(100)
        # elif diff <= 5 :
        #     grades.append(80)
        # elif diff <= 10 :
        #     grades.append(60)
        # else:
        #     grades.append(40)
    print(grades)
    return score_m, threshold, flags,  grades
