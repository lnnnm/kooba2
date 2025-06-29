# This is EKNO code main
# wrote by lujing
# lujing_cafuc@nuaa.edu.cn
# 2023-06-01
#######################################################################
import random
from scipy import signal
import sklearn.metrics
import mne
from myHankel import myHankelTensor, myHankelRerverse_Tensor
from myObserver import *
from myKoopman import *
from myDatasets import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings("ignore")

def print_hi(name):
    print(r'{name}')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #GPU mode
    print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))
    tf.config.set_soft_device_placement(True)

    # Initial parameters
    # k1 = m
    k2 = 30
    k3 = 50
    k4 = 100
    k5 = 937
    k6 = 2500
    k7 = 5000
    k8 = 10000
    k9 = 20000

    epsilon = 2  # hankel矩阵维度：(epsilon+1)*d * (epsilon+1)*d
    d = 4  # d是输入值的维度 1-9
    par = np.arange(d)
    step = 3

    TotalDatalen=k6
    kk_train = TotalDatalen
    kk_test = math.floor(kk_train * 1.2)

    flag_train = True
    flag_hankel = False
    flag_datasets = 'CAFUC'
    flag = False
    # flag = True
    # flag_hankel = True
    # flag_koopman = True  # 是否重新计算koopman模式，true是重新计算，false是直接调用已保存模式
    # flag_koopman= False #是否重新计算koopman模式，true是重新计算，false是直接调用已保存模式

    # Load the datasets
    datasets = 'EGGDatasets'
    datasetsName = "CAFUC-abnormal3"  #飞行训练科目
    CAFUC_datasetsName = "temp"
    # Lorenz DATA
    #datasets = 'LorenzDatasets'

    # dirbegin = 1
    # dirend = 2
    if datasets == 'CAFUC':
        BigData, BigDataFilename, colNeed = datasets_CAFUC(datasetsName=datasetsName, dirName="dataRaw/",
                                                           outDir="output/")
        dir = 'output/' + datasetsName + '/'
        if(flag == True):   #没有bigdata时运行
            # datasets_CAFUC_prepare(dirbegin,dirend,"./dataRaw/",datasetsName,dir,colNeed,BigDataFilename)
            datasets_CAFUC_prepare(1,2, "./dataRaw/", datasetsName, dir, colNeed, BigDataFilename)
        print("BigDataFilename", BigDataFilename)

        inputData_raw = pd.read_csv(BigDataFilename)  ##true data
        d = 3  # d是输入值的维度 1-12
        par = np.arange(d)
        Normal_data= inputData_raw.iloc[:, :d]
        anom_start, anom_end = 0, 1000
        anomaly_data = Normal_data.copy()
        anomaly_data[anom_start:anom_end] += (np.random.rand(1000, 3) * 8 - 4)
        print(f"anomaly_data 共 {anomaly_data.shape[0]} 行，{anomaly_data.shape[1]} 列")

        # np.save('CAFUC_normal.npy', Normal_data)
        np.save('CAFUC_anomaly-3.npy', Normal_data)
        inputData = Normal_data
        flag_datasets = 'CAFUC'

    elif datasets == 'LorenzDatasets11':
        # ——— 1. 生成干净的 Lorenz 轨迹 —————————————————
        BigData = dataset_lorenz_attractor(N=20000)  # shape=(20000,3)
        normal_data = BigData.copy()

        # ——— 2. 在前 rows 步注入“异常”噪声 ————————————
        rows = 1000
        anomaly_data = BigData.copy()
        # 对每个点、每个坐标都加独立噪声
        noise =(np.random.rand(rows, 3) * 8 - 4)
        anomaly_data[0:0+rows, :] += noise

        np.save('LorenzDatasets-anomaly.npy', anomaly_data)
        # ——— 3. 画图：左图正常，右图异常 —————————————————
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 5))

        # 左：正常轨迹
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot(*normal_data.T, lw=0.5, c='blue', label='Normal Trajectory')
        ax1.scatter(*normal_data[0], c='green', s=60, marker='o', label='Start')
        ax1.scatter(*normal_data[-1], c='black', s=60, marker='X', label='End')
        ax1.set_box_aspect((1, 1, 1))
        ax1.set_title("Normal Lorenz Trajectory")
        ax1.set_xlabel("X");
        ax1.set_ylabel("Y");
        ax1.set_zlabel("Z")
        ax1.legend()

        # 右：注入噪声后的轨迹
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        # 红色：前 rows 步“异常”
        ax2.plot(*anomaly_data[0:0+rows].T, lw=0.5, c='red', label='Abnormal Segment')
        # 蓝色：后面恢复正常
        ax2.plot(*anomaly_data[rows:].T, lw=0.5, c='blue', label='Back to Normal')
        #ax2.plot(*anomaly_data[12000:].T, lw=0.5, c='blue', label='Back to Normal')
        ax2.scatter(*anomaly_data[0], c='green', s=60, marker='o', label='Start')
        ax2.scatter(*anomaly_data[-1], c='black', s=60, marker='X', label='End')
        ax2.set_box_aspect((1, 1, 1))
        ax2.set_title("Trajectory with Injected Anomaly")
        ax2.set_xlabel("X");
        ax2.set_ylabel("Y");
        ax2.set_zlabel("Z")
        ax2.legend()

        plt.tight_layout()
        plt.show()

        # ——— 4. 把 anomaly_data 作为后续算法的 inputData —————————
        inputData = anomaly_data
        d = inputData.shape[1]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, 'LorenzDatasets-normal.npy')
        np.save(save_path, normal_data)
        print("已保存到：", save_path)
        np.save('anomaly_data.npy', anomaly_data)

    elif datasets == 'LorenzDatasets':
        BigData = dataset_lorenz_attractor(N=20000)
        normal_data = BigData.copy()

        start = 10000
        end = 11000
        rows = 1000
        for row in range(start, end):
            for clo in range(3):
                BigData[row, clo] += 1*random.uniform(-1,1)

        anomaly_data = BigData
        #保存未经过koopman函数处理的原始数据
        np.save('Lorenz_normal.npy', normal_data)
        np.save('Lorenz_anomaly-10000-11000.npy', anomaly_data)

        d = BigData.shape[1]
        inputData = BigData
    elif datasets == 'RosslerDatasets':
        BigData = dataset_rossler(N=20000)
        normal_data = BigData.copy()
        rows = 1000     #异常行数
        for row in range(0, 0 + rows):  #异常行数的范围
            for clo in range(3):
                BigData[row, clo] = random.uniform(-4, 4)  #异常值设置的范围
        start = 10000
        end = 12000
        rows = 2000
        for row in range(start, end):
            for clo in range(3):
                BigData[row, clo] += 1 * random.uniform(-4, 4)

        anomaly_data = BigData
        np.save('Rossler_normal.npy', normal_data)
        np.save('Rossler_anomaly-10000-12000(-4,4).npy', anomaly_data)
        d = BigData.shape[1]
        inputData = BigData
    elif datasets == 'EGGDatasets':
        edf_path = r"D:\Lujing\EKNO20231212 (2)\dataRaw\EGGDatasets\chb24_01.edf"  # 替换为实际EDF文件路径
        raw = mne.io.read_raw_edf(edf_path, preload=True)

        # 预处理：滤波并提取EEG通道
        raw.filter(1, 40)  # 1-40Hz带通滤波，保留癫痫相关频段
        eeg_channels = raw.copy().pick_types(eeg=True)

        # 转换为numpy数组并降采样（可选）
        data = eeg_channels.get_data().T  # [样本数, 通道数]
        if data.shape[0] > 50000:  # 如果数据太长，降采样减少计算量
            data = signal.resample(data, 50000)
        normal_data = data.copy()

        # 注入模拟癫痫异常
        seizure_start = 0  # 异常起始点
        seizure_duration = 1000  # 异常持续长度（样本数）
        seizure_end = seizure_start + seizure_duration

        # 对选定时间段内的数据添加癫痫特征（高频、高幅波动）
        for ch in range(data.shape[1]):  # 对每个通道
            # 添加高频正弦波（模拟癫痫棘波）
            seizure_wave = 20 * np.sin(2 * np.pi * 15 * np.arange(seizure_duration) / 256)
            # 添加随机噪声增加真实性
            noise = np.random.normal(-4, 4, seizure_duration)
            data[seizure_start:seizure_end, ch] += seizure_wave + noise
        anomaly_data = data
        np.save('EGG_normal.npy', normal_data)
        # np.save('EGG_anomaly-0-1000(-4,4).npy', anomaly_data)
        # 为Koopman处理准备输入数据
        inputData = anomaly_data
    # input and prepare
    (m, n) = inputData.shape
    print(m, n)

    # Initial the data for train and test
    print('................................Datasets prepare is starting................................')
    #生成训练数据
    #stannd scaler
    stand_scaler = StandardScaler()
    stand_scaler = stand_scaler.fit(inputData)
    inputData_stand = stand_scaler.transform(inputData)
    print('stand_scaler.mean_:', stand_scaler.mean_)

    for i in range(d):
        X_value = inputData_stand[:kk_train, par[i]].reshape(-1, 1)
        X_t = X_value
        if i == 0:
            X = X_t
        else:
            X = np.concatenate((X, X_t), axis=1)
    X_train = X
    print("X_train.shape:", X_train.shape)

    #生成测试数据
    for i in range(d):
        Y_value = inputData_stand[kk_train:kk_test, par[i]].reshape(-1, 1)
        Y_t = Y_value
        if i == 0:
            Y = Y_t
        else:
            Y = np.concatenate((Y, Y_t), axis=1)
    X_test = Y
    print("X_test.shape:", X_test.shape)
    X_ll_train = X_train[:, :2]  # save lat log
    X_ll_test = X_test[:, :2]

    print('................................Datasets prepare is end................................')

    if flag_hankel==True:
        # Hankel Emedding 调用myhankel函数
        Gamma_train_Tensor = myHankelTensor(X_train, epsilon, d, "Gamma_train_Tensor")
        #hankel_test
        Gamma_test_Tensor = myHankelTensor(X_test, epsilon, d, "Gamma_test_Tensor")

        # Observer g learning
        print('................................Observer Gamma Train is starting........................')
        dt = 1
        Gamma_train = mode3Unfolding_T(Gamma_train_Tensor)
        Gamma_test = mode3Unfolding_T(Gamma_test_Tensor)
        model_observer_Tensor = myObserver(Gamma_train_Tensor, dt)
        time_start = time.time()  # 开始计时

        dif_Gamma_train_Tensor = model_observer_Tensor.differentiate(Gamma_train_Tensor, t=dt)  ###求导数
        dif_Gamma_test_Tensor = model_observer_Tensor.differentiate(Gamma_test_Tensor, t=dt)

        res_Gamma_train = model_observer_Tensor.predict(Gamma_train_Tensor)
        res_Gamma_train_Tensor = mode1Folding(res_Gamma_train, np.array(Gamma_train_Tensor.shape))
        err_train_Tensor_hat = sklearn.metrics.mean_squared_error(Gamma_train, res_Gamma_train)

        score_train_Tensor = model_observer_Tensor.score(Gamma_train_Tensor, t=dt)
        err_train_Tensor, err_train_d_Tensor = err_order_Tensor(d, res_Gamma_train_Tensor, dif_Gamma_train_Tensor)
        error_Reconstruct_train_Tensor = err_train_Tensor[err_train_d_Tensor[0]]  ##g观测函数重构误差  导数  取最小的重构误差

        res_Gamma_test = model_observer_Tensor.predict(Gamma_test_Tensor)
        res_Gamma_test_Tensor = mode1Folding(res_Gamma_test, np.array(Gamma_test_Tensor.shape))
        err_test_hat = sklearn.metrics.mean_squared_error(Gamma_test, res_Gamma_test)
        score_test = model_observer_Tensor.score(Gamma_test_Tensor, t=dt)  ##采用的r2_score
        err_test, err_test_d = err_order_Tensor(d, res_Gamma_test_Tensor, dif_Gamma_test_Tensor)
        error_Reconstruct_test = err_test[err_test_d[0]]  ##g观测函数重构误差  导数  取最小的重构误差

        print('score_train_Tensor:', score_train_Tensor, "/score_test:", score_test)
        print('err_train_Tensor:', err_train_Tensor_hat, "/err_test:", err_test_hat)
        print('error_Reconstruct_train_Tensor:', error_Reconstruct_train_Tensor, '/error_Reconstruct_test:',
              error_Reconstruct_test)
        print('res_Gamma_train.shape', res_Gamma_train_Tensor.shape, '/res_Gamma_test.shape', res_Gamma_test.shape)
        time_end = time.time()
        print('predicting time cost:', time_end - time_start, 's')
        X_train_dif_hat = myHankelRerverse_Tensor(res_Gamma_train_Tensor, X_train, epsilon, d)
        X_train_dif = myHankelRerverse_Tensor(dif_Gamma_train_Tensor, X_train, epsilon, d)
        err_real = sklearn.metrics.mean_squared_error(X_train_dif, X_train_dif_hat)
        print("Error_real:", err_real)

        plotResults2D_Res(par, X_train_dif_hat, X_train_dif, d, title="Dif,Score:%.2f" % (score_train_Tensor * 100))
        plot3D(X_train_dif, X_train_dif_hat, flag=flag_datasets, title="Dif",data_ll_true=X_ll_train,data_ll_res=X_ll_test)
        # stannd scaler inverse
        Y_train = stand_scaler.inverse_transform(X_train)
        Y_train_hat = stand_scaler.inverse_transform(X_train_dif_hat)
        plotResults2D_Res(par, data_res=Y_train_hat, data_true=Y_train,
                          d=d, model_eqn=model_observer_Tensor.equations(),
                          title="RealValue,Score:%.2f" % (score_train_Tensor * 100))
        plot3D(Y_train, Y_train_hat, flag=flag_datasets, title="RealValue",data_ll=None)
        Gamma_train = X_train_dif_hat

    else:
        #######################################################################
        # Observer g learning
        print('................................Observer Gamma Train is starting........................')

        #test real value
        if datasets=='CAFUC':
            X_train = X_train[:,2:]
            X_test = X_test[:,2:]
            d=d-2

        plot3D(X_train, X_test, flag=flag_datasets, title="Real Value train&test", data_ll_true=X_ll_train,data_ll_res=X_ll_test)
        plotResults2D_Res(par, X_test, X_train, d, title="Real Value train&test")

        time_start = time.time()  # 开始计时
        model = myObserver(X_train, dt=1)
        input_features = model.feature_names
        output_features = model.model.steps[0][1].get_feature_names(input_features)
        coef = model.model.steps[-1][1].coef_
        print("N=", len(input_features), ": input_features is ", input_features)
        print("N=", len(output_features), ": output_features is ",output_features)
        print("L.shape:", coef.shape)

        dif_X_train = model.differentiate(X_train, t=1)
        dif_X_test = model.differentiate(X_test, t=1)
        dif_X_train_hat = model.predict(dif_X_train)
        X_train_hat = model.predict(X_train)
        X_test_hat = model.predict(X_test)
        score_train = model.score(X_train)
        score_test = model.score(X_test)
        mse_train = sklearn.metrics.mean_squared_error(dif_X_train, X_train_hat)
        mse_train_real = sklearn.metrics.mean_squared_error(X_train,X_train_hat)
        mse_test = sklearn.metrics.mean_squared_error(dif_X_test,X_test_hat)
        print('score_train:', score_train,"/score_test:",score_test)
        print('mse_train:', mse_train,"/mse_test:",mse_test)

        # g_function g_train L
        g_train = mytheta(dif_X_train,input_features,output_features,coef)
        g_train_hat = mytheta(X_train_hat,input_features,output_features,coef)
        g_test = mytheta(dif_X_test,input_features,output_features,coef)
        g_test_hat = mytheta(X_test_hat,input_features,output_features,coef)
        print("X_train.shape:",X_train.shape,"=> g_train.shape:", g_train.shape)
        print("input_features:",input_features)
        print("output_features:",output_features)
        plotResults2D_Res(par,g_train_hat,g_train,d=9,title="g_train")
        time_end = time.time()
        print('predicting time cost:', time_end - time_start, 's')

        plot3D(dif_X_train, X_train_hat,flag=flag_datasets, title="Dif", data_ll_true=X_ll_train,data_ll_res=X_ll_train)
        plotResults2D_Res(par, X_train_hat, dif_X_train, d, title="NO Hankel Dif,Score:%.2f,mse:%.2f" % (score_train * 100,mse_train_real))

        Gamma_train = X_train_hat

    print('................................Observer Gamma Train is end................................')

    time_start = time.time()  # 开始计时
    x0=np.array(X_test[0,:])   #test第一个值 向前预测
    print('x0.shape',x0.shape)
    step_simulation=X_test.shape[0]
    print('step_simulation:',step_simulation)
    t_end_test=step_simulation
    tt=np.arange(0,t_end_test)
    pred_X = model.simulate(x0,tt,integrator="odeint")
    true_X = dif_X_test[:step_simulation,:]
    # 如果有空值则把空值换成平均值
    true_X_no_nan = np.nan_to_num(true_X, nan=np.nanmean(true_X))
    pred_X_no_nan = np.nan_to_num(pred_X, nan=np.nanmean(pred_X))

    err_simulaiton_noK=sklearn.metrics.mean_squared_error(true_X, pred_X)
    print('err_simulaiton_noK:%f for step: %d'%(err_simulaiton_noK,step_simulation))

    X_hat = np.append(dif_X_train_hat, pred_X,axis=0)
    X = np.append(dif_X_train,true_X,axis=0)
    X_hat =X_hat[-300:,:]
    X= X[-300:,:]

    # Koopman K learning
    print('................................Koopman K Train is starting........................')

    X = g_train
    Y = g_test
    pred_train, pred_test, Koopman_matrix=myModel(X,Y,window_width=20,columns=output_features)
    plotResults2D_Res(par,pred_train,g_train,d=10,title='koopman_train')
    plot3D(g_train,pred_train,flag=flag_datasets,title='Koopman_train',  data_ll_true=X_ll_train,data_ll_res=X_ll_train)

    # 将 Tensor 转换为 NumPy 数组
    numpy_array = Koopman_matrix.numpy()

    # 保存为 .npy 文件
    # np.save(f'./Koopman_matrix/{datasetsName}{dirbegin}-{dirend}.npy', numpy_array)
    if datasets != "CAFUC" :
        np.save(f'./Koopman_matrix/{datasetsName}.npy', numpy_array)
    else :
        np.save(f'./Koopman_matrix/{CAFUC_datasetsName}.npy', numpy_array)

    print("Koopman_matrix 已保存")
    #output to csv
    ffname = 'output/' + datasetsName
    np.savetxt(ffname + '_g_train.csv', np.column_stack((X_ll_train[:, 1], X_ll_train[:, 0], g_train[:, 0])),
               fmt='%.5f', delimiter=',')
    np.savetxt(ffname + '_pred_train.csv', np.column_stack((X_ll_train[:, 1], X_ll_train[:, 0], pred_train[:, 0])),
               fmt='%.5f', delimiter=',')

    np.savetxt(ffname + '_g_test.csv', np.column_stack((X_ll_test[:, 1], X_ll_test[:, 0], g_test[:, 0])),
               fmt='%.5f', delimiter=',')
    np.savetxt(ffname + '_pred_test.csv', np.column_stack((X_ll_test[:, 1], X_ll_test[:, 0], pred_test[:, 0])),
               fmt='%.5f', delimiter=',')

    print('................................Koopman K Train is ending........................')
    plt.show()