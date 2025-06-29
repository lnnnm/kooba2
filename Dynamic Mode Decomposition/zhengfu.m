% -------- Koopman 模态分解--------
% 加载复数数据矩阵
S = load('LorenzAnmoDatasets-10000-11000-(-4,4).mat');
Data = S.data;
%fprintf('Loaded MAT data:size = [%d %d]\n',size(Data,1),size(Data,2));

X1 = Data(:,1:end-1);
X2 = Data(:,2:end);

% 构造快照序列：X1, X2
X1 = data(:,1:end-1);  % size: [n × (m-1)]
X2 = data(:,2:end);    % size: [n × (m-1)]

% DMD 分解
[U, S, V] = svd(X1, 'econ');
r = 50;  % 截断秩，可调
Ur = U(:,1:r);
Sr = S(1:r,1:r);
Vr = V(:,1:r);

Atilde = Ur' * X2 * Vr / Sr;
[W, D] = eig(Atilde);         % D: Koopman λ
Phi = X2 * Vr / Sr * W;       % Koopman 模态

lambda = diag(D);
log_lambda = log(lambda);

% 计算特征量
period = 2 * pi ./ abs(imag(log_lambda));  % 周期
growth = real(log_lambda);                 % 增长率
amplitude = vecnorm(real(Phi), 2, 1);      % 振幅（L2范数）

% ---------------- 图 (a): Period vs Amplitude ----------------
figure;
scatter(period, amplitude, 35, 'filled');
xlabel('Period (day)');
ylabel('Amplitude');
title('(a) Koopman 模态周期 vs 振幅');
grid on;
xlim([0, min(800, max(period)*1.05)]);

% ---------------- 图 (b): Growth rate vs Amplitude ----------------
figure;
scatter(amplitude, growth, 35, 'filled');
xlabel('Amplitude');
ylabel('Growth rate');
title('(b) Koopman 增长率 vs 振幅');
grid on;
