% === Step 1: 加载数据 ===
S = load('LorenzDatasets1-2.mat');
varname = fieldnames(S);
data = S.(varname{1});
fprintf('Loaded: %s [%d × %d]\n', varname{1}, size(data,1), size(data,2));

% === Step 2: 构造快照对 & 做 DMD ===
X1 = data(:, 1:end-1);
X2 = data(:, 2:end);

[U, S, V] = svd(X1, 'econ');
r = 50;
Ur = U(:,1:r);
Sr = S(1:r,1:r);
Vr = V(:,1:r);

Atilde = Ur' * X2 * Vr / Sr;
[W, D] = eig(Atilde);
lambda = diag(D);              % Koopman 特征值 λ_j
log_lambda = log(lambda);      % 用于谱图

% === Step 3: 分类稳定性 ===
abs_lambda = abs(lambda);
stability = zeros(length(lambda),1);  % 1=stable, 2=neutral, 3=unstable
stability(abs_lambda < 0.98) = 1;      % stable
stability(abs(abs_lambda - 1) < 0.02) = 2;  % neutral
stability(abs_lambda > 1.02) = 3;      % unstable

% === Step 4: 配色 ===
cmap = [0 0.447 0.741;     % blue for stable
        0.466 0.674 0.188; % green for neutral
        0.850 0.325 0.098];% red for unstable
colors = cmap(stability,:);

% === Step 5: 图 (a) 单位圆复平面 ===
figure;
hold on;
theta = linspace(0, 2*pi, 300);
plot(cos(theta), sin(theta), 'k--');  % 单位圆
scatter(real(lambda), imag(lambda), 25, colors, 'filled');
xlabel('Re(\lambda_j)');
ylabel('Im(\lambda_j)');
title('(a) Koopman Eigenvalues');
axis equal; grid on;

% === Step 6: 图 (b) log(λ) 平面 ===
figure;
scatter(imag(log_lambda), real(log_lambda), 25, colors, 'filled');
xlabel('Im(log(\lambda_j))');
ylabel('Re(log(\lambda_j))');
title('(b) Spectral Map (Growth vs Frequency)');
grid on;

% 可选图例
legend({'stable','neutral','unstable'}, 'Location', 'best');
