%whos('-file','RectangeDatasets.mat')
S = load('RectangeDatasets-AnomalyDatasets_dif.mat');
Data = S.data;
%fprintf('Loaded MAT data:size = [%d %d]\n',size(Data,1),size(Data,2));

X1 = Data(:,1:end-1);
X2 = Data(:,2:end);

%DMD分解
[U,S,V] = svd(X1,'econ');
r = 20;
Ur = U(:,1:r);Sr = S(1:r,1:r);Vr = V(:,1:r);

Atilde = Ur' * X2 *Vr /Sr;
[W,D]=eig(Atilde);
Phi = X2 * Vr /Sr *W;


% disp(size(Phi));
% disp(size(model));
model = real(Phi(:,1));
%Z = reshape(model,[265,265]);

figure;
imagesc(model');

%imagesc(real(Data));
%axis image;

colorbar;
%colormap jet;
title('Koopman ')
