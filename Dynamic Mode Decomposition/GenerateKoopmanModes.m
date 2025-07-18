%% Generate Koopman PM2.5 Modes
% Dynamic Mode Decomposition of PM2.5

function [Psi]=GenerateKoopmanModes(data,mode1,mode2,month)
%% Load Data
clc; close all;
disp('Loading Data Set...')
tic
if strcmp(data,'Day_mean')
Data=dlmread(fullfile('month','2018-04.txt'));
%Data=readmatrix(fullfile('delay=7','2018-01.csv'));
delay=7; dtype='Mean'; delt=1; delx=1;
hwy='day'; hwylength=731; xpath='x121.txt'; ypath='y121.txt';

elseif strcmp(data,'Normal')
    S = load('RectangeDatasets-0.mat');
    Data = S.data;
    fprintf('Loaded MAT data:size = [%d %d]\n',size(Data,1),size(Data,2));
    
    delay = 7;delay=7; dtype='Mean'; delt=1; delx=1;hwy='day';hwylength=size(Data,1);
elseif strcmp(data,'Abnormal')
    S = load('CAFUC-abnormal3.mat');
    Data = S.data;
    fprintf('Loaded MAT data:size = [%d %d]\n',size(Data,1),size(Data,2));
    
    delay = 7;delay=7; dtype='Mean'; delt=1; delx=1;hwy='day';hwylength=size(Data,1);
elseif strcmp(data,'Monthly_2018') 
    Data1 = dlmread(strcat('month\',num2str(month),'-01.txt')); 
    Data2 = dlmread(strcat('month\',num2str(month),'-02.txt')); 
    Data3 = dlmread(strcat('month\',num2str(month),'-03.txt'));
    Data4 = dlmread(strcat('month\',num2str(month),'-04.txt'));
    Data5 = dlmread(strcat('month\',num2str(month),'-05.txt'));
    Data6 = dlmread(strcat('month\',num2str(month),'-06.txt'));
    Data7 = dlmread(strcat('month\',num2str(month),'-07.txt'));
    Data8 = dlmread(strcat('month\',num2str(month),'-08.txt'));
    Data9 = dlmread(strcat('month\',num2str(month),'-09.txt'));
    Data10 = dlmread(strcat('month\',num2str(month),'-10.txt'));
    Data11 = dlmread(strcat('month\',num2str(month),'-11.txt'));
    Data12 = dlmread(strcat('month\',num2str(month),'-12.txt'));
    Data = [Data1,Data2,Data3,Data4,Data5,Data6,Data7,Data8,Data9,Data10,Data11,Data12];
    delay=24; dtype='hour'; delt=1; hwy='2018hour'; hwylength=8928; xpath='x121.txt'; ypath='y121.txt'; 
end
toc
%% Compute KMD and Sort Modes
disp('Computing KMD via Hankel-DMD...')
tic
Avg=mean(Data,2);% Compute and Store Time Average
[eigval,Modes1,bo] = H_DMD(Data-repmat(Avg,1,size(Data,2)),delay); 
toc
disp('Sorting Modes...')
tic
% Sampling Frequency of PM2.5 Data is 5 hour.
% scatter(real(diag(eigval)),imag(diag(eigval))) 
% aa=real(diag(eigval));   eigenvalues of real
% bb=imag(diag(eigval));   imaginary of eigenvalue
% cc = real(log(diag(eigval)))   %logarithm
% dd = imag(log(diag(eigval)))   %logarithm
omega=log(diag(eigval))./delt;   % Compute Cont. Time Eigenvalues
Freal=imag(omega)./(2*pi);    % Compute Frequency
[T,Im]=sort((1./Freal),'descend');    % Sort Frequencies
omega=omega(Im); Modes1=Modes1(:,Im); bo=bo(Im);    % Sort Modes
toc

%% Compute and Plot Modes 
disp('Computing and Plotting Modes...')
tic
[nbx,nbt]=size(Data); % Get Data Size
time=(0:nbt-1)*delt;% Specify Time Interval
Psi=zeros(nbx,nbt,mode2-mode1+1);
res=[]
for i=mode1:mode2 % Loop Through all Modes to Plot.
psi=zeros(1,nbt);% Preallocate Time Evolution of Mode.
omeganow=omega(i);% Get Current Eigenvalue.
bnow=bo(i);% Get Current Amplitude Coefficient.
parfor t=1:length(time) 
psi(:,t)=exp(omeganow*time(t))*bnow; % Evolve for Time Length.
end
psi=Modes1(1:nbx,i)*psi;    % Compute Mode.
Psi(:,:,i)=psi;    % Store & Output Modes
m=abs(psi)
mag=mean(m(:)) %average amplitude
res=[res mag]
% csvwrite('period.csv',T)
% csvwrite('psi.csv',res) 
% -------------------------------------------------------------------------%
% -------------------------------------------------------------------------%
FONTSIZE = 35;
TICKSIZE = 28;
if strcmp(hwy,'day')   % Plot NGSIM Modes
[X,Y]=meshgrid(time./1,linspace(0,hwylength,nbx));     % Compute Mesh day.
% [X,Y]=meshgrid(time./30,linspace(0,hwylength,nbx));% Compute Mesh monthly.

h=figure
warning('off','MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame');	
jFrame = get(h,'JavaFrame');	
pause(0.3);					
set(jFrame,'Maximized',1);	
pause(0.5);					
warning('on','MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame');
aaaa=real(psi)

s1=surfc(X,Y,real(psi));% Generate Surface Plot
set(s1,'LineStyle','none')% No Lines

set(gca,'position',[0.1,0.15,0.60,0.78],'TickLabelInterpreter','latex','linewidth',2.5,'FontSize',30)
title(strcat('Mode #',num2str(i)),... 
                     'Interpreter','Latex','FontSize',30)
xlabel(' ','Interpreter','tex','FontSize',30,'rotation',13); 
h=colorbar;
ylabel('Monitoring station','rotation',-25,'position',[-600 -320],'FontSize',30);

if strcmp(dtype,'Mean')
set(get(h,'title'),'string',{' '},'FontSize',30);
elseif strcmp(dtype,'Mean')
set(get(h,'title'),'string', {'��g/m^{3} per hour'});
end

end %End Modes to Plot Loop
toc
disp('All Done')

end %End function
