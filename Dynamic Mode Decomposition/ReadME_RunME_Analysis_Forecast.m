% Dynamic Mode Decomposition 

%% Instructions for Koopman Modes.
% To generate Koopman modes call on the function:
% Modes=GenerateKoopmanModes(Data,Mode1,Mode2,Save)

%Inputs Required:
% Data is a string containing the name of the data set.

% Mode1 and Mode2 are integers indicating which modes to produce.
% Ordered by their period of oscilaliton from slowest to fastest.
% Mode1 can be < or = Mode2.

% Save is a logical (0 or 1) indicating the modes to be saved as jpeg's. 

% Examples:% Save is a logical (0 or 1) indicating the modes to be saved as jpeg's.


%Outputs Returned:
% Modes is an n by m by #modes sized  array. 
% For example Modes(:,:,i) contains the i'th mode.

%Plots Generated:
% The funciton will generate plots of the desired Koopman Modes.
% Examples:1
clc; clear variables; close all;


%Daily=GenerateKoopmanModes('Day_mean',34,34)
Daily=GenerateKoopmanModes('Abnormal',8,8)
%Mmonthly=GenerateKoopmanModes('Monthly_2018',1,20,2018);





