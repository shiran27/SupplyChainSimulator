%% Basic Example: Case 1
clear all
close all
clc

rng(7)
numOfChains = 2            % Number of parallel chains
numOfInventories = 3       % Number of inventories in each chain

% Create Network
netId = 1;               % Supply chain network ID
net = Network(netId, numOfChains, numOfInventories);
save('tempNet1.mat')

clear all
close all
clc

folderName = 'Results/Case1/'
load('tempNet1.mat')
rng(7)
net.gridSearchCompleteDesign(folderName);


%% Basic Example: Case 2
clear all
close all
clc

rng(7)
numOfChains = 3 %4        % Number of parallel chains
numOfInventories = 4 %5   % Each chain has 5 inventories

% Create Network
netId = 1;               % Supply chain network ID
net = Network(netId, numOfChains, numOfInventories);
save('tempNet2.mat')

clear all
close all
clc

folderName = 'Results/Case2/'
load('tempNet2.mat')
rng(7)
net.gridSearchCompleteDesign(folderName);


%% Basic Example: Case 3
clear all
close all
clc

rng(7)
numOfChains = 4            % Number of parallel chains
numOfInventories = 5     % Number of inventories in each chain

% Create Network
netId = 1;               % Supply chain network ID
net = Network(netId, numOfChains, numOfInventories);
save('tempNet3.mat');

clear all
close all
clc

folderName = 'Results/Case3/'
load('tempNet3.mat')
rng(7)
net.gridSearchCompleteDesign(folderName);