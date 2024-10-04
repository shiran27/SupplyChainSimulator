clear all
close all
clc

netId = 1;               % Supply chain network ID
numOfChains = 1;         % Number of parallel chains
numOfInventories = 3;    % Each chain has 3 inventories

% Create Network
net = Network(netId, numOfChains, numOfInventories);

% Set up the figure
figNum = 1;
figure(figNum);
hold on; axis equal;
net.draw(1); 

% Simulation loop
for t = 2:1:10
    net.update();  % Update the entire network

    % Clear the figure, then hold on to plot without erasing previous state
    % clf; 
    % hold on; axis equal;

    figNum = t;
    figure(figNum);
    hold on; axis equal;
    
    % Draw the network state after each update with the current time
    net.draw(t);  % Pass the current time to the draw function
    
    
    % pause(0.1);  % Pause to visualize each step
    
end