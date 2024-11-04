function J = dataGenerator2(d)
    % d is a vector of [d1, d2, d3, d4]
    % d
    isSoft = 1;
    load('tempNet2.mat','net','tMax')
    matFile = 'Results/Case2/gridSearchCompleteDesignResults.mat';
    rng(7)
    % d

    % Extract best parameters using current d1, d2, d3, d4
    % [pVal, deltaCostCoef, gammaCostCoef, comCostLimit] = net.findBestParameters(matFile, d(1), d(2), d(3), d(4))

    pVal = d(1);
    deltaCostCoef = d(2);
    gammaCostCoef = d(3);
    comCostLimit = d(4);

    % Apply the found best parameters
    totalStatusLocal = 1;
    for i = 1:1:net.numOfChains
        [statusLocal, ~, ~, ~, ~, ~] = net.chains{i}.localControlDesign(pVal, deltaCostCoef);
        if ~statusLocal
            totalStatusLocal = 0;
            break;
        end
    end
    
    % Perform the global design if local design succeeded
    if totalStatusLocal
        [statusGlobal, ~, ~, ~, ~] = net.globalControlDesign(gammaCostCoef, comCostLimit, isSoft);
        % Compute the performance metric (mean consensus error)
        net.setAdjointMatrixAndNeighborsAndControllers(); % Load graph, neighbors, and controllers from KVal
    else
        statusGlobal = 0;
    end
    
    
    
    if statusGlobal
        for t = 1:1:(tMax-1)
            net.update();  % Update the entire network
        end
        J = net.cumMeanAbsConError;
    else
        J = 10000;
    end
    
end