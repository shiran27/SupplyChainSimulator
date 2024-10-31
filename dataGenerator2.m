function J = objectiveFunction(d, net, matFile)
    % d is a vector of [d1, d2, d3, d4]
    
    % Load grid search data from matFile
    load(matFile, 'LNorm', 'KNorm', 'gammaTilde', 'KLinks', 'pValRange', 'deltaCostCoefRange', 'gammaCostCoefRange', 'comCostLimitRange');
    
    % Extract best parameters using current d1, d2, d3, d4
    [pVal, deltaCostCoef, gammaCostCoef, comCostLimit] = net.findBestParameters(matFile, d(1), d(2), d(3), d(4));

    % Apply the found best parameters
    totalStatusLocal = 1;
    totalLNorm = 0;
    for i = 1:1:net.numOfChains
        [statusLocal, ~, ~, LNormVal, ~, ~] = net.chains{i}.localControlDesign(pVal, deltaCostCoef);
        if ~statusLocal
            totalStatusLocal = 0;
            break;
        else
            totalLNorm = totalLNorm + LNormVal;
        end
    end
    
    totalLNorm = totalLNorm / net.numOfChains;
    
    % Perform the global design if local design succeeded
    if totalStatusLocal
        [statusGlobal, gammaTildeVal, ~, ~, KNormVal] = net.globalControlDesign(gammaCostCoef, comCostLimit);
    end
    
    % Compute the performance metric (mean consensus error)
    net.setAdjointMatrixAndNeighborsAndControllers(); % Load graph, neighbors, and controllers from KVal
    net.update();  % Update the entire network
    J = net.cumMeanAbsConError;  % Consensus error as objective function
    
    % Penalize if local/global design fails
    if ~totalStatusLocal || ~statusGlobal
        J = inf;
    end
end
    