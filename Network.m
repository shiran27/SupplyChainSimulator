classdef Network < handle
    % This class represents supply chain networks (SCNs)

    properties  %% Network class properties
        netId               % ID Number
        chains              % Cell array to store chain objects
        numOfChains         % Number of chains in the network
        numOfInventories

        KGlobal   % Controller gain matrix (block matrix NxN, each block is nxn)

        % Important Matrices
        BCurl
        DCurl
        ECurl
        E_ii
        E_ij
        adjMatRef   % Prefered Adjacency Matrix
        adjMat      % Adjacency matrix
        costMat       % Com Cost Matrix

        cumMeanAbsTraError     % Cumulative moving average error across the entire network
        cumMeanAbsConError

        numUpdates         % Number of updates at the network level
        numDays
        numWeeks

        consensusErrorHistory
        trackingErrorHistory
    end

    methods

        function obj = Network(netID, numOfChains, numOfInventories)
            disp('Started creating the supply chain network.');
            obj.netId = netID;
            obj.numOfInventories = numOfInventories;

            % Create cell array for chains
            chains = cell(1, numOfChains);  % Use a cell array for chain objects
            
            % Creating each chain and storing in the cell array
            for i = 1:numOfChains
                chain = Chain(i, numOfInventories, numOfChains);  % Create each Chain object
                chains{i} = chain;  % Store the Chain in the cell array
            end
            
            % Assign the chains to the network
            obj.chains = chains;
            obj.numOfChains = numOfChains;

            % Adjoint matrix of the communication network:
            % obj.generateSparseAdj();  
            obj.generateDistanceBasedAdj(200);  % 200 Load an initial graph Adj matrix (reference)
            
            % Load all matrices
            obj.computeNetworkMatrices();

            gammaCostCoef = 10^6;
            comCostLimit = 0.01;
            isSoft = 1;
            obj.globalControlDesign(gammaCostCoef,comCostLimit,isSoft); % Design and Load KGlobal
            obj.setAdjointMatrixAndNeighborsAndControllers(); % Load Graph, neighbors and controllers from KGlobal

            obj.numUpdates = 0;
            obj.numDays = 0;
            obj.numWeeks = 0;
          
            obj.cumMeanAbsTraError = 0;  % Start with 0 average
            obj.cumMeanAbsConError = 0;  % Start with 0 average

            disp('Finished creating the supply chain network.');
        end

        function update(obj)

            if mod(obj.numUpdates, 24) == 0 % new day
                dayNum = floor(obj.numUpdates/24);
                dayOfWeek = mod(dayNum,7) + 1;
                obj.updateDemanders(dayOfWeek); % update the demander behaviors
                
                obj.numDays = dayNum;
                obj.numWeeks = floor(dayNum/7);


                if dayNum == 10    % once every three days, a random physical link fails
                    chainId = randi([1,obj.numOfChains]);
                    linkId = randi([1,obj.numOfInventories]);
                    obj.chains{chainId}.phyLinks{linkId}.delayBuffer = 0*obj.chains{chainId}.phyLinks{linkId}.delayBuffer;

                    chainId = randi([1,obj.numOfChains]);
                    linkId = randi([1,obj.numOfInventories]);
                    obj.chains{chainId}.phyLinks{linkId}.delayBuffer = 0*obj.chains{chainId}.phyLinks{linkId}.delayBuffer;

                    % chainId = randi([1,obj.numOfChains]);
                    % linkId = randi([1,obj.numOfInventories]);
                    % obj.chains{chainId}.phyLinks{linkId}.delayBuffer = 0*obj.chains{chainId}.phyLinks{linkId}.delayBuffer;
                end

                if dayNum == 20   % once every three days, a random physical link fails
                    chainId = randi([1,obj.numOfChains]);
                    invenId = randi([1,obj.numOfInventories]);
                    obj.chains{chainId}.inventories{invenId}.state = 0;

                    chainId = randi([1,obj.numOfChains]);
                    invenId = randi([1,obj.numOfInventories]);
                    obj.chains{chainId}.inventories{invenId}.state = 0;

                    chainId = randi([1,obj.numOfChains]);
                    invenId = randi([1,obj.numOfInventories]);
                    obj.chains{chainId}.inventories{invenId}.state = 0;

                    chainId = randi([1,obj.numOfChains]);
                    invenId = randi([1,obj.numOfInventories]);
                    obj.chains{chainId}.inventories{invenId}.state = 0;
                end

            end

            % Load tracking and consensus errors at the inventories
            obj.computeErrors(); % Inside this, we will have: obj.numUpdates = obj.numUpdates + 1;    % Update the number of updates

            % Execute updates
            for i = 1:1:obj.numOfChains
                obj.chains{i}.update();
            end
            
        end

        function updateDemanders(obj,dayOfWeek)
            for i = 1:1:obj.numOfChains
                newMean = obj.chains{i}.demander.dailyMeans(dayOfWeek);    % Mean demand rate
                newVariance = 0.2*newMean;                                              % Variance in demand
                obj.chains{i}.demander.demRateMean = newMean;
                obj.chains{i}.demander.demRateStd = newVariance;
                % obj.chains{i}.computeChainMatrices();
            end
            % obj.computeNetworkMatrices();
        end
        
        function computeErrors(obj)
            obj.computeConsensusError(); 

            globalTraErrorSum = 0;
            globalConErrorSum = 0;

            % Load tracking errors at the inventories
            for i = 1:1:obj.numOfChains
                obj.chains{i}.computeErrors();
            end

            for i = 1:1:obj.numOfChains
                for k = 1:1:obj.chains{i}.numOfInventories
                    globalTraErrorSum = globalTraErrorSum + abs(obj.chains{i}.inventories{k}.trackingError);
                    globalConErrorSum = globalConErrorSum + abs(obj.chains{i}.inventories{k}.consensusError);
                end
            end
            obj.numUpdates = obj.numUpdates + 1; % Update the number of updates

            % netVal = globalConErrorSum/obj.numOfChains

            % Compute global cumulative average error
            obj.cumMeanAbsTraError = ((obj.numUpdates-1)*obj.cumMeanAbsTraError + (globalTraErrorSum/(obj.numOfChains*obj.chains{1}.numOfInventories)))/obj.numUpdates;
            obj.cumMeanAbsConError = ((obj.numUpdates-1)*obj.cumMeanAbsConError + (globalConErrorSum/(obj.numOfChains*obj.chains{1}.numOfInventories)))/obj.numUpdates;

            % obj.cumMeanAbsTraError
            % obj.cumMeanAbsConError

        end

        % Compute the inventory consensus error and update cumulative moving average
        function computeConsensusError(obj)
            N = obj.numOfChains;
            n = obj.chains{1}.numOfInventories;
            for i = 1:N
                z_i = zeros(n,1);
                for j = 1:N
                    z_ij = [];
                    for k = 1:n
                        y_ik = obj.chains{i}.inventories{k}.state - obj.chains{i}.inventories{k}.refLevel;
                        y_jk = obj.chains{j}.inventories{k}.state - obj.chains{j}.inventories{k}.refLevel;
                        z_ij = [z_ij; (y_ik - y_jk)];
                    end
                    z_i = z_i + z_ij;
                end
                z_i = z_i/N;
                % Passing down consensus errors to inventory level
                for k = 1:n
                    obj.chains{i}.inventories{k}.consensusError = z_i(k);
                end
            end            
        end

        function computeNetworkMatrices(obj)
            
            N = obj.numOfChains;
            n = obj.chains{1}.numOfInventories;

            BCurl = [];
            DCurl = [];
            ECurl = [];
            
            for i = 1:1:N

                BCurl_i = obj.chains{i}.BCurl;
                DCurl_i = obj.chains{i}.DCurl;
                BCurl = blkdiag(BCurl, BCurl_i);
                DCurl = blkdiag(DCurl, DCurl_i);

                E_i = [];
                for j = 1:1:N
                    if i==j
                        E_ij = (1-(1/N))*eye(n);
                    else
                        E_ij = -(1/N)*eye(n);
                    end
                    E_i = [E_i, E_ij];
                end
                ECurl = [ECurl; E_i];

            end
            
            obj.BCurl = BCurl;
            obj.DCurl = DCurl;
            obj.ECurl = ECurl;

            obj.E_ii = (1-(1/N))*eye(n);
            obj.E_ij = -(1/N)*eye(n);

        end


        function [statusGlobal, gammaTildeVal, comCost, conCost, KNorm] = globalControlDesign(obj, gammaCostCoef, comCostLimit, isSoft)

            gammaBar = 250;
            epsilon = 0.000001; % Minimum value
            debugMode = 0;

            N = obj.numOfChains;
            n = obj.chains{1}.numOfInventories;
            
            %% Gedding B, D, E Matrices
            B = obj.BCurl;
            D = obj.DCurl;
            E = obj.ECurl;

            adjMat = obj.adjMatRef;        % Adjacency matrix of the preffered com topology
            costMat = obj.costMat;      % Communication cost matrix
            
            %% Variables 
            gammaTilde = sdpvar(1, 1,'full');

            P = sdpvar(N, N, 'diagonal');
            
            KShadow = sdpvar(n*N, n*N, 'full');
            
            I_n = eye(n);
            X_p_11 = [];
            X_12 = [];
            X_p_22 = [];
            X_11 = [];
            PBlock = [];
            nT = 0;            
            for i = 1:1:N
                ni = size(obj.chains{i}.ACurl,1);
                nT = nT + ni;
                I_ni = eye(ni);
                
                nu_i = obj.chains{i}.nu;
                rho_i = obj.chains{i}.rho;
            
                X_i_11 = -nu_i*I_ni;    %inputs x inputs
                X_i_12 = 0.5*eye(ni,n);      %inputs x outputs
                X_i_22 = -rho_i*I_n;   %outputs x outputs
            
                X_p_11 = blkdiag(X_p_11, P(i,i)*X_i_11);
                X_p_22 = blkdiag(X_p_22, P(i,i)*X_i_22);
                X_12 = blkdiag(X_12, X_i_11\X_i_12);

                X_11 = blkdiag(X_11, X_i_11);
                PBlock = blkdiag(PBlock, P(i,i)*I_n);
            end
            X_21 = X_12';

            L_eta_y = X_11 * B * KShadow;      % Note that M_eta_y = B*K and L_eta_y = X_p^11*M_eta_y; and M_eta_y and L_eta_y have the same structure, and M_eta_y =  
         
            %% Constraints 
            constraints = [];
            constraintTags = {}; % Cell array to hold tags
            constraintMats = {}; % Cell array to hold matrices
            
            % p_i > 0, \forall i
            for i = 1:N
                tagName = ['P_',num2str(i),num2str(i)];
                constraintTags{end+1} = tagName;
                con1 = tag(P(i,i) >= epsilon, tagName);
                constraintMats{end+1} = P(i,i);
                constraints = [constraints, con1];
            end

            % 0 <= gammaTilde <= gammaBar
            tagName = ['gammaTilde_low'];
            constraintTags{end+1} = tagName;
            con2_1 = tag(gammaTilde >= epsilon, tagName);
            constraintMats{end+1} = gammaTilde;
            
            tagName = ['gammaTilde_high'];
            constraintTags{end+1} = tagName;
            con2_2 = tag(gammaTilde <= gammaBar, tagName);
            constraintMats{end+1} = gammaTilde;
            
            constraints = [constraints, con2_1, con2_2];
            
            % Main constraint
            O_nT_nN = zeros(nT,n*N);
            I_nN = eye(n*N);
            O_nN = zeros(n*N);
            
            Mat_11 = X_p_11; 
            Mat_1 = [Mat_11];

            Mat_22 = I_nN;
            Mat_12 = O_nT_nN;
            Mat_2 = [Mat_1, Mat_12;
                    Mat_12', Mat_22];
          
            Mat_33 = -L_eta_y'*X_12 - X_21*L_eta_y - X_p_22;
            Mat_13 = [  L_eta_y;  
                        E         ];
            Mat_3 = [Mat_2, Mat_13;
                    Mat_13', Mat_33];
            
            Mat_44 = gammaTilde*I_nN;
            Mat_14 = [  X_p_11*D;
                        O_nN;
                        -X_21*X_p_11*D  ];
            Mat_4 = [Mat_3, Mat_14;
                    Mat_14', Mat_44];
            
            W = Mat_4;
            
            tagName = ['W'];
            constraintTags{end+1} = tagName;
            con3 = tag(W >= epsilon*eye(size(W)), tagName);
            constraintMats{end+1} = W;
            
            constraints = [constraints, con3];
            
            if ~isSoft
                % Follow strictly the given communication tpology (by A_ij adjacency matrix)
                tagName = ['K_Topology'];
                constraintTags{end+1} = tagName;
                con4 = tag(KShadow.*(adjMat==0) == O_nN, tagName);      % Graph structure : hard constraint
                constraintMats{end+1} = KShadow;
            
                constraints = [constraints, con4];
            else 
                tagName = ['Min_Topology_Cost'];
                constraintTags{end+1} = tagName;
                con5 = tag(sum(sum(KShadow.*costMat)) >= comCostLimit, tagName);      % Graph structure : hard constraint
                constraintMats{end+1} = KShadow;
    
                constraints = [constraints, con5];
            end

            % Consensus constraint
            tagName = ['K_Consensus'];
            constraintTags{end+1} = tagName;
            con6 = tag(KShadow*ones(n*N,1)==zeros(n*N,1), tagName);      % Graph structure : hard constraint
            constraintMats{end+1} = KShadow*ones(n*N,1);

            constraints = [constraints, con6];
            
            
            %% Objective Function
            costFun0 = sum(sum(abs(KShadow).*costMat)); %% This is very interesting idea, think about this
            % costFun0 = norm(KShadow.*costMat,2); %% This is very interesting idea, think about this

            if ~isSoft
                % Total Cost Function
                costFunction = 1*costFun0 + gammaCostCoef*(gammaTilde/gammaBar) + 1*trace(P) + 0;
            else
                % Total Cost Function
                costFunction = 1*costFun0 + gammaCostCoef*(gammaTilde/gammaBar) + 1*trace(P); % soft %%% Play with this
            end
            
            %% Solve the LMI problem
            
            solverOptions = sdpsettings('solver', 'mosek', 'verbose', 0, 'debug', 0);
            
            sol = optimize(constraints,costFunction,solverOptions);
            
            statusGlobal = sol.problem == 0;  
            % debugMode = ~statusGlobal;
            
            %% Extract variable values
            
            PBlockVal = value(PBlock);
            gammaTildeVal = value(gammaTilde);
            comCost = value(costFun0);
            conCost = value(gammaCostCoef*(gammaTilde/gammaBar));

            KShadowVal = value(KShadow); %% check this! 
            
            KVal = PBlockVal\KShadowVal;

            KNorm = norm(KVal);

            % Load KVal elements into a cell structure K{i,j} (i.e., partitioning KVal
            % into N\times N blocks)
            factorVal = 0.001;
            maxNorm = max(abs(KVal(:)));
            KVal(abs(KVal)<factorVal*maxNorm) = 0;
            % if ~isSoft
            %     KVal(adjMat==0) = 0;
            % else
            %     KVal(abs(KVal)<factorVal*maxNorm) = 0;
            % end

            obj.KGlobal = KVal;
            
            %% Debugging            
            if debugMode
            
                feasibility = check(constraints);
                
                % Combine tags and feasibility into one array for sorting
                combinedList = [feasibility(:), (1:length(feasibility))'];
                
                % Sort based on the first column (feasibility values)
                sortedList = sortrows(combinedList, 1);  % Sort by feasibility, ascending order
                
                % Printing
                for i = 1:length(sortedList)
                    idx = sortedList(i, 2);  % Get the original index of the constraint
                    if feasibility(idx) < -1e-6
                        disp(['Constraint "', constraintTags{idx}, '" is violated by ', num2str(feasibility(idx)), ' .']);
                        W_val = value(constraintMats{idx});
                        for j = 1:size(W_val, 1)
                            submatrix = W_val(1:j, 1:j);  % Extract principal minor
                            if det(submatrix) < 0
                                disp(['Principal minor ', num2str(j), ' is not positive semi-definite.']);
                            end
                        end
                    else
                        disp(['Constraint "', constraintTags{idx}, '" is satisfied by ',num2str(feasibility(idx)),' .']);
                        W_val = value(constraintMats{idx});
                    end
                end            
            end
        end


        function gridSearchPassivityIndices(obj)
            % GRIDSEARCHPASSIVITYINDICES - Performs a grid search over nu, rho, and gammaCost in logarithmic scales.
            %
            % This function evaluates global control design performance for different values of
            % nu (negative passivity index), rho (positive passivity index), and gammaCost (cost parameter).
            % The ranges are selected in logarithmic scales.
            
            saveMode = 1;

            % Define the ranges in the log domain
            nuRange = -logspace(-5, 5, 22);          % nu from -1 to -10000 (log spaced)
            rhoRange = logspace(-5, 5, 22);         % rho from 0.01 to 10000 (log spaced)
            gammaCostRange = logspace(-10, 10, 21);     % gammaCost from 1 to 1000000 (log spaced)
            
            % Preallocate result matrices for each gammaCost
            numNu = length(nuRange);
            numRho = length(rhoRange);
            numGammaCostRange = length(gammaCostRange);
            [gammaTilde, comCost, conCost, KNorm] = deal(nan(numNu, numRho, numGammaCostRange));  % Result matrices
            
            for gIdx = 1:length(gammaCostRange)
                gammaCostCoef = gammaCostRange(gIdx);
                disp(['Evaluating gammaCost = ', num2str(gammaCostCoef)]);
        
                for i = 1:numNu
                    nu = nuRange(i);
                    for j = 1:numRho
                        rho = rhoRange(j);
                        
                        for chainIndex = 1:1:obj.numOfChains
                            obj.chains{chainIndex}.nu = nu;
                            obj.chains{chainIndex}.rho = rho;
                        end

                        % Evaluate the global control design for the current grid point
                        comCostLimit = 0.01;
                        isSoft = 1;
                        [statusGlobal, gammaTildeVal, comCostVal, conCostVal, KNormVal] = obj.globalControlDesign(gammaCostCoef, comCostLimit, isSoft);
                        
                        % Only store results if the global control design is feasible (statusGlobal == 1)
                        if statusGlobal == 1
                            gammaTilde(i, j, gIdx) = gammaTildeVal;
                            comCost(i, j, gIdx) = comCostVal;
                            conCost(i, j, gIdx) = conCostVal;
                            KNorm(i, j, gIdx) = KNormVal;
                        end
                    end
                end
                
                % Plot results for the current gammaCost
                fig = figure;
                sgtitle(['Results for \gammaCost = ', num2str(gammaCostCoef)]);
                
                subplot(2,2,1);
                contourf(nuRange, rhoRange, gammaTilde(:,:,gIdx), 'LineColor', 'none');
                set(gca, 'XScale', 'log', 'YScale', 'log');  % Set axes to log scale
                title('Gamma Tilde');
                xlabel('\nu'); ylabel('\rho'); colorbar;
                
                subplot(2,2,2);
                contourf(nuRange, rhoRange, comCost(:,:,gIdx), 'LineColor', 'none');
                set(gca, 'XScale', 'log', 'YScale', 'log');
                title('Communication Cost');
                xlabel('\nu'); ylabel('\rho'); colorbar;
                
                subplot(2,2,3);
                contourf(nuRange, rhoRange, conCost(:,:,gIdx), 'LineColor', 'none');
                set(gca, 'XScale', 'log', 'YScale', 'log');
                title('Control Cost');
                xlabel('\nu'); ylabel('\rho'); colorbar;
                
                subplot(2,2,4);
                contourf(nuRange, rhoRange, KNorm(:,:,gIdx), 'LineColor', 'none');
                set(gca, 'XScale', 'log', 'YScale', 'log');
                title('K Norm');
                xlabel('\nu'); ylabel('\rho'); colorbar;

                if saveMode
                    % Save figure as .fig and .png
                    figFileName = sprintf('Results/gridSearchPassivity_gCost_%d.fig', gIdx);
                    pngFileName = sprintf('Results/gridSearchPassivity_gCost_%d.png', gIdx);
                    
                    savefig(fig, figFileName);  % Save as .fig file
                    print(fig, pngFileName, '-dpng', '-r300');  % Save as high-res PNG (300 dpi)
                    close(fig);  % Close the figure after saving to save memory                
                end
            end

            if saveMode
                save('Results/gridSearchPassivityResults.mat', 'nuRange', 'rhoRange', 'gammaCostRange', 'gammaTilde', 'comCost', 'conCost', 'KNorm');
                disp('Results saved to "results/gridSearchPassivityResults.mat".');
            end
        end


        function gridSearchCompleteDesign(obj, folderName, netFileName)
            % GRIDSEARCHCOMPLETEDESIGN - Performs a grid search over pVal, deltaCostCoef, gammaCostCoef, and comCostLimit.
            %
            % This function evaluates control design performance for different values of
            % pVal (passivity index), deltaCostCoef, gammaCostCoef, and comCostLimit. 
            % The ranges are selected in logarithmic scales.
        
            % Define the ranges in the log domain
            pValRange = logspace(-2, 1, 7);  % pVal values in logarithmic scale
            deltaCostCoefRange = logspace(-3, 6, 4);  % deltaCostCoef range
            gammaCostCoefRange = logspace(-2, 4, 4); % gammaCostCoef range
            comCostLimitRange = logspace(-4, -1, 4); % comCostLimit range
        
            % Preallocate result vectors
            numP = length(pValRange);
            numDelta = length(deltaCostCoefRange);
            numGamma = length(gammaCostCoefRange);
            numComCost = length(comCostLimitRange);
        
            [LNorm, KNorm, gammaTilde, KLinks, JConVals, JTraVals] = deal(nan(numGamma, numDelta, numP, numComCost));  % Result vectors

            isSoft = 1;
        
            for gIdx = 1:numGamma
                gammaCostCoef = gammaCostCoefRange(gIdx);
                disp(['Evaluating gammaCostCoef = ', num2str(gammaCostCoef)]);
        
                for comIdx = 1:numComCost
                    comCostLimit = comCostLimitRange(comIdx);
                    disp(['Evaluating comCostLimit = ', num2str(comCostLimit)]);
        
                    for dIdx = 1:numDelta
                        deltaCostCoef = deltaCostCoefRange(dIdx);
                        disp(['Evaluating deltaCostCoef = ', num2str(deltaCostCoef)]);
        
                        for pIdx = 1:numP
                            pVal = pValRange(pIdx);
                            disp(['Evaluating pVal = ', num2str(pVal)]);
        
                            load(netFileName,'net','tMax')
                            rng(7)

                            % Execute local designs
                            totalLNorm = 0;
                            totalStatusLocal = 1;
                            for i = 1:1:obj.numOfChains
                                [statusLocal, ~, ~, LNormVal, ~, ~] = net.chains{i}.localControlDesign(pVal, deltaCostCoef);
                                if ~statusLocal
                                    totalStatusLocal = 0;
                                    break;
                                else
                                    totalLNorm = totalLNorm + LNormVal;
                                end
                            end
                            totalLNorm = totalLNorm / obj.numOfChains;
        
                            % Execute global designs if local designs were successful
                            if totalStatusLocal
                                [statusGlobal, gammaTildeVal, ~, ~, KNormVal] = net.globalControlDesign(gammaCostCoef, comCostLimit, isSoft);
                                % Store results if the global control design is feasible (statusGlobal == 1)
                                if statusGlobal == 1
                                    LNorm(gIdx, dIdx, pIdx, comIdx) = totalLNorm;
                                    KNorm(gIdx, dIdx, pIdx, comIdx) = KNormVal;
                                    gammaTilde(gIdx, dIdx, pIdx, comIdx) = gammaTildeVal;
                                    KLinks(gIdx, dIdx, pIdx, comIdx) = sum(sum(obj.KGlobal > 0));
                                    for t = 1:1:(tMax-1)
                                        net.update();  % Update the entire network
                                    end
                                    JConVals(gIdx, dIdx, pIdx, comIdx) = net.cumMeanAbsConError;
                                    JTraVals(gIdx, dIdx, pIdx, comIdx) = net.cumMeanAbsTraError;
                                    netVals{gIdx, dIdx, pIdx, comIdx} = net;
                                end
                            end
                        end
                    end
        
                    % try

                        % Generate and save contourf plots for the current gammaCostCoef and comCostLimit
                        fig = figure;
                        sgtitle(['Results for \gammaCostCoef = ', num2str(gammaCostCoef), ', comCostLimit = ', num2str(comCostLimit)]);
            
                        % Plot 1: L Norm as a contour plot
                        subplot(3, 2, 1);
                        contourf(pValRange, deltaCostCoefRange, squeeze(LNorm(gIdx, :, :, comIdx)), 'LineColor', 'none');
                        set(gca, 'XScale', 'log', 'YScale', 'log');
                        title('L Norm');
                        xlabel('pVal'); ylabel('\deltaCostCoef'); colorbar;
            
                        % Plot 2: K Norm as a contour plot
                        subplot(3, 2, 2);
                        contourf(pValRange, deltaCostCoefRange, squeeze(KNorm(gIdx, :, :, comIdx)), 'LineColor', 'none');
                        set(gca, 'XScale', 'log', 'YScale', 'log');
                        title('K Norm');
                        xlabel('pVal'); ylabel('\deltaCostCoef'); colorbar;
            
                        % Plot 3: Gamma Tilde as a contour plot
                        subplot(3, 2, 3);
                        contourf(pValRange, deltaCostCoefRange, squeeze(gammaTilde(gIdx, :, :, comIdx)), 'LineColor', 'none');
                        set(gca, 'XScale', 'log', 'YScale', 'log');
                        title('Gamma Tilde');
                        xlabel('pVal'); ylabel('\deltaCostCoef'); colorbar;
    
                        % Plot 4: KLinks as a contour plot
                        subplot(3, 2, 4);
                        contourf(pValRange, deltaCostCoefRange, squeeze(KLinks(gIdx, :, :, comIdx)), 'LineColor', 'none');
                        set(gca, 'XScale', 'log', 'YScale', 'log');
                        title('K Links');
                        xlabel('pVal'); ylabel('\deltaCostCoef'); colorbar;
            
                        % Plot 5: Consensus Error as a contour plot
                        subplot(3, 2, 5);
                        contourf(pValRange, deltaCostCoefRange, squeeze(JConVals(gIdx, :, :, comIdx)), 'LineColor', 'none');
                        set(gca, 'XScale', 'log', 'YScale', 'log');
                        title('Consensus Error');
                        xlabel('pVal'); ylabel('\deltaCostCoef'); colorbar;
    
                        % Plot 6: Tracking Error as a contour plot
                        subplot(3, 2, 6);
                        contourf(pValRange, deltaCostCoefRange, squeeze(JTraVals(gIdx, :, :, comIdx)), 'LineColor', 'none');
                        set(gca, 'XScale', 'log', 'YScale', 'log');
                        title('Tracking Error');
                        xlabel('pVal'); ylabel('\deltaCostCoef'); colorbar;
                        % Save figure as .fig and .png
                        
                        % figFileName = sprintf([folderName,'gridSearchComplete_gCost_%d_comCost_%d.fig'], gIdx, comIdx);
                        pngFileName = sprintf([folderName,'gridSearchComplete_gCost_%d_comCost_%d.png'], gIdx, comIdx);
                        
                        % savefig(fig, figFileName);  % Save as .fig file
                        print(fig, pngFileName, '-dpng', '-r300');  % Save as high-res PNG (300 dpi)
                        close(fig);  % Close the figure after saving
                        
                    % catch
                    % 
                    % end

                    % Save the results into a .mat file
                    save([folderName,'gridSearchCompleteDesignResults.mat'], 'pValRange', 'deltaCostCoefRange', 'gammaCostCoefRange', 'comCostLimitRange', 'LNorm', 'KNorm', 'gammaTilde','KLinks','JConVals','JTraVals','netVals');
                end
            end
        
           
            
            disp('Results and figures saved to the "results" folder.');
        end


        function [bestPVal, bestDeltaCostCoef, bestGammaCostCoef, bestComCostLimit] = findBestParameters(obj, matFileName, d1, d2, d3, d4, d5, d6)

            % Load the data from the specified .mat file
            data = load(matFileName);
        
            % Extract the variables from the loaded data
            LNorm = data.LNorm;
            KNorm = data.KNorm;
            gammaTilde = data.gammaTilde;
            KLinks = data.KLinks;
            JCon = data.JConVals;
            JTra = data.JTraVals;

            pValRange = data.pValRange;
            deltaCostCoefRange = data.deltaCostCoefRange;
            gammaCostCoefRange = data.gammaCostCoefRange;
            comCostLimitRange = data.comCostLimitRange;
            
        
            % Compute the minimum and maximum values for normalization
            minLNorm = min(LNorm(:), [], 'omitnan');
            maxLNorm = max(LNorm(:), [], 'omitnan');
            minKNorm = min(KNorm(:), [], 'omitnan');
            maxKNorm = max(KNorm(:), [], 'omitnan');
            minGammaTilde = min(gammaTilde(:), [], 'omitnan');
            maxGammaTilde = max(gammaTilde(:), [], 'omitnan');
            minKLinks = min(KLinks(:), [], 'omitnan');
            maxKLinks = max(KLinks(:), [], 'omitnan');
            minJCon = min(JCon(:), [], 'omitnan');
            maxJCon = max(JCon(:), [], 'omitnan');
            minJTra = min(JTra(:), [], 'omitnan');
            maxJTra = max(JTra(:), [], 'omitnan');
            

            % Compute normalization coefficients
            c1 = 1 / (maxLNorm - minLNorm);
            c2 = 1 / (maxKNorm - minKNorm);
            c3 = 1 / (maxGammaTilde - minGammaTilde);
            c4 = 1 / (maxKLinks - minKLinks);
            c5 = 1 / (maxJCon - minJCon);
            c6 = 1 / (maxJTra - minJTra);
        
            % Initialize variables to store the best values
            bestObjective = inf;
            
            bestLNorm = NaN;
            bestKNorm = NaN;
            bestGammaTilde = NaN;
            bestKLinks = NaN;
            bestJCon = NaN;
            bestJTra = NaN;

            bestPVal = NaN;
            bestDeltaCostCoef = NaN;
            bestGammaCostCoef = NaN;
            bestComCostLimit = NaN;
            
        
            % Loop over all parameter combinations and compute the objective
            for gIdx = 1:length(gammaCostCoefRange)
                for comIdx = 1:length(comCostLimitRange)
                    for dIdx = 1:length(deltaCostCoefRange)
                        for pIdx = 1:length(pValRange)
                            % Extract the current values of LNorm, KNorm, and gammaTilde
                            LNormVal = LNorm(gIdx, dIdx, pIdx, comIdx);
                            KNormVal = KNorm(gIdx, dIdx, pIdx, comIdx);
                            gammaTildeVal = gammaTilde(gIdx, dIdx, pIdx, comIdx);
                            KLinksVal = KLinks(gIdx, dIdx, pIdx, comIdx);
                            JConVal = JCon(gIdx, dIdx, pIdx, comIdx);
                            JTraVal = JTra(gIdx, dIdx, pIdx, comIdx);

                            % Check if the current combination is valid (not NaN)
                            if ~isnan(LNormVal) && ~isnan(KNormVal) && ~isnan(gammaTildeVal) && ~isnan(KLinksVal) && ~isnan(JConVal) && ~isnan(JTraVal)
                                % Normalize each value

                                if any(isinf([c1,c2,c3,c4,c5,c6]))
                                    disp('Error in c:');
                                    disp([c1,c2,c3,c4,c5,c6]);
                                    c4 = 1/maxKLinks;
                                end

                                normalizedLNorm = (LNormVal - minLNorm) * c1;
                                normalizedKNorm = (KNormVal - minKNorm) * c2;
                                normalizedGammaTilde = (gammaTildeVal - minGammaTilde) * c3;
                                normalizedKLinks = (KLinksVal - minKLinks)*c4;
                                normalizedJCon = (JConVal - minJCon)*c5;
                                normalizedJTra = (JTraVal - minJTra)*c6;

                                % Compute the objective function using the new weights d1, d2, d3
                                objective = d1 * normalizedLNorm + d2 * normalizedKNorm + d3 * normalizedGammaTilde + d4*normalizedKLinks + d5*normalizedJCon + d6*normalizedJTra;
        
                                % Check if this is the best (minimum) objective found so far
                                if objective < bestObjective
                                    bestObjective = objective;
                                    bestLNorm = LNormVal;
                                    bestKNorm = KNormVal;
                                    bestGammaTilde = gammaTildeVal;
                                    bestKLinks = KLinksVal;
                                    bestJCon = JConVal;
                                    bestJTra = JTraVal;

                                    bestPVal = pValRange(pIdx);
                                    bestDeltaCostCoef = deltaCostCoefRange(dIdx);
                                    bestGammaCostCoef = gammaCostCoefRange(gIdx);
                                    bestComCostLimit = comCostLimitRange(comIdx);
                                end
                            end
                        end
                    end
                end
            end
        
            % Display the best parameter combination and the corresponding objective value
            disp('Best parameter combination found:');
            disp(['Minimum objective value: ', num2str(bestObjective)]);
            disp(['pVal: ', num2str(bestPVal)]);
            disp(['deltaCostCoef: ', num2str(bestDeltaCostCoef)]);
            disp(['gammaCostCoef: ', num2str(bestGammaCostCoef)]);
            disp(['comCostLimit: ', num2str(bestComCostLimit)]);
            disp(['LNorm: ', num2str(bestLNorm)]);
            disp(['KNorm: ', num2str(bestKNorm)]);
            disp(['gammaTilde: ', num2str(bestGammaTilde)]);
            disp(['KLinks: ', num2str(bestKLinks)]);
            disp(['JCon: ', num2str(bestJCon)]);
            disp(['JTra: ', num2str(bestJTra)]);
            
            % % Optionally, save the results to a .mat file
            % save('Results/bestParameters.mat', 'bestPVal', 'bestDeltaCostCoef', 'bestGammaCostCoef', 'bestComCostLimit', 'bestObjective', 'bestLNorm', 'bestKNorm', 'bestGammaTilde');
            % disp('Best parameters saved to "results/bestParameters.mat".');
        end 


        % Generate a sparse K matrix for closest neighboring inventories
        function generateSparseAdj(obj)
            N = obj.numOfChains;  % Number of chains
            n = obj.chains{1}.numOfInventories;  % Number of inventories per chain
            
            % Initialize the K matrix with zeros
            adjMat = zeros(N * n, N * n);
            
            % Iterate over each chain
            for i = 1:N
                for j = 1:N
                    % Within the same chain (i == j)
                    if i == j
                        % Link each inventory with its immediate neighbors
                        for k = 1:n
                            adjMat((i-1)*n + k, (i-1)*n + k) = 1;
                            if k > 1  % Connect to the previous inventory
                                adjMat((i-1)*n + k, (i-1)*n + k - 1) = 1;  % Random scaling factor for backward link
                            end
                            if k < n  % Connect to the next inventory
                                adjMat((i-1)*n + k, (i-1)*n + k + 1) = 1;  % Random scaling factor for forward link
                            end
                        end
                    else
                        % Link inventories between adjacent chains (i != j)
                        if abs(i - j) == 1  % Only link adjacent chains
                            for k = 1:n
                                % Create cross-chain links (e.g., inventory k in chain i to inventory k in chain j)
                                adjMat((i-1)*n + k, (j-1)*n + k) = 1;  % Random scaling factor for cross-chain link
                            end
                        end
                    end
                end
            end
            obj.adjMat = adjMat;
        end


        function generateDistanceBasedAdj(obj, distanceThreshold)
            N = obj.numOfChains;  % Number of chains
            n = obj.chains{1}.numOfInventories;  % Number of inventories per chain
            
            % Initialize the K matrix with zeros
            adjMat = zeros(N * n, N * n);
            
            % Iterate over each chain
            for i = 1:N
                for j = 1:N
                    % Loop through all inventories in chain i and j
                    for inven_i = 1:n
                        for inven_j = 1:n
                            % Get the locations of the inventories
                            loc_i = obj.chains{i}.inventories{inven_i}.location;
                            loc_j = obj.chains{j}.inventories{inven_j}.location;
                            
                            % Calculate the Euclidean distance between the two inventories
                            distance = norm(loc_i - loc_j);
                            
                            % If the distance is within the threshold, assign a random value to K
                            if distance <= distanceThreshold
                                adjMat((i-1)*n + inven_i, (j-1)*n + inven_j) = 1;  % Random scaling factor
                                costMat((i-1)*n + inven_i, (j-1)*n + inven_j) = 1*distance;
                            else
                                adjMat((i-1)*n + inven_i, (j-1)*n + inven_j) = 0;  % No connection if too far
                                costMat((i-1)*n + inven_i, (j-1)*n + inven_j) = 1*distance + 10*(distance-distanceThreshold)^1;
                            end
                        end
                    end
                end
            end
            obj.adjMatRef = adjMat;
            obj.costMat = costMat/max(max(costMat));
        end


        % Function to set in and out neighbors based on the K matrix
        function setAdjointMatrixAndNeighborsAndControllers(obj)
            N = obj.numOfChains;  % Number of chains
            n = obj.chains{1}.numOfInventories;  % Number of inventories per chain
            K = obj.KGlobal;
                        
            % Clearing existing adjMat, neighbors and controller gains
            obj.adjMat = zeros(N * n, N * n);
            for i = 1:N
                for k = 1:n
                    obj.chains{i}.inventories{k}.inNeighbors = [];
                    obj.chains{i}.inventories{k}.KGlobal = [];
                end
            end

            % Loading new adjMat, neighbors and controller gains
            for i = 1:N
                for j = 1:N
                    % If K_ij is non-zero, there is a communication link between chains i and j
                    K_ij = K((i-1)*n+1:i*n, (j-1)*n+1:j*n);
                    if norm(K_ij) > 0
                        for k = 1:n
                            for l = 1:n
                                % Assign in-neighbors for inventories in chain i
                                K_ij_kl = K_ij(k,l);
                                if K_ij_kl ~= 0
                                    obj.adjMat((i-1)*n + k, (j-1)*n + l) = 1;
                                    obj.chains{i}.inventories{k}.inNeighbors = [obj.chains{i}.inventories{k}.inNeighbors, obj.chains{j}.inventories{l}];
                                    obj.chains{i}.inventories{k}.KGlobal = [obj.chains{i}.inventories{k}.KGlobal, K_ij_kl];
                                end
                            end
                        end
                    end
                end
            end
        end


        function plotConnections(obj, adjMat)
            % Number of chains and inventories
            N = obj.numOfChains;
            n = obj.chains{1}.numOfInventories;
        
            % Loop through each chain and plot inventories as dots
            for chainIdx = 1:N
                for invenIdx = 1:n
                    % Get the location of the inventory
                    inventory = obj.chains{chainIdx}.inventories{invenIdx};
                    plot(inventory.location(1), inventory.location(2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', [0.7,1,0.7]);  % Plot inventory as black dot
                end
            end

            for chainIdx = 1:N
                inventory = obj.chains{chainIdx}.inventories{1};
                text(inventory.location(1) - 25, inventory.location(2), ...
                            sprintf('i=%d', chainIdx), 'HorizontalAlignment', 'center','FontSize',8);
            end

            for invenIdx = 1:n
                inventory = obj.chains{1}.inventories{invenIdx};
                    text(inventory.location(1), inventory.location(2) - 25, ...
                        sprintf('k=%d', invenIdx), 'HorizontalAlignment', 'center','FontSize',8);
            end
        
            % Parameters for drawing arcs
            arcHeight = 15;  % Adjust height of arcs
        
            % Loop through the adjacency matrix and plot arcs or straight arrows for connections
            for i = 1:N
                for j = 1:N
                    for inven_i = 1:n
                        for inven_j = 1:n
                            % Check if there's a connection between the inventories
                            if adjMat((i - 1) * n + inven_i, (j - 1) * n + inven_j) ~= 0
                                % Get the locations of the source and target inventories
                                srcLocation = obj.chains{i}.inventories{inven_i}.location;
                                destLocation = obj.chains{j}.inventories{inven_j}.location;
        
                                % Check if the source and destination are within the same chain
                                if i == j
                                    continue
                                    % Inside the same chain, check if inventories are separated
                                    if abs(inven_i - inven_j) > 1
                                        % Draw a curved arc for non-adjacent inventories
                                        midPoint = (srcLocation + destLocation) / 2;
        
                                        % Determine if the arc should curve up or down based on inventory indices
                                        if inven_i < inven_j
                                            % Forward arc (curving upward)
                                            arcMidPoint = midPoint + [0, arcHeight];
                                        else
                                            % Backward arc (curving downward)
                                            arcMidPoint = midPoint - [0, arcHeight];
                                        end
        
                                        % Draw the arc using quadratic Bezier curve
                                        t = linspace(0, 1, 100);  % Parameter for the curve
                                        curveX = (1 - t).^2 * srcLocation(1) + 2 * (1 - t) .* t * arcMidPoint(1) + t.^2 * destLocation(1);
                                        curveY = (1 - t).^2 * srcLocation(2) + 2 * (1 - t) .* t * arcMidPoint(2) + t.^2 * destLocation(2);
        
                                        % Plot the arc
                                        plot(curveX, curveY, 'r', 'LineWidth', 1);

                                        % Find the midpoint index for the arrow placement
                                        midIdx = floor(length(t) / 2);
                                        
                                        % Calculate the direction of the arrow at the midpoint
                                        dx = curveX(midIdx + 1) - curveX(midIdx);
                                        dy = curveY(midIdx + 1) - curveY(midIdx);
                                        
                                        % Normalize the direction vector for consistent arrow size
                                        arrowLength = 3;  % Adjust this value for longer or shorter arrows
                                        normFactor = sqrt(dx^2 + dy^2);
                                        dx = dx / normFactor * arrowLength;
                                        dy = dy / normFactor * arrowLength;
                                        
                                        % Draw a custom arrowhead at the midpoint using a filled triangle
                                        fill([curveX(midIdx) - dy, curveX(midIdx) + dy, curveX(midIdx) + dx], ...
                                             [curveY(midIdx) + dx, curveY(midIdx) - dx, curveY(midIdx) + dy], 'r', 'EdgeColor', 'r');
                                    else
                                        % For adjacent inventories, draw a straight arrow
                                        quiver(srcLocation(1), srcLocation(2), ...
                                               destLocation(1) - srcLocation(1), destLocation(2) - srcLocation(2), ...
                                               'MaxHeadSize', 0.2, 'Color', 'r', 'LineWidth', 1);
                                    end
                                else
                                    % Between different chains, draw a straight arrow
                                    quiver(srcLocation(1), srcLocation(2), ...
                                           destLocation(1) - srcLocation(1), destLocation(2) - srcLocation(2), ...
                                           'MaxHeadSize', 0.2, 'Color', 'r', 'LineWidth', 1);
                                end
                            end
                        end
                    end
                end
            end

            sizeVal = 35;
            xLim0 = obj.chains{1}.inventories{1}.location(1) - sizeVal/2;
            xLim = obj.chains{end}.inventories{end}.location(1) + sizeVal/2;
            yLim0 = obj.chains{1}.inventories{1}.location(2) - sizeVal/2;  % Lower y limit to include the text within bounds
            yLim = obj.chains{end}.inventories{end}.location(2) + sizeVal/2;

            axis([xLim0,xLim,yLim0,yLim])
        end

        function plotNetworkPerformance(obj)

            n = obj.chains{1}.numOfInventories;  % Number of inventories per chain
            N = obj.numOfChains;                 % Number of chains
        
            % --------- Compute and Plot Performance Metrics ---------
            % Compute average consensus error and tracking error
            L = length(obj.chains{1}.inventories{1}.consensusErrorHistory);

            totalConsensusError = zeros(1,L);
            totalTrackingError = zeros(1,L);
            for i = 1:N  % Loop over all chains
                for k = 1:n  % Loop over all inventories
                    totalConsensusError = totalConsensusError + abs(obj.chains{i}.inventories{k}.consensusErrorHistory);
                    totalTrackingError = totalTrackingError + abs(obj.chains{i}.inventories{k}.trackingErrorHistory);
                end
            end
            
            % Compute average errors
            avgConsensusError = totalConsensusError / (N * n) ;
            avgTrackingError = totalTrackingError / (N * n);
    
            % Store the performance metrics
            consensusErrorHistory = avgConsensusError * (100/500);
            trackingErrorHistory = avgTrackingError * (100/500);

            obj.consensusErrorHistory = consensusErrorHistory;
            obj.trackingErrorHistory = trackingErrorHistory;
            
            % Plot the accumulated error profiles
            plot(1:L, consensusErrorHistory, 'r.-', 'LineWidth', 1, 'DisplayName', 'Consensus Error');
            plot(1:L, trackingErrorHistory, 'b.-', 'LineWidth', 1, 'DisplayName', 'Tracking Error');
            
            % Update plot title, labels, and legend
            % title(sprintf('Performance Metrics at Time Step %d', t));
            xlabel('Time Step (Hours)');
            ylabel('Mean Percentage Absolute Error');
            legend('Location', 'northeast');
            
        end

        function plotDemandProfiles(obj)
            L = length(obj.chains{1}.demander.demandHistory);
            N = obj.numOfChains;                 % Number of chains
            hArray = [];
            for i = 1:N  % Loop over all chains
                h = plot(1:L, obj.chains{i}.demander.demandHistory, 'DisplayName', ['Demand ',num2str(i)]);
                hArray = [hArray, h];
                color1 = get(h, 'Color');
                plot(1:L, mean(obj.chains{i}.demander.dailyMeans)*ones(1,L), '--', 'Color', color1, 'LineWidth', 0.5)
            end
            xlabel('Time Step (Hours)');
            ylabel('Demand Value');
            legend(hArray, 'Location', 'southwest');
        end


        function plotMeanDemandProfiles(obj)
            N = obj.numOfChains;
            L = length(obj.chains{1}.demander.dailyMeans);
            hArray = [];
            for i = 1:N  % Loop over all chains
                h = stairs(1:L, obj.chains{i}.demander.dailyMeans, 'DisplayName', ['Demand ',num2str(i)]);
                hArray = [hArray, h];
                color1 = get(h, 'Color');
                plot(1:L, mean(obj.chains{i}.demander.dailyMeans)*ones(1,L), '--', 'Color', color1, 'LineWidth', 0.5)
            end
            xlabel('Day of the Week');
            ylabel('Mean Demand Value');
            legend(hArray, 'Location', 'southwest');
        end


        function runSimulationAndSaveVideos(obj, fileTag, tMax)
            % Define video file names using the provided fileTag
            networkVideoFileName = [fileTag, '_NetworkSimulation.avi'];  
            performanceVideoFileName = [fileTag, '_PerformanceMetrics.avi'];
            
            % Create VideoWriter objects for both videos
            networkVideo = VideoWriter(networkVideoFileName);  % Video for the network simulation
            networkVideo.FrameRate = 10;  % Set frame rate for network video
            open(networkVideo);  % Open the network video for writing
            
            performanceVideo = VideoWriter(performanceVideoFileName);  % Video for the performance metrics
            performanceVideo.FrameRate = 10;  % Set frame rate for performance video
            open(performanceVideo);  % Open the performance video for writing
            
             % Get screen size to center the figures
            screenSize = get(0, 'ScreenSize');
            screenWidth = screenSize(3);
            screenHeight = screenSize(4);

             % Calculate centered positions
            figureWidth = 600;  % Width of each figure
            figureHeight = 400;  % Height of each figure
            xOffset = (screenWidth - 2 * figureWidth) / 2;  % X offset to center both figures
            yOffset = (screenHeight - figureHeight) / 2;  % Y offset to vertically center
            
            % Set up two separate figures, centered on the screen
            figNetwork = figure('Visible', 'on', 'Position', [xOffset, yOffset, figureWidth, figureHeight]);  % Left figure for network
            figPerformance = figure('Visible', 'on', 'Position', [xOffset + figureWidth, yOffset, figureWidth, figureHeight]);  % Right figure for performance metrics
            
            % Number of chains and inventories
            n = obj.chains{1}.numOfInventories;
            N = obj.numOfChains;
            
            % Initialize arrays to store the performance history
            consensusErrorHistory = nan(tMax, 1);
            trackingErrorHistory = nan(tMax, 1);
        
            % Simulation loop
            for t = 1:tMax
                % --------- Update Network ---------
                obj.update();  % Update the entire network
        
                % --------- Draw Network State ---------
                figure(figNetwork);  % Bring the network figure into focus
                clf; 
                hold on; axis equal; axis off;
                obj.draw(t);  % Pass the current time to the draw function
                
                % Capture the frame and write to the network video
                frameNetwork = getframe(figNetwork);
                writeVideo(networkVideo, frameNetwork);
                
                % --------- Compute and Plot Performance Metrics ---------
                figure(figPerformance);  % Bring the performance figure into focus
                clf;  % Clear the figure for updated plotting
                hold on; grid on;
                
                % Compute average consensus error and tracking error
                totalConsensusError = 0;
                totalTrackingError = 0;
                for i = 1:N  % Loop over all chains
                    for k = 1:n  % Loop over all inventories
                        totalConsensusError = totalConsensusError + abs(obj.chains{i}.inventories{k}.consensusErrorHistory(t));
                        totalTrackingError = totalTrackingError + abs(obj.chains{i}.inventories{k}.trackingErrorHistory(t));
                    end
                end
                
                % Compute average errors
                avgConsensusError = totalConsensusError / (N * n) ;
                avgTrackingError = totalTrackingError / (N * n);
        
                % Store the performance metrics
                consensusErrorHistory(t) = avgConsensusError * (100/500);
                trackingErrorHistory(t) = avgTrackingError * (100/500);
                
                % Plot the accumulated error profiles
                plot(1:t, consensusErrorHistory(1:t), 'r.-', 'LineWidth', 1, 'DisplayName', 'Consensus Error');
                plot(1:t, trackingErrorHistory(1:t), 'b.-', 'LineWidth', 1, 'DisplayName', 'Tracking Error');
                
                % Update plot title, labels, and legend
                % title(sprintf('Performance Metrics at Time Step %d', t));
                xlabel('Time Step (Hours)');
                ylabel('Mean Percentage Absolute Error');
                legend('Location', 'northeast');
                
                % Capture the frame and write to the performance video
                framePerformance = getframe(figPerformance);
                writeVideo(performanceVideo, framePerformance);
            end
            
            % Close the video files
            close(networkVideo);
            close(performanceVideo);
        
            % Display completion message
            disp(['Network simulation video saved as ', networkVideoFileName]);
            disp(['Performance metrics video saved as ', performanceVideoFileName]);
        end

        function saveInitialFinalStates(obj, fileTag, tMax)
            % Define file names for the initial and final state images
            initialNetworkFileName = [fileTag, '_InitialNetworkState.png']; 
            finalNetworkFileName = [fileTag, '_FinalNetworkState.png'];  
            finalPerformanceFileName = [fileTag, '_FinalPerformanceState.png'];
            finalDemandFileName = [fileTag, '_FinalDemand.png'];

            % Get screen size to center the figures
            screenSize = get(0, 'ScreenSize');
            screenWidth = screenSize(3);
            screenHeight = screenSize(4);
            
            % Calculate centered positions
            figureWidth = 600;  % Width of each figure
            figureHeight = 400;  % Height of each figure
            xOffset = (screenWidth - 3 * figureWidth) / 4;  % X offset to center both figures
            yOffset = (screenHeight - figureHeight) / 2;  % Y offset to vertically center
            
            % Set up two separate figures, centered on the screen
            figNetwork = figure('Visible', 'on', 'Position', [xOffset, yOffset, figureWidth, figureHeight]);  % Left figure for network
            figPerformance = figure('Visible', 'on', 'Position', [2*xOffset + figureWidth, yOffset, figureWidth, figureHeight]);  % Right figure for performance metrics
            figDemand = figure('Visible', 'on', 'Position', [3*xOffset + 2*figureWidth, yOffset, figureWidth, figureHeight]);  % Right figure for performance metrics
            

            % Number of chains and inventories
            n = obj.chains{1}.numOfInventories;
            N = obj.numOfChains;
            
            % Initialize arrays to store the performance history
            consensusErrorHistory = nan(tMax, 1);
            trackingErrorHistory = nan(tMax, 1);
        
            % --------- Network State ---------
            figure(figNetwork);
            clf; hold on; axis equal; axis off;
            obj.draw(1);  % Draw the initial state of the network at t=1
            % % Set font size for all text in the figure
            % set(figNetwork, 'Units', 'Inches', 'Position', [1, 1, 3, 3]);
            % set(gca, 'FontSize', 8);
            % Adjust figure size for tight layout to remove extra spaces
            % set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02));
            print(figNetwork, initialNetworkFileName, '-dpng', '-r300');  % Save initial network state as high-res PNG
            figFileName = [initialNetworkFileName(1:end-4), '.fig'];
            savefig(figNetwork, figFileName);  % Save as .fig file

            % --------- Simulate the System ---------
            for t = 1:tMax
                obj.update();  % Update the entire network
                
                % Collect performance metrics at each time step
                totalConsensusError = 0;
                totalTrackingError = 0;
                for i = 1:N
                    for k = 1:n
                        totalConsensusError = totalConsensusError + abs(obj.chains{i}.inventories{k}.consensusErrorHistory(t));
                        totalTrackingError = totalTrackingError + abs(obj.chains{i}.inventories{k}.trackingErrorHistory(t));
                    end
                end
                
                % Compute average errors for the current state
                avgConsensusError = totalConsensusError / (N * n);
                avgTrackingError = totalTrackingError / (N * n);
                
                % Store errors in the history
                consensusErrorHistory(t) = avgConsensusError * (100/500);
                trackingErrorHistory(t) = avgTrackingError * (100/500);
            end
            
            % Both following methods gives the same result.
            % cumMeanAbsTraError = obj.cumMeanAbsTraError * (100/500)
            % meanTrackingError = mean(trackingErrorHistory)
            

            % --------- Draw Final State ---------
            % --------- Network State ---------
            figure(figNetwork);
            clf; hold on; axis equal; axis off;
            obj.draw(tMax);  % Draw the final state of the network
            print(figNetwork, finalNetworkFileName, '-dpng', '-r300');  % Save final network state as high-res PNG
            figFileName = [finalNetworkFileName(1:end-4), '.fig'];
            savefig(figNetwork, figFileName);  % Save as .fig file

            % --------- Final Performance Metrics ---------
            figure(figPerformance);
            clf; hold on; grid on;
            plot(1:tMax, consensusErrorHistory, 'r.-', 'LineWidth', 1, 'DisplayName', 'Consensus Error');
            plot(1:tMax, trackingErrorHistory, 'b.-', 'LineWidth', 1, 'DisplayName', 'Tracking Error');
            % title(sprintf('Final Performance Metrics at Time Step %d', tMax));
            xlabel('Time Step (Hours)');
            ylabel('Mean Percentage Absolute Error');
            legend('Location', 'northeast');
            print(figPerformance, finalPerformanceFileName, '-dpng', '-r300');  % Save final performance state as high-res PNG
            figFileName = [finalPerformanceFileName(1:end-4), '.fig'];
            savefig(figPerformance, figFileName);  % Save as .fig file

            % Demands
            figure(figDemand);
            clf; hold on; grid on;
            for i = 1:N
                plot(1:tMax, obj.chains{i}.demander.demandHistory, '.-', 'LineWidth', 1, 'DisplayName', ['Demand ',num2str(i)]);
            end
            xlabel('Time Step (Hours)');
            ylabel('Demand Value');
            legend('Location', 'northeast');
            print(figDemand, finalDemandFileName, '-dpng', '-r300');  % Save final performance state as high-res PNG
            figFileName = [finalDemandFileName(1:end-4), '.fig'];
            savefig(figDemand, figFileName);  % Save as .fig file

            % Display completion message
            disp(['Initial and final network states saved as ', initialNetworkFileName, ' and ', finalNetworkFileName]);
            disp(['The final performance metrics saved as ', finalPerformanceFileName]);
            disp(['The final demand profile saved as ', finalDemandFileName]);

        end

        function printDefaultProperties(obj)
            fprintf('Supply Chain Network Default Parameters:\n\n');
            
            % Supply Class - Access the first instance in the network
            fprintf('--- Supply Class ---\n');
            fprintf('supRateMax: %f\n', obj.chains{1}.supplier.supRateMax);
            
            % Demand Class - Access the first instance in the network
            fprintf('\n--- Demand Class ---\n');
            disp('dailyMeans Obtained using: 100 + 20*randi([1,numOfChains],7,1) + 20*demId;')
            for i = 1:1:obj.numOfChains
                fprintf('Chain %f\n',i)
                fprintf('dailyMeans: ')
                disp(obj.chains{i}.demander.dailyMeans);
                fprintf('demRateMean (mean): %f\n', mean(obj.chains{i}.demander.dailyMeans));
                fprintf('demRateStd (0.1x): %f\n', 0.1*mean(obj.chains{i}.demander.dailyMeans));
            end
            
            % Inventory Class - Access the first inventory in the first chain
            
            fprintf('\n--- Inventory Class ---\n');
            fprintf('refLevel: %f\n', obj.chains{1}.inventories{1}.refLevel);
            fprintf('maxLevel: %f\n', obj.chains{1}.inventories{1}.maxLevel);
            fprintf('minLevel: %f\n', obj.chains{1}.inventories{1}.minLevel);
            fprintf('perishRate: %f\n', obj.chains{1}.inventories{1}.perishRate);
            disp('wasteRateMean Obtained using: 5 + randi([1,5])')
            disp('wasteRateStd Obtained using: 0.2*mean')
            for i = 1:1:obj.numOfChains
                for k = 1:1:obj.numOfInventories
                    fprintf('Chain %f',i)
                    fprintf(', Inventory %f\n',k)
                    fprintf('wasteRateMean: %f\n', obj.chains{i}.inventories{k}.wasteRateMean);
                    fprintf('wasteRateStd (0.2x): %f\n', obj.chains{i}.inventories{k}.wasteRateStd);
                end
            end
            
            % Physical Link Class - Access the first physical link in the first chain
            fprintf('\n--- Physical Link Class ---\n');
            disp('tranDelay Obtained using: 1 + randi([1,4])')
            disp('wasteRateMean Obtained using: 5 + randi([1,5])')
            disp('wasteRateStd Obtained using: 0.2xmean')
            for i = 1:1:obj.numOfChains
                for k = 1:1:obj.numOfInventories
                    fprintf('Chain %f',i)
                    fprintf(', Inventory %f\n',k)
                    fprintf('tranDelay: %d\n', obj.chains{i}.phyLinks{k}.tranDelay);
                    fprintf('wasteRateMean: %f\n', obj.chains{i}.phyLinks{k}.wasteRateMean);
                    fprintf('wasteRateStd (0.2x): %f\n', obj.chains{i}.phyLinks{k}.wasteRateStd);
                end
            end
            
        end
            
        
        function draw(obj, currentTime)
            % Draw each chain in the network
            for i = 1:1:obj.numOfChains
                obj.chains{i}.draw();
            end
            
            sizeVal = 35;
            
            % Set axis limits based on supplier and demander positions
            xLim0 = obj.chains{1}.supplier.location(1) - sizeVal/2 - 10;
            xLim = obj.chains{end}.demander.location(1) + sizeVal/2 + 10;
            yLim0 = obj.chains{1}.supplier.location(2) - sizeVal/2 - 10;  % Lower y limit to include the text within bounds
            yLim = obj.chains{end}.demander.location(2) + sizeVal/2 + 10;
        
             % Create a single text line for time, tracking error, and consensus error
            bottomText1 = sprintf('Time Steps: %d; Days: %d; Hours: %d', ...
                                 currentTime, obj.numDays, mod(currentTime, 24));
            bottomText2 = sprintf('Consensus PCMAE: %.2f%% || Tracking PCMAE: %.2f%%', ...
                                 obj.cumMeanAbsConError * (100/500), obj.cumMeanAbsTraError * (100/500));
                             
            % Place the combined text below the network
            text(xLim0 + 5, yLim0 - 20, bottomText1, 'FontSize', 8, 'Color', 'k', 'HorizontalAlignment', 'left');
            text(xLim0 + 5, yLim0 - 40, bottomText2, 'FontSize', 8, 'Color', 'r', 'HorizontalAlignment', 'left');

            % Set tight axis limits to avoid extra space
            axis([xLim0, xLim, yLim0-40, yLim]);

            % % Display the global cumulative average error on the plot with adjusted positions
            % timeText = ['Time Steps: ', num2str(currentTime), '; Days: ', num2str(obj.numDays), '; Hours: ', num2str(mod(currentTime,24)), '.'];
            % text(xLim0 + 10, yLim0 - 30, timeText, 'FontSize', 8, 'Color', 'k');  % Move closer to the plot
            % 
            % % Display tracking and consensus error percentages
            % errorTextTracking = ['Tracking PCMAE: ', num2str(obj.cumMeanAbsTraError * (100/500), 4), '%'];
            % text(xLim0 + 10, yLim0 - 50, errorTextTracking, 'FontSize', 8, 'Color', 'b');  % Adjust position within bounds
            % 
            % errorTextConsensus = ['Consensus PCMAE: ', num2str(obj.cumMeanAbsConError * (100/500), 4), '%'];
            % text(xLim0 + 10, yLim0 - 70, errorTextConsensus, 'FontSize', 8, 'Color', 'r');  % Adjust position within bounds
        
            
        end


        function saveFormattedFigure(obj, figHandle, folderPath, fileName, figWidth, figHeight, fontSize)
            % Ensure the target folder exists
            if ~exist(folderPath, 'dir')
                mkdir(folderPath);
            end
            
            % Adjust the figure size
            % set(figHandle, 'Units', 'inches');
            % set(figHandle, 'Position', [1, 1, figWidth, figHeight]);
        
            ax = gca;

            % Calculate the required figure height to maintain aspect ratio with axis equal
            % if figHeight == 0
            xRange = diff(ax.XLim);  % Width of the plotted content
            yRange = diff(ax.YLim);  % Height of the plotted content
            aspectRatio = yRange / xRange;
            figHeight = max(figWidth * aspectRatio + 0.1, figHeight);
            % end
            
            % Set the figure size based on desired width and calculated height
            figHandle = gcf;
            set(figHandle, 'Units', 'inches', 'Position', [1, 1, figWidth, figHeight]);
            set(gcf, 'PaperPositionMode', 'auto');

            % Set font size for all text in the figure
            set(findall(figHandle, '-property', 'FontSize'), 'FontSize', fontSize);
        

            % Adjust layout to remove whitespace
            % outerpos = ax.OuterPosition;
            % ti = ax.TightInset;
            % left = outerpos(1) + ti(1);
            % bottom = outerpos(2) + ti(2) + 0.1;
            % axWidth = outerpos(3) - ti(1) - ti(3);
            % axHeight = outerpos(4) - ti(2) - ti(4) - 0.1;
            % ax.Position = [left bottom axWidth axHeight];
            
            % Remove empty spaces
            set(figHandle, 'PaperPositionMode', 'auto');
            set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02));
        
            % Save in .png format with high resolution
            pngFilePath = fullfile(folderPath, [fileName, '.png']);
            print(figHandle, pngFilePath, '-dpng', '-r300');  % 300 DPI for high-res output
        
            % Save as .fig for further editing in MATLAB
            figFilePath = fullfile(folderPath, [fileName, '.fig']);
            savefig(figHandle, figFilePath);
        
            % Display save message
            disp(['Figure saved as ', pngFilePath, ' and ', figFilePath]);
        end


    end
end