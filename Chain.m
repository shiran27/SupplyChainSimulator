classdef Chain < handle
    % This class represents supply chains

    properties %% Chain class properties
        chainId            % ID number
        
        supplier           % Supplier object
        inventories        % Array of Inventory objects
        numOfInventories   % Number of inventories
        numOfChains
        phyLinks           % Array of PhyLink objects (transport between inventories)
        demander           % Demand object (at the end of the chain)

        % Characteristic Subsystem Matrices
        A
        B
        ABar
        BBar
        CBar
        DBar
        ACurl
        BCurl
        DCurl
        CCurl
        LLocal      % Local Controller

        % Passivity Indices
        nu
        rho

        % Steady State Vectors
        xOverBar
        wOverBar
        xBarOverBar
        wBarOverBar
        dOverBar
        uOverBar        

        location           % Chain location (optional for plotting)

        cumMeanAbsTraError    % Cumulative moving average of the tracking error for the chain
        cumMeanAbsConError    % Cumulative moving average of the tracking error for the chain
        numUpdates         % Number of updates performed at the chain level
    end

    methods

        function obj = Chain(chainId, numOfInventories, numOfChains)

            disp(['Started creating chain ', num2str(chainId)])
            obj.chainId = chainId;
            obj.numOfChains = numOfChains;

            % Physical parameters (start and end points of the chain)
            obj.location = [0, 100*chainId; 100*(numOfInventories+1), 100*chainId]; 

            % Create a supplier
            sizeS = 35;
            obj.supplier = Supply(chainId, obj.location(1,:), sizeS);

            % Create inventories as a cell array (to store object references)
            sizeI = 35;
            invenList = cell(1, numOfInventories);
            for k = 1:numOfInventories
                inventory = Inventory(chainId, k, obj.location(1,2), sizeI);
                invenList{k} = inventory;
            end
            obj.inventories = invenList;
            obj.numOfInventories = numOfInventories;

            % Create a demander
            sizeD = 35;
            obj.demander = Demand(chainId, numOfChains, obj.location(2,:), sizeD);

            % Create phyLinks and assign them to the supplier, inventories, and demander
            phyLinks = cell(1, numOfInventories + 1);  % One link for each transition
            for k = 1:(numOfInventories+1)
                phyLinkId = numOfInventories * (chainId - 1) + k;
                if k == 1
                    % Link from supplier to the first inventory
                    location = [obj.supplier.location; obj.inventories{k}.location];
                    location(1,1) = location(1,1) + sizeS/2; 
                    location(2,1) = location(2,1) - sizeI/2; 
                    tranDelay = 1+randi([1,4]); %5
                    phyLink = PhyLink(phyLinkId, chainId, location, tranDelay, obj.supplier, obj.inventories{k}); 
                    % Assign phyLink to the first inventory's phyLinkIn and supplier's phyLinkOut
                    obj.inventories{k}.phyLinkIn = phyLink;
                    obj.supplier.phyLinkOut = phyLink;
                elseif k <= numOfInventories
                    % Link between two consecutive inventories
                    location = [obj.inventories{k-1}.location; obj.inventories{k}.location];
                    location(1,1) = location(1,1) + sizeI/2; 
                    location(2,1) = location(2,1) - sizeI/2; 
                    tranDelay = 1+randi([1,4]); %5
                    phyLink = PhyLink(phyLinkId, chainId, location, tranDelay, obj.inventories{k-1}, obj.inventories{k});
                    % Assign phyLink to the inventories' phyLinkIn and phyLinkOut
                    obj.inventories{k-1}.phyLinkOut = phyLink;  % Outgoing link for previous inventory
                    obj.inventories{k}.phyLinkIn = phyLink;     % Incoming link for current inventory
                else
                    % Link from the last inventory to the demander
                    location = [obj.inventories{k-1}.location; obj.demander.location];
                    location(1,1) = location(1,1) + sizeI/2; 
                    location(2,1) = location(2,1) - sizeD/2; 
                    tranDelay = 1;
                    phyLink = PhyLink(phyLinkId, chainId, location, tranDelay, obj.inventories{k-1}, obj.demander);
                    % Assign phyLink to the last inventory's phyLinkOut and demander's phyLinkIn
                    obj.inventories{k-1}.phyLinkOut = phyLink;
                    obj.demander.phyLinkIn = phyLink;
                end
                phyLinks{k} = phyLink;  % Store the PhyLink reference
            end 
            obj.phyLinks = phyLinks;

            % Load all matrices
            obj.computeChainMatrices();

            % obj.localPassivityAnalysis();
            pVal = 0.1;
            deltaCostCoef = 1;
            obj.localControlDesign(pVal,deltaCostCoef);
            
            obj.numUpdates = 0;
            obj.cumMeanAbsTraError = 0;  % Start with 0 average
            obj.cumMeanAbsConError = 0;  % Start with 0 average

            disp('Finished creating a chain...')
        end


        function update(obj)
            % Step 1: Demand generation
            obj.demander.generateDemand(obj.numUpdates); % Compute order in 

            % Step 2: Construct error state vector
            % chainID = obj.chainId
            x = [];
            xBar = [];
            for i = 1:1:obj.numOfInventories
                x = [x; obj.inventories{i}.state];
                xBar = [xBar; obj.phyLinks{i}.delayBuffer'];
            end
            xTilde = x - obj.xOverBar;
            xBarTilde = xBar - obj.xBarOverBar;
            
            % Step 3: Inventories compute orders local and global control
            % inputs (i.e., orderIn values)
            uTilde = obj.LLocal*[xTilde; xBarTilde];
            for i = 1:1:obj.numOfInventories
                obj.inventories{i}.uTilde = uTilde(i);
                obj.inventories{i}.computeOrder();  % Compute orderIn values based on uOverBar, uTilde and uTildeTilde
            end
            
            % Step 5: Supplier provides products based on the first inventory's request
            obj.supplier.supplyProducts(); % Compute product out
            
            % Step 6: Update state of all inventories and demander
            for i = 1:obj.numOfInventories
                obj.inventories{i}.updateState(obj.numUpdates);    % Compute product outs
            end
            
            % Step 7: Transport products through each PhyLink
            for i = 1:(obj.numOfInventories+1)
                obj.phyLinks{i}.transportGoods(obj.numUpdates); % Computer product ins at downstream
            end

            % Step 8: Update the demander to generate a new demand
            obj.demander.updateState();
     
        end


        function randomizeInitialState(obj)

            for i = 1:1:obj.numOfInventories
                obj.inventories{i}.state = randi([100,900]);
                obj.phyLinks{i}.delayBuffer = randi([100,900],1,obj.phyLinks{i}.tranDelay);
            end

        end


        function computeErrors(obj)
            % Step 2: Compute the tracking and consensus errors 
            chainTraErrorSum = 0;
            chainConErrorSum = 0;
            
            for k = 1:1:obj.numOfInventories
                obj.inventories{k}.computeErrors(); % (consensus errors already computed)
            end

            for k = 1:1:obj.numOfInventories
                chainTraErrorSum = chainTraErrorSum + abs(obj.inventories{k}.trackingError);  % Sum cumulative average tracking errors
                chainConErrorSum = chainConErrorSum + abs(obj.inventories{k}.consensusError);  % Sum cumulative average consensus errors
            end
            obj.numUpdates = obj.numUpdates + 1; % Update the number of updates
            
            % Compute chain's cumulative average error
            obj.cumMeanAbsTraError = ((obj.numUpdates-1)*obj.cumMeanAbsTraError + (chainTraErrorSum/obj.numOfInventories))/obj.numUpdates;
            obj.cumMeanAbsConError = ((obj.numUpdates-1)*obj.cumMeanAbsConError + (chainConErrorSum/obj.numOfInventories))/obj.numUpdates;

        end


        function computeChainMatrices(obj)
            n = obj.numOfInventories;

            % Setting matrix A
            rho_i = []; % Perish rate values
            for k = 1:1:n
                rho_ik = obj.inventories{k}.perishRate;
                rho_i = [rho_i, rho_ik];
            end
            A = diag(1-rho_i);
            obj.A = A;

            % Setting Matrix B
            B = [zeros(n-1,1), eye(n-1); 0, zeros(1,n-1)];
            obj.B = B;

            % Setting Matrices ABar, BBar and CBar
            tau_i = 0;
            ABar = [];
            BBar = [];
            CBar = [];
            DBar = [];
            for k = 1:1:n
                tau_ik = obj.phyLinks{k}.tranDelay;
                tau_i = tau_i + tau_ik;

                ABar_ik = [zeros(tau_ik-1,1), eye(tau_ik-1); 0, zeros(1,tau_ik-1)];
                ABar = blkdiag(ABar,ABar_ik);

                BBar_ik = [zeros(tau_ik-1,1);1];
                BBar = blkdiag(BBar,BBar_ik);

                CBar_ik = [1, zeros(1, tau_ik-1)];
                CBar = blkdiag(CBar,CBar_ik);

                DBar_ik = ones(tau_ik,1);
                DBar = blkdiag(DBar, DBar_ik);
            end
            obj.ABar = ABar;
            obj.BBar = BBar;
            obj.CBar = CBar;
            obj.DBar = DBar;

            % Setting Matrices ACurl, BCurl and CCurl
            obj.ACurl = [A, CBar; zeros(tau_i,n), ABar];
            obj.BCurl = [-B; BBar];
            obj.DCurl = [-eye(n); zeros(tau_i,n)];
            obj.CCurl = [eye(n), zeros(n, tau_i)];

            % % Old approach
            % realParts = linspace(0.1, 0.3, n + tau_i);  % Real parts of the poles
            % imagMagnitude = 0.9;  % Small magnitude for the imaginary part
            % 
            % % Create complex conjugate pairs
            % P = [];
            % for i = 1:2:length(realParts)
            %     if i < length(realParts)
            %         P = [P, realParts(i) + 1j * imagMagnitude, realParts(i) - 1j * imagMagnitude];
            %     else
            %         P = [P, realParts(i)];  % If odd number of poles, keep the last one real
            %     end
            % end
            % obj.LLocal = -place(obj.ACurl,obj.BCurl,P); % zeros(n, n+tau_i); % Placeholder

            % Computing Steady State Vectors
            xOverBar = [];
            wOverBar = [];
            wBarOverBar = [];
            for k = 1:1:n
                xOverBar_k = obj.inventories{k}.refLevel;
                xOverBar = [xOverBar; xOverBar_k];

                wOverBar_k = obj.inventories{k}.wasteRateMean;
                wOverBar = [wOverBar; wOverBar_k];

                wBarOverBar_k = obj.phyLinks{k}.wasteRateMean;
                wBarOverBar = [wBarOverBar; wBarOverBar_k];
            end 
            dOverBar = [zeros(n-1,1); obj.demander.demRateMean];
            
            uOverBar  = (eye(n)-B) \ ( (eye(n)-A)*xOverBar + wOverBar + wBarOverBar + dOverBar);
            xBarOverBar = DBar*uOverBar;
            
            obj.xOverBar = xOverBar;
            obj.wOverBar = wOverBar;
            obj.wBarOverBar = wBarOverBar;
            obj.dOverBar = dOverBar;
            obj.xBarOverBar = xBarOverBar; 
            obj.uOverBar = uOverBar;

            for k = 1:1:n
                obj.inventories{k}.uOverBar = uOverBar(k);
            end

        end

        


        function [statusLocal, nuVal, rhoVal, LNormVal, gammaTildeVal, deltaVal] = localControlDesign(obj,pVal,deltaCostCoef)

            N = obj.numOfChains;
            
            debugMode = 0;
            i = obj.chainId;
            n = obj.numOfInventories;
            epsilon = 0.000001; % Minimum value
            ni = size(obj.ACurl, 1);

            I_ni = eye(ni);

            P = sdpvar(ni, ni, 'symmetric');
            K = sdpvar(n, ni, 'full');
            nu = sdpvar(1,1,'full');
            rhoTilde = sdpvar(1,1,'full'); %rhoTilde = 1/rho

            C = eye(ni);
            X_22Tilde = rhoTilde*eye(ni);
            X_12 = 0.5*eye(ni);

            if ~isnan(pVal)
                gammaTilde = sdpvar(1,1,'full');
                % delta = sdpvar(1,1,'full');
            else
                % C = obj.CCurl;
                % X_22Tilde = rhoTilde*eye(n);
                % X_12 = 0.5*eye(ni,n); 
            end

            A = obj.ACurl;
            B = obj.BCurl;

            X_11 = -nu*eye(ni);

            X_21 = X_12';

            O_n_ni = zeros(n,ni);
            O_n_n = zeros(n,n);
            I_n = eye(n);
            O_ni_ni = zeros(ni);

            constraintTags = {}; % Cell array to hold tags
            constraintMats = {}; % Cell array to hold matrices
                        
            tagName = ['P_',num2str(i)];
            constraintTags{end+1} = tagName;
            con1 = tag(P >= epsilon*I_ni, tagName);
            constraintMats{end+1} = P;

            % To check passivity
            W = [X_22Tilde, O_ni_ni, C*P, O_ni_ni; 
                 O_ni_ni', P, A*P + B*K, I_ni;
                 P*C', P*A'+K'*B', P, P'*C'*X_21;
                 O_ni_ni', I_ni, X_12*C*P, X_11];

            if ~isnan(pVal)
                
            else
                % W = [X_22Tilde, O_n_ni, C*P, O_n_ni; 
                %  O_n_ni', P, A*P + B*K, I_ni;
                %  P*C', P*A'+K'*B', P, P'*C'*X_21;
                %  O_n_ni', I_ni, X_12*C*P, X_11];
                % W = [P, A*P + B*K;
                %  P*A'+K'*B', P];
            end

            tagName = ['W_',num2str(i)];
            constraintTags{end+1} = tagName;
            con2 = tag(W >= epsilon*eye(size(W)), tagName);
            constraintMats{end+1} = W;

            % To enforce nu >= 0 
            tagName = ['nu_',num2str(i)];
            constraintTags{end+1} = tagName;
            con3 = tag(nu <= -epsilon, tagName);
            constraintMats{end+1} = nu;

            % To enforce rho >= 0 
            tagName = ['rho_',num2str(i)];
            constraintTags{end+1} = tagName;
            con4 = tag(rhoTilde >= epsilon, tagName);
            constraintMats{end+1} = rhoTilde;

            if ~isnan(pVal)
                % To support global design
                W0 = [-pVal*nu, 0, 0, pVal*nu; 
                      0, 1, (1-(1/N))*rhoTilde, 0;
                      0, (1-(1/N))*rhoTilde, pVal*rhoTilde, 0.5*pVal*rhoTilde;
                      pVal*nu, 0, 0.5*pVal*rhoTilde, gammaTilde];
                tagName = ['Global_Helper_',num2str(i)];
                constraintTags{end+1} = tagName;
                con5 = tag(W0 >= epsilon*eye(size(W0)), tagName);
                constraintMats{end+1} = W0;
            
                % To enforce delta >= 0 
                % tagName = ['delta_',num2str(i)];
                % constraintTags{end+1} = tagName;
                % con6 = tag(delta >= epsilon, tagName);
                % constraintMats{end+1} = delta;
            end


            % magnitude bounds
            tagName = ['nu_',num2str(i)];
            constraintTags{end+1} = tagName;
            con7 = tag(nu >= -10^5, tagName);
            constraintMats{end+1} = nu;

            % magnitude bounds
            tagName = ['rho_',num2str(i)];
            constraintTags{end+1} = tagName;
            con8 = tag(rhoTilde <= 10^5, tagName);
            constraintMats{end+1} = rhoTilde;


            if ~isnan(pVal)
                constraints = [con1, con2, con3, con4, con5, con7, con8];
                % costFunction = 1*gammaTilde + deltaCostCoef*delta + 1*trace(P) + 0*norm(K);
                % costFunction = 1*gammaTilde + 1*trace(P);
                costFunction = 1*gammaTilde + deltaCostCoef*trace(P);
            else
                constraints = [con1, con2, con3, con4, con7, con8];
                % constraints = [con1, con2];
                % costFunction = -1*nu + 1*rhoTilde + deltaCostCoef*trace(P);
                costFunction = -1*nu + 1*rhoTilde + 0.01*trace(P);
            end

            solverOptions = sdpsettings('solver', 'mosek', 'verbose', 0, 'debug', 0);
            
            sol = optimize(constraints, costFunction, solverOptions);

            statusLocal = sol.problem == 0;
            % debugMode = ~statusLocal;
            
            PVal = value(P);
            KVal = value(K);
            LVal = KVal/PVal;
            nuVal = value(nu);
            rhoVal = 1/value(rhoTilde);
            LNormVal = norm(LVal);
            if ~isnan(pVal)
                gammaTildeVal = value(gammaTilde);
                deltaVal = 0;
            else
                gammaTildeVal = 0;
                deltaVal = 0;
            end

            % Final Step
            obj.LLocal = LVal;
            obj.nu = nuVal;
            obj.rho = rhoVal;

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


        function gridSearchLocalControlDesignParameter(obj)
            
            deltaCostCoef = 1000;
            pValRange = logspace(-5, 5, 1000);  % Example range of pVal values in logarithmic scale

            % Preallocate result vectors
            numP = length(pValRange);
            [nuVal, rhoVal, LNormVal, gammaTildeVal, deltaVal] = deal(nan(1, numP));  % Result vectors
            
            for i = 1:numP
                i
                pVal = pValRange(i)
                
                % Evaluate the local control design for the current pVal
                [statusLocal, nu, rho, LNorm, gammaTilde, delta] = obj.localControlDesign(pVal,deltaCostCoef);
                
                % Only store results if the local control design is feasible (statusLocal == 1)
                if statusLocal == 1
                    nuVal(i) = nu;
                    rhoVal(i) = rho;
                    LNormVal(i) = LNorm;
                    gammaTildeVal(i) = gammaTilde;
                    deltaVal(i) = delta;
                end
            end
            
            % Plot the results
            fig = figure;
    
            subplot(3, 2, 1);
            plot(pValRange, nuVal, '.-', 'LineWidth', 2);
            title('\nu vs pVal');
            xlabel('pVal'); ylabel('\nu');
            set(gca, 'XScale', 'log');  % Set x-axis to log scale
            grid on;
            
            subplot(3, 2, 2);
            plot(pValRange, rhoVal, '.-', 'LineWidth', 2);
            title('\rho vs pVal');
            xlabel('pVal'); ylabel('\rho');
            set(gca, 'XScale', 'log');  % Set x-axis to log scale
            grid on;
            
            subplot(3, 2, 3);
            plot(pValRange, LNormVal, '.-', 'LineWidth', 2);
            title('L Norm vs pVal');
            xlabel('pVal'); ylabel('L Norm');
            set(gca, 'XScale', 'log');  % Set x-axis to log scale
            grid on;
            
            subplot(3, 2, 4);
            plot(pValRange, gammaTildeVal, '.-', 'LineWidth', 2);
            title('\gammaTilde vs pVal');
            xlabel('pVal'); ylabel('\gammaTilde');
            set(gca, 'XScale', 'log');  % Set x-axis to log scale
            grid on;
            
            subplot(3, 2, 5);
            plot(pValRange, deltaVal, '.-', 'LineWidth', 2);
            title('\delta vs pVal');
            xlabel('pVal'); ylabel('\delta');
            set(gca, 'XScale', 'log');  % Set x-axis to log scale
            grid on;

             % Save figure as .fig and .png
            figFileName = sprintf('Results/gridSearchLocalControl2.fig');
            pngFileName = sprintf('Results/gridSearchLocalControl2.png');
            
            savefig(fig, figFileName);  % Save as .fig file
            print(fig, pngFileName, '-dpng', '-r300');  % Save as high-res PNG (300 dpi)
            close(fig);  % Close the figure after saving
            
            save('Results/gridSearchLocalControlResults2.mat', 'pValRange', 'nuVal', 'rhoVal', 'LNormVal', 'gammaTildeVal', 'deltaVal');
            
            disp('Results saved to "results/gridSearchLocalControlResults2.mat".');
        end


        function plotPerformance(obj)
            % Number of inventories in the chain
            numOfInventories = obj.numOfInventories;
            lineWidthVal = 1;
            
            % Create figure for this chain's performance
            figure;
            sgtitle(['Performance Metrics for Chain ', num2str(obj.chainId)]);
            
            % Create a tiled layout with 3 rows and 1 column with increased height
            t = tiledlayout(4, 1, 'TileSpacing', 'loose', 'Padding', 'loose');  % Adjust TileSpacing and Padding for more space
            
            % Plot 0: Consensus Error History
            nexttile;
            hold on; grid on;
            for i = 1:numOfInventories
                plot(obj.inventories{i}.consensusErrorHistory, '.-', 'LineWidth', lineWidthVal);
            end
            % title('Consensus Error');
            % xlabel('Time Step');
            ylabel('Consensus Error');
            % legend(arrayfun(@(i) ['Inv ', num2str(i)], 1:numOfInventories, 'UniformOutput', false));
            axis([-inf,inf,-500,500])
            hold off;

            % Plot 1: Tracking Error History
            nexttile;
            hold on; grid on;
            for i = 1:numOfInventories
                plot(obj.inventories{i}.trackingErrorHistory, '.-', 'LineWidth', lineWidthVal);
            end
            % title('Tracking Error');
            % xlabel('Time Step');
            ylabel('Tracking Error');
            % legend(arrayfun(@(i) ['Inv ', num2str(i)], 1:numOfInventories, 'UniformOutput', false));
            axis([-inf,inf,-500,500])
            hold off;
            
            % Plot 2: Inventory Level History
            nexttile;
            hold on; grid on;
            for i = 1:numOfInventories
                plot(obj.inventories{i}.inventoryHistory, '.-', 'LineWidth', lineWidthVal);
            end
            % title('Inventory Level');
            % xlabel('Time Step');
            ylabel('Inventory');
            % legend(arrayfun(@(i) ['Inv ', num2str(i)], 1:numOfInventories, 'UniformOutput', false));
            axis([-inf,inf,0,1000])
            hold off;
            
            % Plot 3: Order Magnitudes (In and Out)
            nexttile;
            hold on; grid on;
            for i = 1:numOfInventories
                plot(obj.inventories{i}.orderInHistory, '.-', 'LineWidth', lineWidthVal);  % Incoming orders (dashed line)
            end
            % title('Order Magnitudes');
            xlabel('Time Step');
            ylabel('Order Magnitude');
            legend(arrayfun(@(i) ['Inv ', num2str(i)], 1:numOfInventories, 'UniformOutput', false),'Location','southoutside');
            axis([-inf,inf,0,1000])
            hold off;
            
            % Adjust the layout size
            t.TileSpacing = 'compact';  % Adjust spacing between tiles if needed
            t.Padding = 'compact';      % Adjust padding around the layout if needed

        end


        function draw(obj)
            obj.supplier.draw();
            for k = 1:1:obj.numOfInventories
                obj.inventories{k}.draw();
                obj.phyLinks{k}.draw();
            end
            obj.phyLinks{k+1}.draw();
            obj.demander.draw();            
        end

    end
end