classdef Network < handle
    % This class represents supply chain networks (SCNs)

    properties  %% Network class properties
        netId               % ID Number
        chains              % Cell array to store chain objects
        numOfChains         % Number of chains in the network
        globalController   % Controller gain matrix (block matrix NxN, each block is nxn)

        cumMeanAbsError     % Cumulative moving average error across the entire network
        numUpdates         % Number of updates at the network level
    end

    methods
        function obj = Network(netID, numOfChains, numOfInventories)
            disp('Started creating the supply chain network.');
            obj.netId = netID;

            % Create cell array for chains
            chains = cell(1, numOfChains);  % Use a cell array for chain objects
            
            % Creating each chain and storing in the cell array
            for i = 1:numOfChains
                chain = Chain(i, numOfInventories);  % Create each Chain object
                chains{i} = chain;  % Store the Chain in the cell array
            end
            
            % Assign the chains to the network
            obj.chains = chains;
            obj.numOfChains = numOfChains;

            % Controller gain:
            obj.globalController = obj.generateDistanceBasedK(200); % obj.generateSparseK();  % This is just a placeholder, will be designed later;
            obj.setNeighborsAndControllers();

            obj.numUpdates = 0;
            obj.cumMeanAbsError = 0;  % Start with 0 average

            disp('Finished creating the supply chain network.');
        end

        function update(obj)
            globalErrorSum = 0;
            for i = 1:1:obj.numOfChains
                obj.chains{i}.update();
                 globalErrorSum = globalErrorSum + obj.chains{i}.cumMeanAbsError;
            end
            
            obj.numUpdates = obj.numUpdates + 1;    % Update the number of updates
    
            % Compute global cumulative average error
            obj.cumMeanAbsError = ((obj.numUpdates - 1) * obj.cumMeanAbsError + (globalErrorSum / obj.numOfChains)) / obj.numUpdates;
        end

        % Function to set in and out neighbors based on the K matrix
        function setNeighborsAndControllers(obj)
            N = obj.numOfChains;  % Number of chains
            n = obj.chains{1}.numOfInventories;  % Number of inventories per chain
            K = obj.globalController;

            % Loop through each chain
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
                                    obj.chains{i}.inventories{k}.inNeighbors = [obj.chains{i}.inventories{k}.inNeighbors, obj.chains{j}.inventories{l}];
                                    % Assign the corresponding scaling factor (K_ij element)
                                    obj.chains{i}.inventories{k}.globalControllers = [obj.chains{i}.inventories{k}.globalControllers, K_ij_kl];
                                end
                            end
                        end
                    end
                end
            end
        end

        % Generate a sparse K matrix for closest neighboring inventories
        function K = generateSparseK(obj)
            N = obj.numOfChains;  % Number of chains
            n = obj.chains{1}.numOfInventories;  % Number of inventories per chain
            
            % Initialize the K matrix with zeros
            K = zeros(N * n, N * n);
            
            % Iterate over each chain
            for i = 1:N
                for j = 1:N
                    % Within the same chain (i == j)
                    if i == j
                        % Link each inventory with its immediate neighbors
                        for k = 1:n
                            K((i-1)*n + k, (i-1)*n + k) = 2*rand();
                            if k > 1  % Connect to the previous inventory
                                K((i-1)*n + k, (i-1)*n + k - 1) = rand();  % Random scaling factor for backward link
                            end
                            if k < n  % Connect to the next inventory
                                K((i-1)*n + k, (i-1)*n + k + 1) = rand();  % Random scaling factor for forward link
                            end
                        end
                    else
                        % Link inventories between adjacent chains (i != j)
                        if abs(i - j) == 1  % Only link adjacent chains
                            for k = 1:n
                                % Create cross-chain links (e.g., inventory k in chain i to inventory k in chain j)
                                K((i-1)*n + k, (j-1)*n + k) = rand();  % Random scaling factor for cross-chain link
                            end
                        end
                    end
                end
            end
        end


        function K = generateDistanceBasedK(obj, distanceThreshold)
            N = obj.numOfChains;  % Number of chains
            n = obj.chains{1}.numOfInventories;  % Number of inventories per chain
            
            % Initialize the K matrix with zeros
            K = zeros(N * n, N * n);
            
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
                                K((i-1)*n + inven_i, (j-1)*n + inven_j) = randn();  % Random scaling factor
                            else
                                K((i-1)*n + inven_i, (j-1)*n + inven_j) = 0;  % No connection if too far
                            end
                        end
                    end
                end
            end
        end

        function plotConnections(obj)
            
            % Number of chains and inventories
            N = obj.numOfChains;
            n = obj.chains{1}.numOfInventories;
            K = obj.globalController;

            % Loop through each chain and plot inventories as dots
            for chainIdx = 1:N
                for invenIdx = 1:n
                    % Get the location of the inventory
                    inventory = obj.chains{chainIdx}.inventories{invenIdx};
                    plot(inventory.location(1), inventory.location(2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');  % Plot inventory as black dot
                    
                    % Label the inventory for reference
                    text(inventory.location(1), inventory.location(2)+5, ...
                         sprintf('Chain %d, Inv. %d', chainIdx, invenIdx), 'HorizontalAlignment', 'center');
                end
            end
            
            % Now, loop through the K matrix and plot arrows for non-zero connections
            for i = 1:N
                for j = 1:N
                    for inven_i = 1:n
                        for inven_j = 1:n
                            % Get the corresponding element from the K matrix
                            if K((i-1)*n+inven_i, (j-1)*n+inven_j) ~= 0
                                % Get the locations of the source and target inventories
                                srcLocation = obj.chains{i}.inventories{inven_i}.location;
                                destLocation = obj.chains{j}.inventories{inven_j}.location;
                                
                                % Draw an arrow to indicate the connection
                                quiver(srcLocation(1), srcLocation(2), ...
                                       destLocation(1)-srcLocation(1), destLocation(2)-srcLocation(2), ...
                                       'MaxHeadSize', 0.2, 'Color', 'r', 'LineWidth', 1.5);
                            end
                        end
                    end
                end
            end
            
        end


        function plotNetworkPerformance(obj)
            for i = 1:obj.numOfChains
                obj.chains{i}.plotPerformance();
            end
        end

        function draw(obj, currentTime)
            % Draw each chain in the network
            for i = 1:1:obj.numOfChains
                obj.chains{i}.draw();
            end
            
             % Display the global cumulative average error on the plot
            timeText = ['Time: ', num2str(currentTime)];
            text(0, 25, timeText, 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
            
            errorText = ['Mean Absolute Tracking Error: ', num2str(obj.cumMeanAbsError,3)];
            text(0, 10, errorText, 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'r');  % Display the global average error below the time
        end

    end
end