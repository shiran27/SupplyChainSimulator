classdef Inventory < handle
    % This class represents an inventory

    properties    %% Inventory class properties
        chainId         % Chain ID
        invenId         % ID Number
        refLevel        % Preferred inventory level
        maxLevel        % Maximum inventory level
        minLevel        % Minimum inventory level

        perishRate 
        wasteRateMean      % Mean waste rate
        wasteRateStd       % Standard deviation of the waste rate
        waste

        uOverBar        % Mean order
        uTilde
        uTildeTilde

        productIn       % Incoming product from upstream (supplier or previous inventory)
        productOut      % Outgoing product to downstream (next inventory or demander)

        orderIn         % Amount requested from upstream (previous inventory or supplier)
        orderOut        % Amount requested by downstream (next inventory or demander)

        state           % Current inventory level (number of products in stock)

        trackingError       % Inventory tracking error
        consensusError       % Inventory tracking error
        cumMeanAbsTraError      % Cumulative moving average of the inventory tracking error
        cumMeanAbsConError      % Cumulative moving average of the inventory tracking error
        numUpdates       % Number of updates performed so far (used for cumulative moving average)

        inventoryHistory    % Stores the history of inventory levels
        orderInHistory      % Stores the history of order inputs
        orderOutHistory     % Stores the history of order outputs
        trackingErrorHistory % Stores the history of tracking errors
        consensusErrorHistory   % Stores the history of consensus errors\
        uHistory

        inNeighbors         % List of inventories from which this inventory receives error information
        KGlobal   % Scaling factors (from K) for each in-neighbor

        phyLinkIn       % Reference to incoming PhyLink (from upstream)
        phyLinkOut      % Reference to outgoing PhyLink (to downstream)

        location        % Location for visualization
        size            % Size for visualization
    end

    methods
        
        function obj = Inventory(chainId, invenId, locationY, sizeI)
            % disp('Started creating an inventory...')
            obj.chainId = chainId;
            obj.invenId = invenId;

            % Load inventory characteristics
            obj.refLevel = 500;
            obj.maxLevel = 1000;
            obj.minLevel = 10;

            obj.perishRate = 0.1;
            obj.wasteRateMean = 10 + 2*randi([1,5]) + 2*chainId;
            obj.wasteRateStd = 0.2*obj.wasteRateMean;
            obj.waste = obj.wasteRateMean;

            % Location
            obj.location = [100*invenId, locationY];
            obj.size = sizeI;

            % Initial State
            obj.state = randi([obj.refLevel-100,obj.refLevel+100]);

            obj.cumMeanAbsTraError = 0;  % Start with 0 average
            obj.cumMeanAbsConError = 0;  % Start with 0 average
            obj.numUpdates = 0;

            obj.inventoryHistory = [];    % Stores the history of inventory levels
            obj.orderInHistory = [];      % Stores the history of order inputs
            obj.orderOutHistory = [];     % Stores the history of order outputs
            obj.trackingErrorHistory = []; % Stores the history of tracking errors
            obj.consensusErrorHistory = []; % Stores the history of consensus errors
            obj.uHistory = [];
            
            % disp('Finished creating an inventory...')
        end


        % Compute the inventory tracking and consensus errors and update
        % cumulative moving averages
        function computeErrors(obj)

            % Tracking error: difference between actual state and reference level
            obj.trackingError = obj.state - obj.refLevel;
            % Consensus error is already computed
            obj.consensusError;

            % Update the number of updates
            obj.numUpdates = obj.numUpdates + 1;
            
            % Update the cumulative moving averages
            obj.cumMeanAbsTraError = ((obj.numUpdates-1)*obj.cumMeanAbsTraError + abs(obj.trackingError))/obj.numUpdates;
            obj.cumMeanAbsConError = ((obj.numUpdates-1)*obj.cumMeanAbsConError + abs(obj.consensusError))/obj.numUpdates;

            % Log tracking and consensus errors
            obj.trackingErrorHistory = [obj.trackingErrorHistory, obj.trackingError];
            obj.consensusErrorHistory = [obj.consensusErrorHistory, obj.consensusError];
        end


        % Compute how much to request from upstream (orderIn)
        function computeOrder(obj)
            % Compute how much to request from upstream (orderIn)
            % obj.orderIn = randi([1,5]);

            % Initialize the requested order
            consensusControlInput = 0;
            % Sum contributions from in-neighbors, scaled by the corresponding K elements
            % inventoryID = [obj.chainId,obj.invenId]
            for neighborIdx = 1:length(obj.inNeighbors)
                neighbor = obj.inNeighbors(neighborIdx);
                % neighborID = [neighbor.chainId, neighbor.invenId]
                controllerGain = obj.KGlobal(neighborIdx);  % Get the corresponding scaling factor from K
                consensusControlInput = consensusControlInput + controllerGain * neighbor.trackingError;
            end
            obj.uTildeTilde = consensusControlInput;

            % Ensure the requested order is non-negative and reasonable
            totalOrder = obj.uOverBar + 1*obj.uTilde + 1*obj.uTildeTilde;
            totalOrderFiltered = max(min(round(totalOrder), 1*obj.maxLevel),0);
            obj.orderIn = totalOrderFiltered;

            % Log order history
            obj.orderInHistory = [obj.orderInHistory, obj.orderIn];
            obj.uHistory = [obj.uHistory, [obj.uOverBar; obj.uTilde; obj.uTildeTilde; obj.orderIn]];
        end


        function updateState(obj)
            % Update the state based on incoming products from the last inventory
            obj.productIn = obj.phyLinkIn.delayBuffer(end);  % Receive products after delay
            
            % Product Waste
            wasteValue = obj.wasteRateMean + obj.wasteRateStd * randn();
            % Smoothing factor (between 0 and 1, where 1 means no smoothing)
            alpha = 0.5; 
            % Apply exponential moving average to smooth waste
            wasteSmooth = alpha * wasteValue + (1 - alpha) * obj.waste;
            wasteFiltered = max(0, round(wasteSmooth));
            obj.waste = wasteFiltered;

            % Update inventory level
            newState = (1 - obj.perishRate) * obj.state + obj.productIn - obj.waste;
            obj.state = max(0, round(newState));
            
            % Fulfill as much of the downstream order as possible
            obj.productOut = min([obj.phyLinkOut.downInven.orderIn, obj.state - obj.minLevel]);
            
            % Update the state after sending out products, ensuring it doesn't drop below the min level
            obj.state = min([obj.state - obj.productOut, obj.maxLevel]);  % Use max to ensure state >= minLevel

            % Log inventory level and order magnitudes
            obj.inventoryHistory = [obj.inventoryHistory, obj.state];
            obj.orderOutHistory = [obj.orderOutHistory, obj.productOut];
        end


        function draw(obj)
            % Draw the inventory as a rectangle (box)
            x = obj.location(1) - obj.size/2;
            y = obj.location(2) - obj.size/2;
            rectangle('Position', [x, y, obj.size, obj.size], 'FaceColor', [0.7 1 0.7]);
        
            % Code to plot tracking error (obj.trackingError) and consensus error (obj.consensusError) drawn as verticle
            % lines drawn parallel to each other on top of a horizontal line drawn in the middle of the box (like an x axis).
            % Draw a horizontal line at the middle of the box
            middleY = obj.location(2);
            line([obj.location(1) - obj.size/3, obj.location(1) + obj.size/3], [middleY, middleY], 'Color', 'k', 'LineWidth', 1);  % Horizontal line
        
            del = 2;
            % Plot tracking error as a vertical line (blue) from the middle line
            trackErrHeight = obj.trackingError/5;  % Scale error to fit inside the box
            if trackErrHeight > obj.size/2 - del
                trackErrHeight = obj.size/2 - del;
            elseif trackErrHeight < -obj.size/2 + del
                trackErrHeight = -obj.size/2 + del;
            end
            line([obj.location(1) - obj.size/6, obj.location(1) - obj.size/6], ...
                 [middleY, middleY + trackErrHeight], 'Color', 'b', 'LineWidth', 1.5);  % Blue for tracking error
            
            % Plot consensus error as a vertical line (green) from the middle line
            conErrHeight = obj.consensusError/5;  % Scale error to fit inside the box
            if conErrHeight > obj.size/2 - del
                conErrHeight = obj.size/2 - del;
            elseif conErrHeight < -obj.size/2 + del
                conErrHeight = -obj.size/2 + del;
            end
            line([obj.location(1) + obj.size/6, obj.location(1) + obj.size/6], ...
                 [middleY, middleY + conErrHeight], 'Color', 'r', 'LineWidth', 1.5);  % Green for consensus error
        

            % Display the product count as a number on top of the inventory box
            text(obj.location(1), obj.location(2) - obj.size/2 - 10, ['x:',num2str(obj.state)], 'HorizontalAlignment', 'center', 'Color', 'b','FontSize',8);
            text(obj.location(1), obj.location(2) + obj.size/2 + 10, ['u:',num2str(obj.orderIn)], 'HorizontalAlignment', 'center', 'Color', 'r','FontSize',8);
        end


    end
end