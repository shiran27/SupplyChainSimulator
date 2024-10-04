classdef Inventory < handle
    % This class represents an inventory

    properties    %% Inventory class properties
        chainId         % Chain ID
        invenId         % ID Number
        refLevel        % Preferred inventory level
        maxLevel        % Maximum inventory level
        minLevel        % Minimum inventory level
        wasteRate       % Waste rate

        productIn       % Incoming product from upstream (supplier or previous inventory)
        productOut      % Outgoing product to downstream (next inventory or demander)

        orderIn         % Amount requested from upstream (previous inventory or supplier)
        orderOut        % Amount requested by downstream (next inventory or demander)

        state           % Current inventory level (number of products in stock)

        trackingError       % Inventory tracking error
        cumMeanAbsError      % Cumulative moving average of the inventory tracking error
        numUpdates       % Number of updates performed so far (used for cumulative moving average)

        inventoryHistory    % Stores the history of inventory levels
        orderInHistory      % Stores the history of order inputs
        orderOutHistory     % Stores the history of order outputs
        trackingErrorHistory % Stores the history of tracking errors

        inNeighbors         % List of inventories from which this inventory receives error information
        globalControllers   % Scaling factors (from K) for each in-neighbor

        phyLinkIn       % Reference to incoming PhyLink (from upstream)
        phyLinkOut      % Reference to outgoing PhyLink (to downstream)

        location        % Location for visualization
        size            % Size for visualization
    end

    methods
        function obj = Inventory(chainId, invenId, locationY, sizeI)
            disp('Started creating an inventory...')
            obj.chainId = chainId;
            obj.invenId = invenId;

            % Load inventory characteristics
            obj.refLevel = 50;
            obj.maxLevel = 100;
            obj.minLevel = 10;
            obj.wasteRate = 0;

            % Location
            obj.location = [100*invenId, locationY];
            obj.size = sizeI;

            % Initial State
            obj.state = 50;
            obj.computeTrackingError();

            obj.numUpdates = 0;
            obj.cumMeanAbsError = 0;  % Start with 0 average

            obj.inventoryHistory = [];    % Stores the history of inventory levels
            obj.orderInHistory = [];      % Stores the history of order inputs
            obj.orderOutHistory = [];     % Stores the history of order outputs
            obj.trackingErrorHistory = []; % Stores the history of tracking errors
            
            disp('Finished creating an inventory...')
        end

        % Compute the inventory tracking error and update cumulative moving average
        function computeTrackingError(obj)
            % Tracking error: difference between actual state and reference level
            obj.trackingError = obj.refLevel - obj.state;
    
            % Update the number of updates
            obj.numUpdates = obj.numUpdates + 1;
    
            % Update the cumulative moving average of the tracking error
            obj.cumMeanAbsError = ((obj.numUpdates - 1) * obj.cumMeanAbsError + abs(obj.trackingError)) / obj.numUpdates;
        end

        % Compute how much to request from upstream (orderIn)
        function computeOrder(obj)
            % Compute how much to request from upstream (orderIn)
            % obj.orderIn = randi([1,5]);

            % Initialize the requested order
            requestedOrder = 0;
            % Sum contributions from in-neighbors, scaled by the corresponding K elements
            for neighborIdx = 1:length(obj.inNeighbors)
                neighbor = obj.inNeighbors(neighborIdx);
                scalingFactor = obj.globalControllers(neighborIdx);  % Get the corresponding scaling factor from K
                requestedOrder = requestedOrder + scalingFactor * neighbor.trackingError;
            end
            requestedOrder = round(requestedOrder);

            % Ensure the requested order is non-negative and reasonable
            obj.orderIn = max(min(-0.1*requestedOrder, obj.maxLevel/2), 0);

            % Log tracking error
            obj.trackingErrorHistory = [obj.trackingErrorHistory, obj.trackingError];
            obj.orderInHistory = [obj.orderInHistory, obj.orderIn];
        end
        
        function updateState(obj)
            % Update the state based on incoming products from the last inventory
            obj.productIn = obj.phyLinkIn.delayBuffer(end);  % Receive products after delay
            
            % Update inventory level
            obj.state = round((1 - obj.wasteRate) * obj.state) + obj.productIn;
            
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
            rectangle('Position', [x, y, obj.size, obj.size], 'FaceColor', [0.8 0.8 0.8]);
        
            % Grid arrangement for product dots
            numProducts = floor(obj.state);
            gridCols = 5;
            maxProducts = obj.maxLevel;  % Maximum number of products (for visualization)
            gridRows = ceil(maxProducts / gridCols);
            
            % Place products in a uniform grid inside the box
            for i = 1:numProducts
                col = mod(i-1, gridCols) + 1;
                row = ceil(i / gridCols);
                plot(x + (col - 0.5) * (obj.size / gridCols), ...
                     y + (row - 0.5) * (obj.size / gridRows), '.', 'MarkerSize', 4, 'Color', 'k');  % Dots for products
            end
            
            % Display the product count as a number on top of the inventory box
            text(obj.location(1), obj.location(2) + obj.size/2 + 10, num2str(numProducts), 'HorizontalAlignment', 'center');
            text(obj.location(1), obj.location(2) - obj.size/2 - 10, num2str(obj.orderIn), 'HorizontalAlignment', 'center', 'Color', 'r');
            text(obj.location(1), obj.location(2) - obj.size/2 - 25, num2str(obj.trackingError), 'HorizontalAlignment', 'center', 'Color', 'g');
        end




    end
end