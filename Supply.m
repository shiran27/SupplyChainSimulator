classdef Supply < handle
    %UNTITLED12 Summary of this class goes here
    %   Detailed explanation goes here

    
    properties %% Supply class properties
        supId               % ID Number
        supRateMax          % Maximum supply rate per time unit
        location            % Location for visualization
        size                % Size for visualization
        productOut          % Outgoing product to the first inventory
        phyLinkOut          % Outgoing physical link to the first inventory
    end

    methods
        function obj = Supply(supId, location, sizeS)
            % disp('Started creating a supply...');
            obj.supId = supId;
            obj.location = location;
            obj.size = sizeS;

            % Loading supply characteristics
            obj.supRateMax = 100000;  % Maximum supply rate per time step
            obj.productOut = randi([1,50], 1);     % Initialize outgoing product

            % disp('Finished creating a supply...');
        end

        function supplyProducts(obj)
            % Supply products to the first inventory based on its orderOut request
            requestedAmount = obj.phyLinkOut.downInven.orderIn;
            obj.productOut = min(requestedAmount, obj.supRateMax);  % Fulfill as much as possible
        end
        
        function updateState(obj)
            % Supplier has no internal state to update
        end

        function draw(obj)
            % Draw the supplier as a rectangle (box)
            x = obj.location(1) - obj.size/2;
            y = obj.location(2) - obj.size/2;
            rectangle('Position', [x, y, obj.size, obj.size], 'FaceColor', [1, 0.7, 0.7]);  % Light red color
            
            % Grid arrangement for product dots
            numProducts = floor(obj.productOut);  % Using productOut for supplier
            gridCols = 5;
            maxProducts = 100;  % Assume a maximum level for visualization purposes
            gridRows = ceil(maxProducts / gridCols);
            
            % Place products in a uniform grid inside the box
            % for i = 1:numProducts
            %     col = mod(i-1, gridCols) + 1;
            %     row = ceil(i / gridCols);
            %     plot(x + (col - 0.5) * (obj.size / gridCols), ...
            %          y + (row - 0.5) * (obj.size / gridRows), '.', 'MarkerSize', 4, 'Color', 'k');  % Dots for products
            % end
            
            % Display the product count as a number on top of the supplier box
            text(obj.location(1), obj.location(2) + obj.size/2 + 10, num2str(numProducts), 'HorizontalAlignment', 'center','FontSize',8);
        end




    end
end