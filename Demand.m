classdef Demand < handle
    %UNTITLED9 Summary of this class goes here
    %   Detailed explanation goes here

    properties              % Demand class properties
        demId               % ID Number

        dailyMeans 
        demRateMean         % Mean rate of product consumption
        demRateStd          % Standard deviation in the rate of demand
        
        demand
        demandHistory

        location            % Location for visualization
        size                % Size for visualization
        productIn           % Incoming products from the last inventory
        orderIn             % Amount requested from upstream
        phyLinkIn           % Incoming physical link from the last inventory
    end

    methods
        function obj = Demand(demId, numOfChains, location, sizeD)
            % disp('Started creating a demand...');
            obj.demId = demId;
            obj.location = location;
            obj.size = sizeD;

            % Loading demand characteristics
            obj.dailyMeans = 100 + 2*randi([1,10*numOfChains],7,1) + 20*demId;
            obj.demRateMean = mean(obj.dailyMeans);        % Mean demand rate
            obj.demRateStd = 0.2*obj.demRateMean;           % Variance in demand
            
            obj.productIn = randi([100,900],1);     % Initialize incoming products
            obj.demand = obj.demRateMean;
            obj.demandHistory = [];
            % disp('Finished creating a demand...');
        end

        function generateDemand(obj, t)
            % Generate a demand (integer) randomly at the beginning of each time step
            % with a specified mean and variance
            demand = obj.demRateMean + obj.demRateStd * randn();

            % Smoothing factor (between 0 and 1, where 1 means no smoothing)
            alpha = 0.1; % 0.1
            % Apply exponential moving average to smooth demand (obj.previousDemand is the previous value)
            demandSmooth = alpha * demand + (1 - alpha) * obj.demand;
            demandFiltered = max(0, round(demandSmooth));

            obj.demand = demandFiltered;
            obj.demandHistory = [obj.demandHistory, obj.demand];
            % Round the generated demand to the nearest integer and ensure it's non-negative
            obj.orderIn = demandFiltered;
        end

        
        function updateState(obj)
            % Update the state based on incoming products from the last inventory
            obj.productIn = obj.phyLinkIn.delayBuffer(end);  % Receive products after delay

            % Track how much of the demand was fulfilled (optional)
            fulfilledDemand = min(obj.orderIn, obj.productIn);  % What was actually fulfilled
            remainingDemand = obj.orderIn - fulfilledDemand;  % If any demand is unmet
            
            % Update internal states (if needed) based on fulfilled demand
            % This could be useful for demanders to track unmet demands, etc.

        end

        function draw(obj)
            % Draw the demand as a rectangle (box)
            x = obj.location(1) - obj.size/2;
            y = obj.location(2) - obj.size/2;
            rectangle('Position', [x, y, obj.size, obj.size], 'FaceColor', [0.7, 0.7, 1]);  % Light blue color
            
            % Grid arrangement for product dots
            numProducts = floor(obj.productIn);  % Using productIn for demand
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
            
            % Display the product count as a number on top of the demand box
            text(obj.location(1), obj.location(2) + obj.size/2 + 10, num2str(numProducts), 'HorizontalAlignment', 'center','FontSize',8);
            text(obj.location(1), obj.location(2) - obj.size/2 - 10, num2str(obj.orderIn), 'HorizontalAlignment', 'center', 'Color', 'r','FontSize',8);
        end





    end
end