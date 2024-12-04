classdef PhyLink < handle
    %UNTITLED7 Summary of this class goes here
    %   Detailed explanation goes here
  
    properties   %% PhyLink class properties
        phyLinkId       % ID Number
        upInven         % Upstream Inventory or Supply
        downInven       % Downstream Inventory or Demand
        tranDelay       % Delay in transportation (in time steps)
        delayBuffer     % Buffer to store products in transit
        wasteRateMean      % Mean waste rate
        wasteRateStd       % Standard deviation of the waste rate
        waste
        location        % Location for visualization
    end

    methods
        function obj = PhyLink(id, chainId, location, tranDelay, upInven, downInven)
            % disp('Started creating a PhyLink...');
            obj.phyLinkId = id;

            % Passing references to up and down inventories (or supplier/demander)
            obj.upInven = upInven;
            obj.downInven = downInven;
            obj.tranDelay = tranDelay;
            obj.wasteRateMean = 10 + 2*randi([1,5]) + 2*chainId;      % Mean waste rate
            obj.wasteRateStd = 0.2*obj.wasteRateMean;       % Standard deviation of the waste rate
            obj.waste = obj.wasteRateMean;

            % Initialize the delay buffer (FIFO queue)
            obj.delayBuffer = randi([100,900], 1, tranDelay);  % Buffer size equal to the delay [1,50]

            % Set location for visualization
            obj.location = location;

            % disp('Finished creating a PhyLink...');
        end

        function transportGoods(obj,t)
            % Transport goods from upstream to downstream with delay

            % Deliver the oldest product in the buffer to the downstream node
            % obj.downInven.productIn = obj.delayBuffer(end);

            % Product Waste
            wasteValue = obj.wasteRateMean + obj.wasteRateStd * randn();

            % dayNum = floor(t/24);
            % if dayNum >= 25
            %     wasteValue = obj.wasteRateMean + 0.25 * obj.wasteRateStd * randn();
            % end

            % Smoothing factor (between 0 and 1, where 1 means no smoothing)
            alpha = 0.5; 
            % Apply exponential moving average to smooth waste
            wasteSmooth = alpha * wasteValue + (1 - alpha) * obj.waste;
            wasteValueFiltered = min(round(wasteSmooth), obj.upInven.productOut); 
            obj.waste = wasteValueFiltered;


            newProductIn = max(0, obj.upInven.productOut - obj.waste);

            % Shift the buffer and add the new product from the upstream node
            if obj.tranDelay > 1
                obj.delayBuffer = [newProductIn, obj.delayBuffer(1:end-1)];
            else
                obj.delayBuffer = newProductIn;
            end

        end

        function draw(obj)
            % Draw the physical link as a line
            plot(obj.location(:,1), obj.location(:,2), 'k', 'LineWidth', 2);
            
            % Plot bold dots to represent buffer spots (not on edges)
            numSteps = length(obj.delayBuffer);  % Number of delay steps (columns)
            for i = 1:numSteps
                % Interpolate position along the line for each buffer step
                xPos = interp1([1, numSteps+1], obj.location(:,1), i+0.5);  % +0.5 to avoid edges
                yPos = interp1([1, numSteps+1], obj.location(:,2), i+0.5);
                
                % Bold dot to represent the buffer spot
                plot(xPos, yPos, 'k.', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
                
                % Vertical column of dots to represent number of products in this buffer spot
                numProducts = obj.delayBuffer(i);
                heightVal = numProducts/20;
                if heightVal > 30
                    heightVal = 30;
                end
                plot([xPos, xPos], [yPos, yPos + heightVal], '-k', 'MarkerSize', 4, 'Color', 'k');  % Use dots instead of circle
                
                % Display the buffered product count as text below the dot
                text(xPos, yPos - 10, num2str(numProducts), 'HorizontalAlignment', 'left', 'Rotation', -90,'FontSize',6);
            end
        end



    end
end