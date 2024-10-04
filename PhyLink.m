classdef PhyLink < handle
    %UNTITLED7 Summary of this class goes here
    %   Detailed explanation goes here
  
    properties   %% PhyLink class properties
        phyLinkId       % ID Number
        upInven         % Upstream Inventory or Supply
        downInven       % Downstream Inventory or Demand
        tranDelay       % Delay in transportation (in time steps)
        delayBuffer     % Buffer to store products in transit
        location        % Location for visualization
    end

    methods
        function obj = PhyLink(id, location, tranDelay, upInven, downInven)
            disp('Started creating a PhyLink...');
            obj.phyLinkId = id;

            % Passing references to up and down inventories (or supplier/demander)
            obj.upInven = upInven;
            obj.downInven = downInven;
            obj.tranDelay = tranDelay;

            % Initialize the delay buffer (FIFO queue)
            obj.delayBuffer = randi([1,10], 1, tranDelay);  % Buffer size equal to the delay

            % Set location for visualization
            obj.location = location;

            disp('Finished creating a PhyLink...');
        end

        function transportGoods(obj)
            % Transport goods from upstream to downstream with delay

            % Deliver the oldest product in the buffer to the downstream node
            % obj.downInven.productIn = obj.delayBuffer(end);


            % Shift the buffer and add the new product from the upstream node
            if obj.tranDelay > 1
                obj.delayBuffer = [obj.upInven.productOut, obj.delayBuffer(1:end-1)];
            else
                obj.delayBuffer = obj.upInven.productOut;
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
                plot(xPos, yPos, 'k.', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
                
                % Vertical column of dots to represent number of products in this buffer spot
                numProducts = obj.delayBuffer(i);
                for j = 1:numProducts
                    plot(xPos, yPos + j*5, 'k.', 'MarkerSize', 4, 'Color', 'k');  % Use dots instead of circles
                end
                
                % Display the buffered product count as text below the dot
                text(xPos, yPos - 10, num2str(numProducts), 'HorizontalAlignment', 'center');
            end
        end



    end
end