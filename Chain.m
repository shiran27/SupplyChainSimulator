classdef Chain < handle
    % This class represents supply chains

    properties %% Chain class properties
        chainId            % ID number
        supplier           % Supplier object
        inventories        % Array of Inventory objects
        numOfInventories   % Number of inventories
        phyLinks           % Array of PhyLink objects (transport between inventories)
        demander           % Demand object (at the end of the chain)
        location           % Chain location (optional for plotting)

        cumMeanAbsError    % Cumulative moving average of the tracking error for the chain
        numUpdates         % Number of updates performed at the chain level
    end

    methods
        function obj = Chain(chainId, numOfInventories)
            disp(['Started creating chain ', num2str(chainId)])
            obj.chainId = chainId;

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
            obj.demander = Demand(chainId, obj.location(2,:), sizeD);

            % Create phyLinks and assign them to the supplier, inventories, and demander
            phyLinks = cell(1, numOfInventories + 1);  % One link for each transition
            for k = 1:(numOfInventories+1)
                phyLinkId = numOfInventories * (chainId - 1) + k;
                if k == 1
                    % Link from supplier to the first inventory
                    location = [obj.supplier.location; obj.inventories{k}.location];
                    location(1,1) = location(1,1) + sizeS/2; 
                    location(2,1) = location(2,1) - sizeI/2; 
                    tranDelay = 1;
                    phyLink = PhyLink(phyLinkId, location, tranDelay, obj.supplier, obj.inventories{k}); 
                    % Assign phyLink to the first inventory's phyLinkIn and supplier's phyLinkOut
                    obj.inventories{k}.phyLinkIn = phyLink;
                    obj.supplier.phyLinkOut = phyLink;
                elseif k <= numOfInventories
                    % Link between two consecutive inventories
                    location = [obj.inventories{k-1}.location; obj.inventories{k}.location];
                    location(1,1) = location(1,1) + sizeI/2; 
                    location(2,1) = location(2,1) - sizeI/2; 
                    tranDelay = 5;
                    phyLink = PhyLink(phyLinkId, location, tranDelay, obj.inventories{k-1}, obj.inventories{k});
                    % Assign phyLink to the inventories' phyLinkIn and phyLinkOut
                    obj.inventories{k-1}.phyLinkOut = phyLink;  % Outgoing link for previous inventory
                    obj.inventories{k}.phyLinkIn = phyLink;     % Incoming link for current inventory
                else
                    % Link from the last inventory to the demander
                    location = [obj.inventories{k-1}.location; obj.demander.location];
                    location(1,1) = location(1,1) + sizeI/2; 
                    location(2,1) = location(2,1) - sizeD/2; 
                    tranDelay = 1;
                    phyLink = PhyLink(phyLinkId, location, tranDelay, obj.inventories{k-1}, obj.demander);
                    % Assign phyLink to the last inventory's phyLinkOut and demander's phyLinkIn
                    obj.inventories{k-1}.phyLinkOut = phyLink;
                    obj.demander.phyLinkIn = phyLink;
                end
                phyLinks{k} = phyLink;  % Store the PhyLink reference
            end 
            obj.phyLinks = phyLinks;

            obj.numUpdates = 0;
            obj.cumMeanAbsError = 0;  % Start with 0 average

            disp('Finished creating a chain...')
        end



        function update(obj)
            % Step 1: Demand generation
            obj.demander.generateDemand(); % Compute order in 

            % Step 2: Compute the tracking errors
            chainErrorSum = 0;
            for i = 1:1:obj.numOfInventories
                obj.inventories{i}.computeTrackingError(); 
                chainErrorSum = chainErrorSum + obj.inventories{i}.cumMeanAbsError;  % Sum cumulative average errors
            end
            obj.numUpdates = obj.numUpdates + 1; % Update the number of updates
            % Compute chain's cumulative average error
            obj.cumMeanAbsError = ((obj.numUpdates - 1) * obj.cumMeanAbsError + (chainErrorSum / obj.numOfInventories)) / obj.numUpdates;

            % Step 3: Inventories compute orders based on downstream demand
            for i = 1:1:obj.numOfInventories
                obj.inventories{i}.computeOrder();  % Compute order in values
            end
            
            % Step 4: Supplier provides products based on the first inventory's request
            obj.supplier.supplyProducts(); % Compute product out
            
            % Step 5: Update state of all inventories and demander
            for i = 1:obj.numOfInventories
                obj.inventories{i}.updateState();    % Compute product outs
            end
            
            % Step 6: Transport products through each PhyLink
            for i = 1:(obj.numOfInventories+1)
                obj.phyLinks{i}.transportGoods(); % Computer product ins at downstream
            end

            % Step 6: Statistics
            obj.demander.updateState();

            
        end

        function plotPerformance(obj)
            % Number of inventories in the chain
            numOfInventories = obj.numOfInventories;
            
            % Create figure for this chain's performance
            figure;
            sgtitle(['Performance Metrics for Chain ', num2str(obj.chainId)]);
    
            % Plot 1: Tracking Error History
            subplot(3, 1, 1);
            hold on; grid on;
            for i = 1:numOfInventories
                plot(obj.inventories{i}.trackingErrorHistory, 'LineWidth', 2);
            end
            title('Tracking Error');
            xlabel('Time Step');
            ylabel('Error');
            legend(arrayfun(@(i) ['Inv ', num2str(i)], 1:numOfInventories, 'UniformOutput', false));
            hold off;
    
            % Plot 2: Inventory Level History
            subplot(3, 1, 2);
            hold on; grid on;
            for i = 1:numOfInventories
                plot(obj.inventories{i}.inventoryHistory, 'LineWidth', 2);
            end
            title('Inventory Level');
            xlabel('Time Step');
            ylabel('Inventory');
            legend(arrayfun(@(i) ['Inv ', num2str(i)], 1:numOfInventories, 'UniformOutput', false));
            hold off;
    
            % Plot 3: Order Magnitudes (In and Out)
            subplot(3, 1, 3);
            hold on; grid on;
            for i = 1:numOfInventories
                plot(obj.inventories{i}.orderInHistory, '--', 'LineWidth', 2);  % Incoming orders (dashed line)
                plot(obj.inventories{i}.orderOutHistory, 'LineWidth', 2);       % Outgoing orders (solid line)
            end
            title('Order Magnitudes');
            xlabel('Time Step');
            ylabel('Order Magnitude');
            legend(arrayfun(@(i) ['Inv ', num2str(i)], 1:numOfInventories, 'UniformOutput', false));
            hold off;
        end

        function outputArg = draw(obj)
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