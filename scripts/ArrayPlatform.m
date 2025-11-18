classdef ArrayPlatform < handle
    % ArrayPlatform Class: Manages the geometry and kinematics of the antenna array.
    
    properties
        % Physical layout of the array elements relative to the platform's origin.
        % A [N_phy x 3] matrix, where N_phy is the number of physical elements.
        physical_elements 
        
        % MIMO configuration
        tx_indices % Indices of elements used as transmitters
        rx_indices % Indices of elements used as receivers
        
        % Platform trajectory function handle. 
        % It takes time 't' as input and returns a struct with position and orientation.
        % e.g., trajectory = @(t) struct('position', [vx*t, vy*t, vz*t], 'orientation', [roll, pitch, yaw]);
        trajectory_func
    end
    
    methods
        function obj = ArrayPlatform(physical_elements, tx_indices, rx_indices)
            % Constructor: Initializes the array with its physical layout.
            if nargin > 0
                obj.physical_elements = physical_elements;
                obj.tx_indices = tx_indices;
                obj.rx_indices = rx_indices;
                % Default to a static trajectory
                obj.trajectory_func = @(t) struct('position', [0, 0, 0], 'orientation', [0, 0, 0]);
            end
        end
        
        function state = get_platform_state(obj, t)
            % Returns the platform's state (position and orientation) at time t
            state = obj.trajectory_func(t);
        end
        
        function num = get_num_virtual_elements(obj)
            %GET_NUM_VIRTUAL_ELEMENTS Returns the total number of MIMO virtual elements.
            num = numel(obj.tx_indices) * numel(obj.rx_indices);
        end
        
        function num = get_num_tx(obj)
            num = numel(obj.tx_indices);
        end
        
        function num = get_num_rx(obj)
            num = numel(obj.rx_indices);
        end
        
        function traj = get_trajectory(obj)
            % Returns the trajectory function handle
            traj = obj.trajectory_func;
        end
        
        function obj = set_trajectory(obj, trajectory_func)
            % Method to define the motion of the platform.
            obj.trajectory_func = trajectory_func;
        end
        
        function positions = get_physical_positions_local(obj, t)
            % GET_PHYSICAL_POSITIONS Calculates the physical positions of all elements
            % in the PLATFORM'S LOCAL coordinate frame at a given time t.
            
            state = obj.trajectory_func(t);
            orientation = state.orientation; % [roll, pitch, yaw] in degrees
            
            % --- 3D Rotation ---
            % This part calculates the rotated positions of the elements relative
            % to the platform's origin.
            Rz = [cosd(orientation(3)) -sind(orientation(3)) 0; sind(orientation(3)) cosd(orientation(3)) 0; 0 0 1];
            Ry = [cosd(orientation(2)) 0 sind(orientation(2)); 0 1 0; -sind(orientation(2)) 0 cosd(orientation(2))];
            Rx = [1 0 0; 0 cosd(orientation(1)) -sind(orientation(1)); 0 sind(orientation(1)) cosd(orientation(1))];
            R = Rz * Ry * Rx;
            
            % Apply rotation to the base element positions
            positions = (R * obj.physical_elements')';
        end

        function virtual_positions = get_mimo_virtual_positions(obj, t)
            % GET_MIMO_VIRTUAL_POSITIONS Calculates the virtual element positions
            % in the GLOBAL coordinate frame at a given time t.
            
            state = obj.trajectory_func(t);
            platform_pos = state.position;
            
            % Get physical element positions in the GLOBAL frame
            physical_pos_local = obj.get_physical_positions_local(t);
            physical_pos_global = physical_pos_local + platform_pos;
            
            tx_positions = physical_pos_global(obj.tx_indices, :);
            rx_positions = physical_pos_global(obj.rx_indices, :);
            
            num_tx = size(tx_positions, 1);
            num_rx = size(rx_positions, 1);
            num_virtual = num_tx * num_rx;
            virtual_positions = zeros(num_virtual, 3);
            
            % --- 修正：虚拟阵元排序必须与SignalGenerator中的RDC reshape顺序一致 ---
            % SignalGenerator使用permute([1,2,4,3])，导致顺序为：Tx在内层，Rx在外层
            % 即：(tx1,rx1), (tx2,rx1), ..., (txN,rx1), (tx1,rx2), ..., (txN,rxM)
            idx = 1;
            for j = 1:num_rx  % Rx在外层循环
                for i = 1:num_tx  % Tx在内层循环
                    % The virtual element position is the midpoint of the Tx-Rx pair
                    virtual_positions(idx, :) = (tx_positions(i, :) + rx_positions(j, :)) / 2;
                    idx = idx + 1;
                end
            end
        end
    end
    
    methods(Static)
        function obj = create_ula(num_elements, spacing)
            % Factory method to create a Uniform Linear Array (ULA) along the x-axis.
            x_pos = ((0:num_elements-1) - (num_elements-1)/2) * spacing;
            y_pos = zeros(1, num_elements);
            z_pos = zeros(1, num_elements);
            elements = [x_pos', y_pos', z_pos'];
            
            % Default: first element is TX, all elements are RX (SIMO)
            tx = 1;
            rx = 1:num_elements;
            obj = ArrayPlatform(elements, tx, rx);
        end
    end
end
