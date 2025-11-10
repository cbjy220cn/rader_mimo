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
        
        function obj = set_trajectory(obj, trajectory_func)
            % Method to define the motion of the platform.
            obj.trajectory_func = trajectory_func;
        end
        
        function positions = get_physical_positions(obj, t)
            % Calculates the absolute positions of all physical elements at a given time t.
            platform_state = obj.trajectory_func(t);
            platform_pos = platform_state.position;
            platform_orientation_deg = platform_state.orientation; % [roll, pitch, yaw] in degrees

            % --- FIX: Implement 3D rotation based on orientation ---
            % Convert orientation from degrees to radians for trigonometric functions
            orientation_rad = deg2rad(platform_orientation_deg);
            alpha = orientation_rad(3); % Yaw around Z-axis
            beta = orientation_rad(2);  % Pitch around Y-axis
            gamma = orientation_rad(1); % Roll around X-axis

            % Create rotation matrices for Z-Y-X convention
            Rz = [cos(alpha), -sin(alpha), 0;
                  sin(alpha),  cos(alpha), 0;
                  0,           0,          1];
            
            Ry = [cos(beta),  0, sin(beta);
                  0,          1, 0;
                 -sin(beta), 0, cos(beta)];

            Rx = [1, 0,           0;
                  0, cos(gamma), -sin(gamma);
                  0, sin(gamma),  cos(gamma)];
            
            % Combined rotation matrix
            R = Rz * Ry * Rx;

            % Apply rotation to the relative element positions, then add platform's position.
            % Note: physical_elements is [N x 3], so we need to transpose it for matrix
            % multiplication with R [3 x 3], and then transpose back.
            rotated_elements = (R * obj.physical_elements')';
            
            positions = rotated_elements + platform_pos;
        end
        
        function virtual_positions = get_mimo_virtual_positions(obj, t)
            % Calculates the positions of the MIMO virtual elements at a given time t.
            num_tx = length(obj.tx_indices);
            num_rx = length(obj.rx_indices);
            
            % Get absolute positions of all physical elements at time t
            all_physical_pos = obj.get_physical_positions(t);
            
            tx_pos = all_physical_pos(obj.tx_indices, :);
            rx_pos = all_physical_pos(obj.rx_indices, :);
            
            % Generate virtual element positions using convolution (p_v = p_t + p_r)
            virtual_positions = zeros(num_tx * num_rx, 3);
            count = 1;
            for i = 1:num_tx
                for j = 1:num_rx
                    virtual_positions(count, :) = tx_pos(i, :) + rx_pos(j, :);
                    count = count + 1;
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
