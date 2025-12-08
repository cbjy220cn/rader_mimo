classdef DoaEstimator
    % DoaEstimator Class: Implements DOA estimation algorithms.
    
    properties
        array_platform % An instance of the ArrayPlatform class
        radar_params   % Struct with radar parameters
        lambda         % Wavelength
        use_gpu        % Flag to enable GPU acceleration
    end
    
    methods
        function obj = DoaEstimator(array_platform, radar_params)
            % Constructor
            if nargin > 0
                obj.array_platform = array_platform;
                obj.radar_params = radar_params;
                obj.lambda = physconst('LightSpeed') / radar_params.fc;
                
                % --- GPU Acceleration Check ---
                obj.use_gpu = (gpuDeviceCount > 0);
                if obj.use_gpu
                    % Use a persistent variable to ensure this message only prints once
                    % per MATLAB session, which is useful inside parfor loops.
                    persistent has_shown_gpu_msg;
                    if isempty(has_shown_gpu_msg)
                        fprintf('GPU detected. MUSIC spectrum calculation will be accelerated.\n');
                        has_shown_gpu_msg = true;
                    end
                end
            end
        end
        
        function [spectrum, A_u_out] = estimate_gmusic(obj, snapshots, t_axis, num_targets, search_grid, u_debug, positions_override)
            %ESTIMATE_GMUSIC Implements the Generalized MUSIC algorithm.
            %   ... (existing documentation)
            %   u_debug (optional): A 3x1 unit vector. If provided, the function
            %                     will return the steering vector A_u for this
            %                     specific direction and skip the full spectrum search.
            %   positions_override (optional): [M x 3] matrix, directly specify array positions
            
            A_u_out = []; % Default output

            [num_virtual_elements, num_snapshots] = size(snapshots);
            
            if obj.use_gpu
                snapshots = gpuArray(snapshots);
            end

            % 1. Calculate the sample covariance matrix
            Rxx = (snapshots * snapshots') / num_snapshots;
            
            % 2. Eigendecomposition to find the noise subspace
            [eigenvectors, eigenvalues] = eig(Rxx);
            [~, sorted_indices] = sort(diag(eigenvalues));
            noise_indices = sorted_indices(1:num_virtual_elements - num_targets);
            Qn = eigenvectors(:, noise_indices);
            
            % Pre-calculate for efficiency
            Qn_proj = Qn * Qn'; % This is now on the GPU if enabled
            
            % --- Special Debug Mode ---
            if nargin > 5 && ~isempty(u_debug)
                if nargin > 6 && ~isempty(positions_override)
                    A_u_out = obj.build_steering_matrix_internal(t_axis, u_debug, positions_override);
                else
                    A_u_out = obj.build_steering_matrix_internal(t_axis, u_debug, []);
                end
                spectrum = []; % Skip spectrum calculation
                return;
            end
            
            % 3. Search the angle space
            theta_search = search_grid.theta;
            phi_search = search_grid.phi;
            spectrum = zeros(length(theta_search), length(phi_search));
            
            % Determine if using override positions (static array mode)
            use_override = (nargin > 6 && ~isempty(positions_override));
            
            if ~use_override
                % Pre-calculate all positions for all snapshots (SAR mode)
                all_positions = zeros(num_virtual_elements, 3, num_snapshots);
                for k = 1:num_snapshots
                    all_positions(:,:,k) = obj.array_platform.get_mimo_virtual_positions(t_axis(k));
                end
            end

            % Loop through all search directions
            for phi_idx = 1:numel(phi_search)
                phi = phi_search(phi_idx);
                for theta_idx = 1:numel(theta_search)
                    theta = theta_search(theta_idx);
                    
                    u = [sind(theta)*cosd(phi); sind(theta)*sind(phi); cosd(theta)];
                    
                    if use_override
                        A_u = obj.build_steering_matrix_internal(t_axis, u, positions_override);
                    else
                        A_u = obj.build_steering_matrix_internal(t_axis, u, []);
                    end
                    
                    % ⚠️ 关键修复：归一化导向矢量
                    A_u = A_u / norm(A_u, 'fro');
                    
                    if obj.use_gpu
                        A_u = gpuArray(A_u); % Transfer steering vector to GPU
                    end
                    
                    % 5. Calculate the MUSIC spectrum value
                    denominator = trace(A_u' * (Qn_proj * A_u));
                    spectrum(theta_idx, phi_idx) = 1 / abs(denominator);
                    
                    if obj.use_gpu
                        spectrum(theta_idx, phi_idx) = gather(spectrum(theta_idx, phi_idx)); % Get result back from GPU
                    end
                end
            end
            
            grid = search_grid;
        end
        
        function A_u = build_steering_matrix_internal(obj, t_axis, u, positions_override)
            % Builds the generalized steering matrix A(u) for a given direction u.
            % A(u) is a [M x K] matrix, where M is num_virtual_elements and K is num_snapshots.
            %
            % positions_override: [M x 3] matrix, if provided, uses these positions
            %                     instead of querying array_platform (for static arrays)
            %
            % 合成孔径模式：
            % - 每个时刻的阵列位置不同（全局坐标系）
            % - 目标方向u在全局坐标系中固定
            % - 导向矢量反映从不同位置观测同一目标的相位关系
            
            num_snapshots = numel(t_axis);
            lambda = obj.radar_params.lambda;
            
            if nargin > 3 && ~isempty(positions_override)
                % Static array mode: use provided positions for all snapshots
                num_virtual_elements = size(positions_override, 1);
                A_u = zeros(num_virtual_elements, num_snapshots);
                
                phase = 4 * pi / lambda * (positions_override * u);
                for k = 1:num_snapshots
                    A_u(:, k) = exp(1j * phase);  % Same for all snapshots (static)
                end
            else
                % SAR mode: query positions for each snapshot
                num_virtual_elements = obj.array_platform.get_num_virtual_elements();
                A_u = zeros(num_virtual_elements, num_snapshots);
                
                for k = 1:num_snapshots
                    % 获取虚拟元素在全局坐标系的位置
                    positions_k = obj.array_platform.get_mimo_virtual_positions(t_axis(k));
                    
                    % 在全局坐标系中计算相位
                    phase = 4 * pi / lambda * (positions_k * u);
                    A_u(:, k) = exp(1j * phase);
                end
            end
        end
        
        % Backward compatibility wrapper
        function A_u = build_steering_matrix(obj, t_axis, u)
            A_u = obj.build_steering_matrix_internal(t_axis, u, []);
        end

    end
    
    methods (Static)
        function [theta_peaks, phi_peaks, peak_vals] = find_peaks(spectrum, grid, num_peaks)
            % A simple peak finder to locate the N largest peaks in the spectrum.
            % For more robust peak finding, consider using imregionalmax or other
            % more advanced image processing functions.

            [sorted_vals, sort_idx] = sort(spectrum(:), 'descend');

            theta_peaks = zeros(1, num_peaks);
            phi_peaks = zeros(1, num_peaks);
            peak_vals = zeros(1, num_peaks);

            for i = 1:num_peaks
                peak_idx = sort_idx(i);
                peak_vals(i) = sorted_vals(i);
                [theta_idx, phi_idx] = ind2sub(size(spectrum), peak_idx);
                theta_peaks(i) = grid.theta(theta_idx);
                phi_peaks(i) = grid.phi(phi_idx);
            end
        end
    end
end
