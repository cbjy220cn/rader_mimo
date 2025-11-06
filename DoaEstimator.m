classdef DoaEstimator
    % DoaEstimator Class: Implements DOA estimation algorithms.
    
    properties
        array_platform % An instance of the ArrayPlatform class
        radar_params   % Struct with radar parameters
    end
    
    methods
        function obj = DoaEstimator(array_platform, radar_params)
            % Constructor
            if nargin > 0
                obj.array_platform = array_platform;
                obj.radar_params = radar_params;
            end
        end
        
        function [spectrum, grid] = estimate_gmusic(obj, snapshots, t_axis, num_sources, search_grid)
            % Estimates the DOA using the Generalized MUSIC algorithm for moving arrays.
            % snapshots: The [N_virt x M] matrix of received signals.
            % t_axis: The [1 x M] vector of time points for each snapshot.
            % num_sources: The assumed number of signal sources.
            % search_grid: A struct with fields 'theta' and 'phi' vectors for the search space.
            
            [num_virtual_elements, num_snapshots] = size(snapshots);
            
            % 1. Calculate the sample covariance matrix
            Rxx = (snapshots * snapshots') / num_snapshots;
            
            % 2. Eigendecomposition to find the noise subspace
            [eigenvectors, eigenvalues] = eig(Rxx);
            [~, sorted_indices] = sort(diag(eigenvalues));
            noise_indices = sorted_indices(1:num_virtual_elements - num_sources);
            Qn = eigenvectors(:, noise_indices);
            
            % 3. Search the angle space
            theta_search = search_grid.theta;
            phi_search = search_grid.phi;
            spectrum = zeros(length(theta_search), length(phi_search));
            
            lambda = physconst('LightSpeed') / obj.radar_params.fc;
            
            % Pre-calculate all positions for all snapshots to build the GSV
            all_positions = zeros(num_virtual_elements, 3, num_snapshots);
            for k = 1:num_snapshots
                all_positions(:,:,k) = obj.array_platform.get_mimo_virtual_positions(t_axis(k));
            end

            for i = 1:length(theta_search)
                for j = 1:length(phi_search)
                    theta = theta_search(i);
                    phi = phi_search(j);
                    
                    % Unit vector for the current search direction
                    % --- Reverting BUG FIX 3 ---
                    % The coordinate system should be consistent. The bug was in the
                    % signal generation phase sign, not here.
                    u = [cosd(phi)*sind(theta); sind(phi)*sind(theta); cosd(theta)];
                    
                    % 4. Construct the Generalized Steering Vector (GSV) Matrix A(u)
                    A_u = zeros(num_virtual_elements, num_snapshots);
                    for k = 1:num_snapshots
                        positions_k = all_positions(:,:,k);
                        % --- BUG FIX: Sign was inconsistent with the physical signal model.
                        % The phase should be positive in the exponent for the exp(j*k*r) convention.
                        phase = 2 * pi / lambda * (positions_k * u);
                        A_u(:, k) = exp(1j * phase);
                    end
                    
                    % 5. Calculate the MUSIC spectrum value
                    denominator = trace(A_u' * (Qn * Qn') * A_u);
                    spectrum(i, j) = 1 / abs(denominator);
                end
            end
            
            grid = search_grid;
        end
    end
end
