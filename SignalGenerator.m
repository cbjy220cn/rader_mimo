classdef SignalGenerator
    % SignalGenerator Class: Generates radar snapshots for a given scenario.
    
    properties
        radar_params   % Struct with radar parameters (fc, B, fs, etc.)
        array_platform % An instance of the ArrayPlatform class
        targets        % A list of Target objects
    end
    
    methods
        function obj = SignalGenerator(radar_params, array_platform, targets)
            % Constructor
            if nargin > 0
                obj.radar_params = radar_params;
                obj.array_platform = array_platform;
                obj.targets = targets;
            end
        end
        
        function snapshots = generate_snapshots(obj, t_axis, snr_db)
            % Generates a series of snapshots for the given time axis.
            % t_axis: A vector of time points (e.g., start of each chirp).
            % snr_db: Signal-to-Noise Ratio in dB.
            
            num_snapshots = length(t_axis);
            
            % Get reference virtual positions at t=0 to determine array size
            ref_virtual_pos = obj.array_platform.get_mimo_virtual_positions(0);
            num_virtual_elements = size(ref_virtual_pos, 1);
            
            snapshots = zeros(num_virtual_elements, num_snapshots);
            
            lambda = physconst('LightSpeed') / obj.radar_params.fc;
            
            for k = 1:num_snapshots
                t = t_axis(k);
                snapshot_k = zeros(num_virtual_elements, 1);
                
                % Get the instantaneous virtual element positions at time t
                virtual_positions_t = obj.array_platform.get_mimo_virtual_positions(t);
                
                % Accumulate signal from all targets
                for i = 1:length(obj.targets)
                    target_pos_t = obj.targets(i).get_position_at(t);
                    
                    % --- BUG FIX: Replaced inconsistent spherical wave model ---
                    % with a far-field plane wave model to match the estimator.
                    
                    % Common path length to origin and true direction vector
                    dist_to_origin = norm(target_pos_t);
                    u_target = target_pos_t(:) / dist_to_origin; % Ensure column vector

                    % Calculate relative phase for each virtual element based on plane wave
                    % This convention MUST match the DoaEstimator's steering vector: exp(j*k*p*u)
                    relative_phase = 2 * pi / lambda * (virtual_positions_t * u_target);
                    
                    % Simplified amplitude for far-field
                    amplitude = sqrt(obj.targets(i).rcs) / (dist_to_origin^2);
                    
                    % Add this target's contribution to the snapshot
                    % The common phase exp(-j*2*pi*dist_to_origin/lambda) is omitted 
                    % as it does not affect the covariance matrix eigenvectors.
                    % --- Definitive Fix based on first principles derivation ---
                    % The relative phase of the baseband signal for a two-way radar path
                    % is exp(+j * k * (p_t+p_r) . u). The previous negative sign was incorrect.
                    snapshot_k = snapshot_k + amplitude * exp(1j * relative_phase);
                end
                
                snapshots(:, k) = snapshot_k;
            end
            
            % Add complex white Gaussian noise
            if isfinite(snr_db)
                signal_power = mean(abs(snapshots(:)).^2);
                noise_power = signal_power / (10^(snr_db / 10));
                noise = sqrt(noise_power/2) * (randn(size(snapshots)) + 1j * randn(size(snapshots)));
                snapshots = snapshots + noise;
            end
        end
    end
end
