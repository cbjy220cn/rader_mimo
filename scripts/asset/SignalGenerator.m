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
        
        function [snapshots, debug_info] = generate_snapshots(obj, t_axis, snr_db, is_debug_mode)
            % --- FMCW Signal Generation (Physically Correct Version) ---
            
            % --- DEFINITIVE FIX: Make randomness repeatable for debugging ---
            % By fixing the seed here, every call to this function will produce
            % the exact same "random" baseband signal, making debugging deterministic.
            rng(0);
            
            % Handle optional debug argument
            if nargin < 4
                is_debug_mode = false;
            end
            debug_info = []; % Default return value
            
            % Unpack parameters
            num_snapshots = length(t_axis); % Number of chirps (slow time)
            num_targets = length(obj.targets);
            num_tx = obj.array_platform.get_num_tx();
            num_rx = obj.array_platform.get_num_rx();
            num_virtual_elements = num_tx * num_rx;
            
            % --- DEFINITIVE FIX: Random baseband signal should be constant per target, not per snapshot ---
            % The old implementation created a (num_targets, num_snapshots) matrix,
            % destroying phase coherence across snapshots.
            % The correct implementation is a (num_targets, 1) vector, so each
            % target gets one random phase that is constant for the entire CPI.
            baseband_signals = (randn(num_targets, 1) + 1j * randn(num_targets, 1)) / sqrt(2);
            
            % Fast time axis (within one chirp)
            t_fast = (0:obj.radar_params.num_samples-1) / obj.radar_params.fs;
            
            % RDC stored as [samples x chirps x rx x tx]
            rdc = zeros(obj.radar_params.num_samples, num_snapshots, num_rx, num_tx);
            
            % Loop over each chirp (slow time)
            for k = 1:num_snapshots
                t_slow = t_axis(k);
                
                % Get platform's state (position and orientation) at this time
                platform_state_t = obj.array_platform.get_platform_state(t_slow);
                platform_pos_t = platform_state_t.position;

                % Get all physical element positions (these are relative to the platform's origin)
                all_physical_pos_local = obj.array_platform.get_physical_positions_local(t_slow);
                
                % --- DEFINITIVE FIX: Convert to Global Coordinates ---
                % We must add the platform's own position to the elements' local
                % positions to get their absolute positions in the world frame.
                tx_positions_global = all_physical_pos_local(obj.array_platform.tx_indices, :) + platform_pos_t;
                rx_positions_global = all_physical_pos_local(obj.array_platform.rx_indices, :) + platform_pos_t;
                
                % Loop over each target
                for i = 1:num_targets
                    target_obj = obj.targets{i};
                    target_pos_t = target_obj.get_position_at(t_slow);
                    
                    % Loop over each physical Tx-Rx path
                    for tx_idx = 1:num_tx
                        for rx_idx = 1:num_rx
                            % Calculate true physical path length and delay using GLOBAL positions
                            range = norm(tx_positions_global(tx_idx, :) - target_pos_t) + ...
                                    norm(rx_positions_global(rx_idx, :) - target_pos_t);
                            tau = range / obj.radar_params.c;
                            
                            % --- Definitive Doppler Calculation using Numerical Differentiation ---
                            dt = 1e-7; % A small time step for differentiation
                            t_plus_dt = t_slow + dt;

                            % Get platform state and element positions at t+dt
                            platform_state_dt = obj.array_platform.get_platform_state(t_plus_dt);
                            platform_pos_dt = platform_state_dt.position;
                            all_physical_pos_local_dt = obj.array_platform.get_physical_positions_local(t_plus_dt);
                            
                            tx_positions_global_dt = all_physical_pos_local_dt(obj.array_platform.tx_indices, :) + platform_pos_dt;
                            rx_positions_global_dt = all_physical_pos_local_dt(obj.array_platform.rx_indices, :) + platform_pos_dt;
                            target_pos_dt = target_obj.get_position_at(t_plus_dt);
                            
                            % Calculate range at t+dt using GLOBAL positions
                            range_dt = norm(tx_positions_global_dt(tx_idx, :) - target_pos_dt) + ...
                                       norm(rx_positions_global_dt(rx_idx, :) - target_pos_dt);
                            
                            % Calculate range rate and the true Doppler frequency
                            range_rate = (range_dt - range) / dt;
                            doppler_freq = -range_rate / obj.radar_params.lambda; 

                            % --- THE DEFINITIVE, FINAL FIX: Beat Frequency Sign ---
                            % The previous implementation produced a negative beat frequency, while the
                            % subsequent range_bin extraction logic assumed a positive frequency.
                            % This mismatch meant we were feeding pure noise to the MUSIC algorithm.
                            % We now correct the sign to ensure the simulated signal matches the
                            % processing chain, which is the standard convention in most literature.
                            beat_freq = obj.radar_params.slope * tau;
                            
                            % The total phase of the beat signal at the beginning of the chirp (t_fast=0)
                            % consists of the geometric path length phase and the RVP component.
                            start_phase = 2 * pi * (-obj.radar_params.fc * tau ...
                                         + 0.5 * obj.radar_params.slope * tau^2);

                            % --- Capture Debug Info ---
                            if is_debug_mode && tx_idx == 1 && rx_idx == 1
                                if k == 1 % First snapshot
                                    debug_info.t0.tx_pos = tx_positions_global(tx_idx, :);
                                    debug_info.t0.rx_pos = rx_positions_global(rx_idx, :);
                                    debug_info.t0.range = range;
                                    debug_info.t0.tau = tau;
                                    debug_info.t0.geom_phase_component = -2 * pi * obj.radar_params.fc * tau;
                                    debug_info.t0.rvp_phase_component = 2 * pi * 0.5 * obj.radar_params.slope * tau^2;
                                    debug_info.t0.baseband_phase_component = angle(baseband_signals(i)); % Capture the random phase
                                elseif k == num_snapshots % Last snapshot
                                    debug_info.t_end.tx_pos = tx_positions_global(tx_idx, :);
                                    debug_info.t_end.rx_pos = rx_positions_global(rx_idx, :);
                                    debug_info.t_end.range = range;
                                    debug_info.t_end.tau = tau;
                                    debug_info.t_end.geom_phase_component = -2 * pi * obj.radar_params.fc * tau;
                                    debug_info.t_end.rvp_phase_component = 2 * pi * 0.5 * obj.radar_params.slope * tau^2;
                                    debug_info.t_end.baseband_phase_component = angle(baseband_signals(i)); % Capture the random phase
                                end
                            end
                            
                            beat_signal = target_obj.rcs * exp(1j * (start_phase + 2*pi*beat_freq*t_fast));
                            
                            % Apply random baseband signal for target decorrelation
                            % --- DEFINITIVE FIX: Use baseband_signals(i) not baseband_signals(i,k) ---
                            beat_signal = beat_signal * baseband_signals(i);
                            
                            % Add to the RDC
                            rdc(:, k, rx_idx, tx_idx) = rdc(:, k, rx_idx, tx_idx) + beat_signal.';
                        end
                    end
                end
            end
            
            % Reshape RDC to [samples x chirps x virtual_elements]
            % The order must match how the virtual array is constructed elsewhere.
            % Assuming Tx-major order: [rx1-tx1, rx2-tx1, ..., rxN-txM]
            rdc_reshaped = reshape(permute(rdc, [1, 2, 4, 3]), ...
                                   [obj.radar_params.num_samples, num_snapshots, num_virtual_elements]);
            
            % --- Signal Processing: Range-FFT ---
            range_fft = fft(rdc_reshaped, [], 1);
            
            % --- RVP (Residual Video Phase) Correction ---
            % --- DEFINITIVE FIX: The RVP term is exp(j * pi * slope * tau^2).
            % We must apply its conjugate to remove it, leaving the desired geometric term.
            range_bins = (0:obj.radar_params.num_samples-1)';
            freq_grid = (0:obj.radar_params.num_samples-1) * obj.radar_params.fs / obj.radar_params.num_samples;
            time_delay_per_bin = freq_grid / obj.radar_params.slope; % This is tau for each range bin
            
            % The phase error to be removed is pi * slope * tau^2
            rvp_phase_error = pi * obj.radar_params.slope * time_delay_per_bin.^2;
            rvp_correction_factor = exp(-1j * rvp_phase_error.');
            
            % --- DEFINITIVE FIX: Broadcasting Shape ---
            % rvp_correction_factor must be a column vector [num_samples x 1] to
            % correctly broadcast across the 2nd and 3rd dimensions (chirps and antennas)
            % of the range_fft matrix. A row vector would be applied incorrectly.
            rvp_correction_factor = reshape(rvp_correction_factor, [], 1);

            % Apply correction to all chirps and antennas
            range_fft_corrected = range_fft .* rvp_correction_factor;

            % --- Snapshot Extraction ---
            % --- DEFINITIVE FIX: Use the correct range relative to the platform ---
            % The range bin must be calculated based on the distance from the platform's 
            % center at t=0 to the target, not from the world origin (0,0,0).
            platform_state_t0 = obj.array_platform.get_platform_state(t_axis(1));
            platform_pos_t0 = platform_state_t0.position;
            target_pos_t0 = obj.targets{1}.get_position_at(t_axis(1));
            
            % The one-way distance determines the range bin
            range_true = norm(target_pos_t0 - platform_pos_t0); 
            range_bin = round(range_true / obj.radar_params.range_res) + 1;
            
            % Clamp range_bin to be within valid FFT indices
            range_bin = max(1, min(range_bin, obj.radar_params.num_samples));
            
            target_data_slice = range_fft_corrected(range_bin, :, :);
            snapshots = squeeze(target_data_slice).';
 
            % --- Capture Final Complex Value for Debug ---
            if is_debug_mode
                % Get value for the first virtual element at t0 and t_end
                debug_info.t0.final_complex_val = range_fft_corrected(range_bin, 1, 1);
                debug_info.t_end.final_complex_val = range_fft_corrected(range_bin, num_snapshots, 1);
            end

            % Add noise if specified
            if isfinite(snr_db)
                signal_power = mean(abs(snapshots(:)).^2);
                noise_power = signal_power / (10^(snr_db/10));
                noise = (randn(size(snapshots)) + 1j * randn(size(snapshots))) * sqrt(noise_power/2);
                snapshots = snapshots + noise;
            end
        end
    end
end
