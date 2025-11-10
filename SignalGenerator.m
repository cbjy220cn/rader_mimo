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
            % --- FMCW Signal Generation (Physically Correct Version) ---
            
            % Unpack parameters
            num_snapshots = length(t_axis); % Number of chirps (slow time)
            num_targets = length(obj.targets);
            num_tx = obj.array_platform.get_num_tx();
            num_rx = obj.array_platform.get_num_rx();
            num_virtual_elements = num_tx * num_rx;
            
            % --- Decorrelation Fix for Coherent Targets ---
            % Even with FMCW, targets at the same range can be highly coherent in slow-time.
            % We multiply each target's signal by a unique random sequence over the slow-time
            % axis to ensure they are uncorrelated, which is a prerequisite for standard MUSIC.
            baseband_signals = (randn(num_targets, num_snapshots) + 1j * randn(num_targets, num_snapshots)) / sqrt(2);
            
            % Fast time axis (within one chirp)
            t_fast = (0:obj.radar_params.num_samples-1) / obj.radar_params.fs;
            
            % RDC stored as [samples x chirps x rx x tx]
            rdc = zeros(obj.radar_params.num_samples, num_snapshots, num_rx, num_tx);
            
            % Loop over each chirp (slow time)
            for k = 1:num_snapshots
                t_slow = t_axis(k);
                
                % Get all physical element positions for this chirp
                all_physical_pos_t = obj.array_platform.get_physical_positions(t_slow);
                tx_positions = all_physical_pos_t(obj.array_platform.tx_indices, :);
                rx_positions = all_physical_pos_t(obj.array_platform.rx_indices, :);
                
                % Loop over each target
                for i = 1:num_targets
                    target_obj = obj.targets{i};
                    target_pos_t = target_obj.get_position_at(t_slow);
                    
                    % Loop over each physical Tx-Rx path
                    for tx_idx = 1:num_tx
                        for rx_idx = 1:num_rx
                            % Calculate true physical path length and delay
                            range = norm(tx_positions(tx_idx, :) - target_pos_t) + ...
                                    norm(rx_positions(rx_idx, :) - target_pos_t);
                            tau = range / obj.radar_params.c;
                            
                            % --- Definitive Doppler Calculation using Numerical Differentiation ---
                            % The true Doppler shift comes from the rate of change of the path length.
                            % We approximate this by finding the range at a slightly later time.
                            dt = 1e-7; % A small time step for differentiation
                            t_plus_dt = t_slow + dt;

                            % Get positions at t+dt
                            all_physical_pos_dt = obj.array_platform.get_physical_positions(t_plus_dt);
                            tx_positions_dt = all_physical_pos_dt(obj.array_platform.tx_indices, :);
                            rx_positions_dt = all_physical_pos_dt(obj.array_platform.rx_indices, :);
                            target_pos_dt = target_obj.get_position_at(t_plus_dt);
                            
                            % Calculate range at t+dt
                            range_dt = norm(tx_positions_dt(tx_idx, :) - target_pos_dt) + ...
                                        norm(rx_positions_dt(rx_idx, :) - target_pos_dt);
                            
                            % Calculate range rate and the true Doppler frequency
                            range_rate = (range_dt - range) / dt;
                            doppler_freq = -range_rate / obj.radar_params.lambda; % Correct Doppler formula
                            
                            % --- DEFINITIVE FIX: Double-Counting Motion ---
                            % The time-varying geometric path length 'tau' (calculated from 'range')
                            % already contains ALL information about the phase changes due to motion.
                            % Adding an extra 'doppler_phase' term based on range_rate is redundant
                            % and incorrect in this high-fidelity simulation. It corrupts the signal.
                            % We therefore remove it completely.
                            % doppler_phase = 2 * pi * doppler_freq * t_slow;
                            
                            % Generate beat signal for this path
                            beat_freq = obj.radar_params.slope * tau;
                            start_phase = 2 * pi * (-obj.radar_params.fc * tau ...
                                         + 0.5 * obj.radar_params.slope * tau^2);
                            
                            beat_signal = target_obj.rcs * exp(1j * (2*pi*beat_freq*t_fast + start_phase));
                            
                            % Apply decorrelating baseband signal (optional but good practice)
                            beat_signal = beat_signal * baseband_signals(i, k);
                            
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
            
            % Apply correction to all chirps and antennas
            range_fft_corrected = range_fft .* rvp_correction_factor;

            % --- Snapshot Extraction ---
            target1_pos = obj.targets{1}.get_position_at(0);
            range_true = norm(target1_pos);
            range_bin = round(range_true / obj.radar_params.range_res) + 1;
            
            % Clamp range_bin to be within valid FFT indices
            range_bin = max(1, min(range_bin, obj.radar_params.num_samples));
            
            target_data_slice = range_fft_corrected(range_bin, :, :);
            snapshots = squeeze(target_data_slice).';
 
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
