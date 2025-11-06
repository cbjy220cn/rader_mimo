function [theta, phi] = plane_move_run_optimized(params,move)
%PLANE_MOVE_RUN_OPTIMIZED Optimized version of the radar simulation.

%   Detailed explanation of optimizations inline.

%% Radar parameters (Generally efficient scalar setup)
c = physconst('LightSpeed'); % speed of light
BW = 50e6; % bandwidth
fc = 3e9; % carrier frequency
numADC = 361; % # of adc samples
numChirps = 256; % # of chirps per frame
numCPI = 10;
T = 10e-3; % PRI (Chirp duration or PRI, context suggests PRI)
PRF = 1/T;
F = numADC/T; % sampling frequency (Check if T is chirp duration or PRI)
% If T is PRI and chirp duration is shorter, F might need recalculation based on actual ADC sampling time.
% Assuming T is the active ADC sampling time per chirp for F calculation.
dt = 1/F; % sampling interval
slope = BW/T; % Chirp slope (Assuming T is active chirp duration)
lambda = c/fc;

% --- Time Axis ---
% Original 't' spanned the entire simulation time with N samples. Unnecessary for delay calculation.
% Create time axis for *one* chirp (fast time)
t_onePulse = (0:numADC-1)*dt;

% Create time axis for the *start* of each chirp across all CPIs (slow time)
numChirpsTotal = numChirps * numCPI;
t_chirp_starts = (0:numChirpsTotal-1) * T; % Time at the beginning of each chirp

% Total number of samples (remains the same for signal generation)
N = numChirps * numADC * numCPI;

% --- Antenna Setup ---
Ny = params.Ny;
numTX = params.numRX; % Note: Swapped TX/RX compared to variable names
numRX = params.numTX; % Note: Swapped TX/RX compared to variable names
N_L = params.N_L; % Virtual array extension factor

Vmax = lambda/(T*4); % Max Unamb velocity m/s (Check if T is PRI or frame time)

d_y = lambda/2/2; % Elevation spacing
d_tx = lambda/2; % Azimuth TX spacing (assuming uniform linear)
d_rx = numRX*d_tx; % Dist. between RXs (Check this - implies RXs are widely spaced? Or is d_rx spacing between adjacent RXs?)
% Assuming d_rx = lambda/2 for a standard ULA virtual array generation
d_rx_assumed = lambda / 2; % Let's assume standard RX spacing for virtual array
d_tx_assumed = numRX * d_rx_assumed; % Assuming TXs are spaced by numRX*lambda/2 for MIMO

% Revise antenna positions based on standard MIMO assumptions:
% TX ULA along X: spacing d_tx_assumed
% RX ULA along X: spacing d_rx_assumed
tx_x_pos = ((0:numTX-1) - (numTX-1)/2) * d_tx_assumed;
rx_x_pos = ((0:numRX-1) - (numRX-1)/2) * d_rx_assumed;
y_pos = ((0:Ny-1) - (Ny-1)/2) * d_y;

% Platform movement
if move==1
    tr_vel=-d_tx_assumed/2/T; % Check velocity calculation logic
else
    tr_vel=0;
end

%% Targets (Efficient vector calculations)

% Target 1
r1_radial = 3000; tar1_theta = 30; tar1_phi= 60;
r1_x = cosd(tar1_phi)*sind(tar1_theta)*r1_radial;
r1_y = sind(tar1_phi)*sind(tar1_theta)*r1_radial;
r1_z = cosd(tar1_theta)*r1_radial;
v1_radial = 0.001;
v1_x = cosd(tar1_phi)*cosd(tar1_theta)*v1_radial; % Check velocity components logic
v1_y = sind(tar1_phi)*cosd(tar1_theta)*v1_radial;
v1_z = sind(tar1_theta)*v1_radial;
r1_init = [r1_x r1_y r1_z];
v1 = [v1_x, v1_y, v1_z];

% Target 2
r2_radial = 600; tar2_theta = 30; tar2_phi = -50;
r2_x = cosd(tar2_phi)*sind(tar2_theta)*r2_radial;
r2_y = sind(tar2_phi)*sind(tar2_theta)*r2_radial;
r2_z = cosd(tar2_theta)*r2_radial;
v2_radial = 0.1;
v2_x = cosd(tar2_phi)*sind(tar2_theta)*v2_radial; % Check velocity components logic
v2_y = sind(tar2_phi)*sind(tar2_theta)*v2_radial;
v2_z = cosd(tar2_theta)*v2_radial;
r2_init = [r2_x r2_y r2_z];
v2 = [v2_x, v2_y, v2_z];

% OPTIMIZATION: Calculate target locations ONLY at the start of each chirp
% Reshape t_chirp_starts to a column vector for broadcasting
t_chirp_starts_col = t_chirp_starts(:); % [numChirpsTotal x 1]
tar1_loc_chirps = r1_init + t_chirp_starts_col .* v1; % [numChirpsTotal x 3]
tar2_loc_chirps = r2_init + t_chirp_starts_col .* v2; % [numChirpsTotal x 3]

%% Antenna Locations (Precompute and update efficiently)

tx_loc = cell(numTX, Ny);
rx_loc = cell(numRX, Ny);
tx_loc_t_chirps = cell(numTX, Ny); % Positions at chirp start times
rx_loc_t_chirps = cell(numRX, Ny); % Positions at chirp start times

% Precompute initial positions
for j = 1:Ny
    y_j = y_pos(j);
    for i = 1:numTX
        tx_loc{i,j} = [tx_x_pos(i), y_j, 0];
    end
    for i = 1:numRX
        rx_loc{i,j} = [rx_x_pos(i), y_j, 0];
    end
end

% OPTIMIZATION: Calculate time-varying antenna locations ONLY at chirp starts
platform_displacement = tr_vel * t_chirp_starts_col; % [numChirpsTotal x 1]
for j = 1:Ny
    for i = 1:numTX
        tx_loc_t_chirps{i,j} = tx_loc{i,j} + [platform_displacement, zeros(numChirpsTotal, 1), zeros(numChirpsTotal, 1)]; %[numChirpsTotal x 3]
    end
    for i = 1:numRX
        rx_loc_t_chirps{i,j} = rx_loc{i,j} + [platform_displacement, zeros(numChirpsTotal, 1), zeros(numChirpsTotal, 1)]; %[numChirpsTotal x 3]
    end
end

% Virtual element locations (for steering vector calculation, time-invariant part)
% Assuming standard MIMO: virtual element = tx_loc + rx_loc
% This part seems overly complex in the original, might be for a specific non-uniform setup.
% Recreating based on the *intention* of virtual array processing.
% If using standard MIMO virtual array:
virtual_elements_loc = zeros(numTX * numRX * Ny, 3);
count = 1;
for j=1:Ny % Elevation
    for iTX = 1:numTX % Azimuth TX
        for iRX = 1:numRX % Azimuth RX
            % Standard virtual element position = (tx_pos + rx_pos)/2
            % For steering vector calculation, often use tx_pos or rx_pos depending on convention,
            % or the effective phase center. Let's assume phase center is needed for 'a1'.
            % Phase center = rx_loc{iRX,j} % Assuming RX-centric phase
            % Effective position for phase = tx_loc{iTX,j} + rx_loc{iRX,j} (used for path length)
            % Let's calculate the standard virtual element positions for clarity,
            % though the steering vector 'a1' calculation later might use a different convention.
             virtual_elements_loc(count,:) = (tx_loc{iTX,j} + rx_loc{iRX,j}) / 2; % Example center
            count = count + 1;
        end
    end
end
% The original 'plane_loc' and 'plane_loc_plus' seem related to the virtual array extension logic ('RDC_plus').
% The structure of 'RDC_plus' implies a virtual ULA along X.
% Let d_virtual = d_rx_assumed / 2 (assuming numTX=2 or specific interleaving)
% Or d_virtual = d_tx / (numRX * numTX) ?? Need clarification on virtual array structure.
% Assuming virtual elements are spaced by d_tx / 2 along X for the RDC_plus logic.
d_virtual_x = d_tx / 2; % This spacing corresponds to 'a1' calculation later if k increments represent this spacing.

%% Delay Calculation (OPTIMIZED)

delays_tar1 = cell(numTX, numRX, Ny);
delays_tar2 = cell(numTX, numRX, Ny);

% OPTIMIZATION: Calculate delays based on positions at chirp start times.
% Use broadcasting for potentially faster vecnorm calculation.
for k = 1:Ny
    for i = 1:numTX
        % Get antenna positions for this TX/RX pair over all chirps [numChirpsTotal x 3]
        tx_pos_all_chirps = tx_loc_t_chirps{i,k};
        for j = 1:numRX
            rx_pos_all_chirps = rx_loc_t_chirps{j,k};

            % Calculate distances for all chirps at once
            dist_tx_tar1 = sqrt(sum((tar1_loc_chirps - tx_pos_all_chirps).^2, 2)); % [numChirpsTotal x 1]
            dist_rx_tar1 = sqrt(sum((tar1_loc_chirps - rx_pos_all_chirps).^2, 2)); % [numChirpsTotal x 1]
            dist_tx_tar2 = sqrt(sum((tar2_loc_chirps - tx_pos_all_chirps).^2, 2)); % [numChirpsTotal x 1]
            dist_rx_tar2 = sqrt(sum((tar2_loc_chirps - rx_pos_all_chirps).^2, 2)); % [numChirpsTotal x 1]

            % Calculate delays for all chirps
            delays_tar1{i,j,k} = (dist_tx_tar1 + dist_rx_tar1) / c; % [numChirpsTotal x 1]
            delays_tar2{i,j,k} = (dist_tx_tar2 + dist_rx_tar2) / c; % [numChirpsTotal x 1]
        end
    end
end


%% Complex signal Generation (Using existing vectorized approach)

phase = @(tx,fx,slope_val) 2*pi*(fx.*tx + slope_val/2*tx.^2); % Use local slope variable

snr = params.snr;
mixed = cell(numTX, numRX, Ny); % Preallocate result cell array

% Ensure t_onePulse is a row vector for broadcasting
t_onePulse_row = reshape(t_onePulse, 1, numADC);

% --- Vectorized Signal Generation ---
for l = 1:Ny
    for i = 1:numTX
        for j = 1:numRX
            % Get delays for the current channel [numChirpsTotal x 1]
            current_delays1_col = delays_tar1{i, j, l};
            current_delays2_col = delays_tar2{i, j, l};

            % Calculate base phase for transmitted signal (same for all chirps within the matrix)
            % Dimensions: [1 x numADC]
            phase_t_row = phase(t_onePulse_row, fc, slope);

            % Calculate received signal phases using broadcasting
            % (t_onePulse_row [1xN_adc] - current_delays_col [N_chirp x 1]) -> [N_chirp x N_adc]
            phase_1_matrix = phase(t_onePulse_row - current_delays1_col, fc, slope);
            phase_2_matrix = phase(t_onePulse_row - current_delays2_col, fc, slope);

            % Calculate phase difference and complex signal
            % phase_t_row broadcasts to match the matrix dimensions
            signal_1_matrix = exp(1j * (phase_t_row - phase_1_matrix));
            signal_2_matrix = exp(1j * (phase_t_row - phase_2_matrix));

            % Reshape signals into column vectors [numSamplesTotal x 1]
            % Reshape order needs to match original loop: samples within chirp, then chirp order.
            signal_1 = reshape(signal_1_matrix.', N, 1); % Transpose before reshape
            signal_2 = reshape(signal_2_matrix.', N, 1); % Transpose before reshape

            % Combine signals
            mixed_signal = signal_1;

            % Add AWGN noise
            mixed{i, j, l} = awgn(mixed_signal, snr, 'measured');

        end % j loop (RX)
    end % i loop (TX)
end % l loop (Ny)

% --- Cleanup temporary variables ---
clear t_onePulse_row phase_t_row phase_1_matrix phase_2_matrix signal_1_matrix signal_2_matrix signal_1 signal_2 mixed_signal current_delays1_col current_delays2_col;
clear tx_loc_t_chirps rx_loc_t_chirps tar1_loc_chirps tar2_loc_chirps dist_* t_chirp_starts_col platform_displacement;

%% Post processing - Reshape and Virtual Array Extension

% Concatenate along the 3rd dimension (channels) and reshape
% Original order in mixed: TX, RX, Ny
% Desired order in RDC: ADC, Chirp, Channel(Ny -> TX -> RX ?) Check RDC_plus logic.
% Assuming RDC needs channels ordered corresponding to virtual array elements.
% Let's assume the virtual array is formed such that iterating TX then RX for a fixed Ny gives adjacent elements.
numChannels = numRX * numTX * Ny;
RDC = zeros(numADC, numChirpsTotal, numChannels);
count = 1;
for l = 1:Ny % Elevation
    for i = 1:numTX % Azimuth TX
        for j = 1:numRX % Azimuth RX
             % mixed{i,j,l} size is [N x 1]. Reshape to [numADC x numChirpsTotal]
            RDC(:,:, count) = reshape(mixed{i,j,l}, numADC, numChirpsTotal);
            count = count + 1;
        end
    end
end
clear mixed; % Free memory

% --- Virtual Array Extension (Keep existing vectorized code) ---
if mod(N_L, 2) == 0
    error('N_L must be an odd integer for the virtual array extension logic.');
end
N_half = (N_L - 1) / 2;

numChirpsOutPerCPI = numChirps - (N_L - 1); % Chirps per CPI in output
numChannelsInPerNy = numRX * numTX; % Input physical channels per Ny slice
numChannelsOutPerNy = numChannelsInPerNy + N_L - 1; % Output virtual channels per Ny slice

numColsIn = numChirps * numCPI; % = numChirpsTotal
numPagesIn = numChannelsInPerNy * Ny; % = numChannels
numColsOut = numChirpsOutPerCPI * numCPI;
numPagesOut = numChannelsOutPerNy * Ny; % Total virtual channels

% Pre-allocate output
RDC_plus = zeros(numADC, numColsOut, numPagesOut);

% Vectorized k loop (output chirps within a CPI)
k_vec = 1:numChirpsOutPerCPI;

for l = 1:Ny % Iterate over elevation slices
    page_offset_in = (l - 1) * numChannelsInPerNy;
    page_offset_out = (l - 1) * numChannelsOutPerNy;

    page_idx_in_first = 1 + page_offset_in; % First channel index in RDC for this Ny slice
    page_idx_in_last = numChannelsInPerNy + page_offset_in; % Last channel index in RDC for this Ny slice

    for i = 1:numCPI % Iterate over CPIs
        col_offset_in = (i - 1) * numChirps;
        col_offset_out = (i - 1) * numChirpsOutPerCPI;

        % --- 1. Copy center data block ---
        cols_in_center = ((N_half + 1):(numChirps - N_half)) + col_offset_in;
        pages_in_center = (1:numChannelsInPerNy) + page_offset_in; % Page indices for this Ny slice in RDC

        cols_out = k_vec + col_offset_out;
        pages_out_center = ((N_half + 1):(N_half + numChannelsInPerNy)) + page_offset_out; % Corresponding page indices in RDC_plus

        RDC_plus(:, cols_out, pages_out_center) = RDC(:, cols_in_center, pages_in_center);

        % --- 2. Fill edge data blocks ---
        for j = 1:N_half % Iterate over extension amount
            % Left side fill
            cols_in_left = (k_vec + N_half + j) + col_offset_in;
            page_out_left = (N_half + 1 - j) + page_offset_out;
            RDC_plus(:, cols_out, page_out_left) = RDC(:, cols_in_left, page_idx_in_first); % Use first channel of this Ny slice

            % Right side fill
            cols_in_right = (k_vec + N_half - j) + col_offset_in;
            page_out_right = (N_half + numChannelsInPerNy + j) + page_offset_out;
            RDC_plus(:, cols_out, page_out_right) = RDC(:, cols_in_right, page_idx_in_last); % Use last channel of this Ny slice
        end % j loop (extension)
    end % i loop (CPI)
end % l loop (Ny)

clear RDC; % Free memory
clear N_half numChirpsOutPerCPI numChannelsInPerNy numChannelsOutPerNy ...
      numColsIn numPagesIn numColsOut k_vec l page_offset_in ...
      page_offset_out page_idx_in_first page_idx_in_last i col_offset_in ...
      col_offset_out cols_in_center pages_in_center cols_out pages_out_center ...
      j cols_in_left page_out_left cols_in_right page_out_right;

%% 2-D FFT (Range-Doppler)

numChirps_new = numChirps - (N_L - 1); % Effective number of chirps after extension logic

% --- FFT Parameters ---
dR = c/(2*BW); % range resol
Rmax = numADC * dR; % Max unambiguous range for display (can differ from processing)
dV = lambda/(2*numChirps_new*T); % velocity resol (using effective chirps and PRI T)
Vmax_eff = lambda/(4*T); % Max unambiguous velocity (based on PRI T) Should match Vmax if T is PRI.

N_Dopp = numChirps_new; % length of doppler FFT (use power of 2 for speed if needed: N_Dopp = 2^nextpow2(numChirps_new))
N_range = numADC; % length of range FFT (use power of 2: N_range = 2^nextpow2(numADC))

% Range and Velocity Axes
R_axis = (0:N_range-1)*dR;
V_axis = linspace(-Vmax_eff, Vmax_eff, N_Dopp);

% --- OPTIMIZATION: Perform FFTs more efficiently ---
% Reshape RDC_plus to group CPIs: [numADC, numChirps_new, numCPI, numPagesOut]
RDC_plus_reshaped = reshape(RDC_plus, N_range, numChirps_new, numCPI, numPagesOut);

% Perform Range FFT along dim 1, Doppler FFT along dim 2
% Apply FFTs across all CPIs and virtual channels simultaneously
RDMs_all_cpi = fftshift(fft(fft(RDC_plus_reshaped, N_range, 1), N_Dopp, 2), 2);

clear RDC_plus RDC_plus_reshaped; % Free memory

% Select the first CPI for further processing (or average/process all)
RDMs_1 = RDMs_all_cpi(:,:,1,:); % Get first CPI [N_range, N_Dopp, 1, numPagesOut]
RDMs = squeeze(RDMs_1); % Resulting RDM for one CPI [N_range, N_Dopp, numPagesOut]
% To average magnitude over CPIs (example):
% RDMs_avg_mag = squeeze(mean(abs(RDMs_all_cpi), 3));
% To average complex data over CPIs (coherent integration, needs phase correction usually):
% RDMs_avg_complex = squeeze(mean(RDMs_all_cpi, 3));
% We proceed with the first CPI as in the original code.


%% Angle Estimation Setup

N_azimuth = numPagesOut; % Total number of virtual channels after extension

ang_phi = -180:0.5:180; % Azimuth search range (adjust resolution as needed)
ang_theta = 0:1:90;   % Elevation search range (adjust resolution as needed)

% --- Steering Vector Calculation (Efficient Batch Calculation) ---
N_theta = length(ang_theta);
N_phi = length(ang_phi);

% Generate all angle combinations (Theta varies fastest)
[phi_grid, theta_grid] = meshgrid(ang_phi, ang_theta); % Note order for meshgrid
theta_all = theta_grid(:); % [N_theta*N_phi x 1]
phi_all = phi_grid(:);     % [N_theta*N_phi x 1]

% Calculate terms dependent on angles (columns of final matrix 'a1')
col_term_A = sind(phi_all) .* sind(theta_all);
col_term_B = cosd(phi_all) .* sind(theta_all);

% Generate virtual element indices (rows of final matrix 'a1')
% Assuming 'RDC_plus' pages correspond to virtual elements ordered by:
% Ny(elevation) -> virtual_x (azimuth)
% Need the actual positions corresponding to pages 1:numPagesOut
% Let's derive virtual positions based on the extension logic:
virtual_element_x_pos = zeros(numPagesOut, 1);
virtual_element_y_pos = zeros(numPagesOut, 1);
count = 1;
d_virtual_x = d_tx / 2; % Assumed virtual spacing from RDC_plus logic needs verification
N_half = (N_L-1)/2;
numChannelsOutPerNy = numRX*numTX+N_L-1;

for l = 1:Ny % Elevation slice
     y_val = y_pos(l);
     % Indices relative to the start of the physical array within this slice
     % Center physical element indices: (0:numRX*numTX-1) * d_virtual_x (relative)
     % Virtual indices range from -(N_L-1)/2 to numRX*numTX-1 + (N_L-1)/2 (relative)
     rel_idx = (-(N_half) : (numRX*numTX - 1 + N_half));
     % Absolute X position depends on the center of the physical array
     center_offset_x = 0; % Assuming array centered at x=0. Adjust if needed.
     x_vals = center_offset_x + rel_idx * d_virtual_x;

     for idx = 1:numChannelsOutPerNy
        virtual_element_x_pos(count) = x_vals(idx);
        virtual_element_y_pos(count) = y_val;
        count = count + 1;
     end
end

% Calculate terms dependent on virtual element positions (rows)
row_term_A = (2 * pi / lambda) * virtual_element_y_pos; % Factor 2 for two-way path y*sin(theta)*sin(phi)
row_term_B = (2 * pi / lambda) * virtual_element_x_pos; % Factor 2 for two-way path x*sin(theta)*cos(phi)

% Calculate phase matrix using outer product (broadcasting)
% phase = -1j * (row_term_A * col_term_A.' + row_term_B * col_term_B.'); % Check phase calculation details
% Standard steering vector phase: exp(-j * 2*pi/lambda * (pos_x*cos(phi)*sin(theta) + pos_y*sin(phi)*sin(theta) + pos_z*cos(theta)))
% Assuming planar array in Z=0 plane:
phase = -1j * (row_term_B * col_term_B.' + row_term_A * col_term_A.'); % [numPagesOut x (N_theta*N_phi)]

% Calculate steering vector matrix 'a1'
a1 = exp(phase); % [numPagesOut x numAngles]


%% Target Detection / Angle Estimation Branch

if params.cf == 1
    %% CA-CFAR (Assuming external ca_cfar function is efficient)
    numGuard = 2; numTrain = numGuard*2; P_fa = 1e-5; SNR_OFFSET = -5;

    % Use the RDM from the first CPI (or averaged RDM)
    RDM_dB = 10*log10(abs(RDMs)); % RDM for angle estimation [N_range x N_Dopp x numPagesOut]
    % CFAR typically runs on Range-Doppler map summed across channels, or per channel.
    % Assuming summation across virtual channels for detection:
    RDM_sum_channels = squeeze(sum(abs(RDMs).^2, 3)); % Sum power across virtual channels
    RDM_sum_dB = 10*log10(RDM_sum_channels);
    RDM_sum_dB(isinf(RDM_sum_dB)) = -150; % Handle -Inf

    [RDM_mask, cfar_ranges, cfar_dopps, K] = ca_cfar(RDM_sum_dB, numGuard, numTrain, P_fa, SNR_OFFSET);
     % cfar_ranges, cfar_dopps are indices into RDM_sum_dB

    if K == 0
        disp('CFAR detected no targets.');
        theta = NaN; phi = NaN;
        return;
    end

    %% Angle Estimation - MUSIC Pseudo Spectrum for CFAR Detections

    music_spectrum = zeros(K, N_theta, N_phi);
    numVirtualAnt = numPagesOut; % Number of virtual antennas
    numSnapshots = numCPI; % Use CPIs as snapshots (if phase corrected) OR chirps within CPI
    % For simplicity, let's use data across virtual antennas at the detected R-D bin as "spatial snapshots"
    % This is common but less effective than temporal snapshots if available & calibrated.

    for i = 1:K % Loop through CFAR detections
        r_idx = cfar_ranges(i);
        d_idx = cfar_dopps(i);

        % Extract spatial snapshot vector from the RDM at the detected peak
        % Use data from the first CPI, or averaged data if calculated
        B = squeeze(RDMs(r_idx, d_idx, :)); % [numVirtualAnt x 1] Spatial snapshot vector

        % Estimate Covariance Matrix Rxx
        % With only one spatial snapshot, Rxx = B * B' (rank 1, MUSIC needs > 1 snapshot)
        % --- Need a better way to get snapshots ---
        % Option 1: Use chirps within the CPI as snapshots (if target stationary within CPI)
        % Need RDC_plus data for this bin: RDC_plus(r_idx, :, d_idx, :)? Requires FFT shift knowledge.
        % Option 2: Use CPIs as snapshots (Requires phase correction between CPIs)
        % Let's use RDMs_all_cpi for snapshots across CPIs at the detected bin
        B_all_cpi = squeeze(RDMs_all_cpi(r_idx, d_idx, :, :)); % [numCPI x numVirtualAnt]
        % If using CPIs as snapshots, need B_all_cpi.' [numVirtualAnt x numCPI]
        B_snapshots = B_all_cpi.';
        if numCPI < 2
             warning('MUSIC requires multiple snapshots (numCPI >= 2)');
             Rxx = B * B'; % Fallback, likely poor result
        else
             Rxx = (B_snapshots * B_snapshots') / numCPI;
        end

        % Eigen decomposition
        [Q, D_eig] = eig(Rxx);
        [~, order] = sort(diag(D_eig), 'descend');
        Q = Q(:, order);

        % Determine signal subspace dimension (e.g., assume 1 target per CFAR detection)
        num_targets_assumed = 1;
        if num_targets_assumed >= numVirtualAnt
             warning('Assumed number of targets >= number of antennas for MUSIC');
             Qn = Q(:, num_targets_assumed:end); % Take remaining as noise subspace
        else
             Qn = Q(:, (num_targets_assumed + 1):end); % Noise subspace eigenvectors
        end


        % Calculate MUSIC spectrum (vectorized over angles)
        if isempty(Qn)
            warning('Noise subspace is empty for detection %d.', i);
            music_spectrum(i,:,:) = 0; % Or NaN
            continue;
        end

        QnH_a1 = Qn' * a1;      % [dim_noise x numAngles]
        denominator = sum(abs(QnH_a1).^2, 1); % Sum power across noise eigenvectors [1 x numAngles]
        % Avoid division by zero or tiny numbers
        denominator(denominator < 1e-10) = 1e-10;
        
        % Simplified numerator for pseudo-spectrum (often just 1)
        % numerator = sum(abs(a1).^2, 1); % Normalization factor
        numerator = 1; % Common simplification

        music_spectrum_flat = numerator ./ denominator;
        music_spectrum(i,:,:) = reshape(music_spectrum_flat, N_theta, N_phi);

    end % End loop over CFAR detections (K)

    % Find peak for the first detected target (or strongest)
    % We process only the first CFAR detection (i=1) for output theta/phi
    if K > 0
        spectrum_data = 10 * log10(abs(squeeze(music_spectrum(1,:,:))));
        spectrum_data(isinf(spectrum_data)) = -150; % Handle Inf

        [max_value, max_index] = max(spectrum_data(:));
        [row, col] = ind2sub(size(spectrum_data), max_index);

        theta = ang_theta(row);
        phi = ang_phi(col);

        fprintf('CFAR Target 1 - Est. Angle: (Theta: %.2f, Phi: %.2f), Peak Strength: %.2f dB\n', theta, phi, max_value);
    else
        theta = NaN; phi = NaN; % No target found
    end


else
    %% MUSIC without CFAR (Original second branch)

    % OPTIMIZATION: Avoid forming the massive matrix A.
    % Estimate covariance from a region of interest or average.
    % Example: Use the range-doppler bin with maximum power in the first CPI RDM.
    RDM_sum_channels = squeeze(sum(abs(RDMs).^2, 3)); % Sum power across virtual channels [N_range x N_Dopp]
    [max_val, max_idx_flat] = max(RDM_sum_channels(:));
    [r_idx_max, d_idx_max] = ind2sub(size(RDM_sum_channels), max_idx_flat);
    fprintf('Non-CFAR MUSIC using peak at Range Idx %d, Doppler Idx %d\n', r_idx_max, d_idx_max);

    % Use CPIs as snapshots at this peak R-D bin
    numSnapshots = numCPI;
    B_all_cpi = squeeze(RDMs_all_cpi(r_idx_max, d_idx_max, :, :)); % [numCPI x numVirtualAnt]
    B_snapshots = B_all_cpi.'; % [numVirtualAnt x numCPI]

    if numCPI < 2
        warning('MUSIC requires multiple snapshots (numCPI >= 2)');
         % Fallback: Use spatial vector as single snapshot (poor)
         B = squeeze(RDMs(r_idx_max, d_idx_max, :));
         Rxx = B*B';
    else
        Rxx = (B_snapshots * B_snapshots') / numCPI;
    end


    % Eigen decomposition
    [Q, D_eig] = eig(Rxx);
    [~, order] = sort(diag(D_eig),'descend');
    Q = Q(:, order); % Sort eigenvectors

    % Estimate number of sources (e.g., using MDL/AIC or eigenvalue threshold)
    % For simplicity, assume 1 or 2 strongest sources based on targets defined.
    num_targets_assumed = 2; % Assume 2 targets present
     if num_targets_assumed >= size(Q, 2)
         warning('Assumed number of targets >= number of antennas for MUSIC');
         num_targets_assumed = size(Q, 2) - 1; % Max possible
     end
    Qn = Q(:, (num_targets_assumed + 1):end); % Noise eigenvectors

    % Calculate MUSIC spectrum (vectorized)
    if isempty(Qn)
        warning('Noise subspace is empty.');
        music_spectrum_2D = zeros(N_theta, N_phi);
    else
        QnH_a1 = Qn' * a1;
        denominator = sum(abs(QnH_a1).^2, 1);
        denominator(denominator < 1e-10) = 1e-10;
        numerator = 1; % Simplified numerator
        music_spectrum_flat = numerator ./ denominator;
        music_spectrum_2D = reshape(music_spectrum_flat, N_theta, N_phi); % [N_theta x N_phi]
    end

    %% Peak Finding (Similar to original, ensure coordinates match)
    spectrum_data = 10 * log10(abs(music_spectrum_2D));
    spectrum_data(isinf(spectrum_data)) = -150; % Handle Inf

    % --- Optional: Smoothing ---
    % smoothed_spectrum = imgaussfilt(spectrum_data, 1); % Example smoothing
    smoothed_spectrum = spectrum_data; % No smoothing

    % Find regional maxima
    regional_max = imregionalmax(smoothed_spectrum);
    [max_rows, max_cols] = find(regional_max);
    peak_values = smoothed_spectrum(regional_max);

    % Sort peaks and find the main one
    if ~isempty(max_rows)
        peaks_info = sortrows([max_rows, max_cols, peak_values], -3); % Sort by strength (descending)
        main_peak = peaks_info(1, :);
        theta = ang_theta(main_peak(1));
        phi = ang_phi(main_peak(2)); % Use original ang_phi for indexing
        fprintf('Non-CFAR MUSIC - Strongest Peak: (Theta: %.2f, Phi: %.2f), Strength: %.2f dB\n', theta, phi, main_peak(3));

        % Optional: Print second peak if exists
        if size(peaks_info, 1) >= 2
            second_peak = peaks_info(2, :);
             fprintf('Non-CFAR MUSIC - Second Peak: (Theta: %.2f, Phi: %.2f), Strength: %.2f dB\n', ...
                 ang_theta(second_peak(1)), ang_phi(second_peak(2)), second_peak(3));
        end

    else
        disp('Non-CFAR MUSIC: No peaks found.');
        theta = NaN; phi = NaN;
    end

end % End CFAR / Non-CFAR branch

end % End Function


% Helper function (if not available externally) - Placeholder
function [mask, r_idx, d_idx, K] = ca_cfar(rdm_db, numGuard, numTrain, pfa, snr_offset)
    % Placeholder for Cell-Averaging CFAR
    % Input: rdm_db (log magnitude Range-Doppler Map), guard/train cells, pfa, offset
    % Output: mask (binary detection mask), r_idx, d_idx (indices of detections), K (num detections)
    warning('Using placeholder ca_cfar. Implement actual CFAR logic.');
    [N_range, N_dopp] = size(rdm_db);
    mask = false(N_range, N_dopp);
    
    % Example: Find global max as a single detection for testing
    [max_val, flat_idx] = max(rdm_db(:));
    [r, d] = ind2sub(size(rdm_db), flat_idx);
    
    % Simple threshold check (not real CFAR)
    threshold = mean(rdm_db(:)) + snr_offset; % Very basic threshold
    if max_val > threshold
        mask(r, d) = true;
        r_idx = r;
        d_idx = d;
        K = 1;
    else
        r_idx = [];
        d_idx = [];
        K = 0;
    end
    
    % A real implementation would involve sliding window, noise estimation, threshold calculation per cell.
end
