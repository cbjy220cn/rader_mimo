function fig_handle = run_rotation_vs_circular_test(radar_params, num_elements, targets, search_grid)
%RUN_ROTATION_VS_CIRCULAR_TEST Compares resolution of different array types.
%
%   Compares a static circular array vs. a rotating linear array vs. a
%   static linear array for resolving two closely spaced targets.
%
%   Inputs:
%       radar_params   - Struct with radar parameters.
%       num_elements   - The number of elements for all arrays.
%       targets        - A cell array of two Target objects.
%       search_grid    - Struct defining the theta and phi search space.
%
%   Outputs:
%       fig_handle     - A handle to the figure with the comparison plots.

fprintf('--- Running Rotation vs. Circular Array Resolution Test ---\n\n');

lambda = physconst('LightSpeed') / radar_params.fc;
spacing = lambda / 2;

% --- Simulation Time & Rotation Setup ---
num_snapshots = 128;
T_chirp = 1e-4;
t_axis = (0:num_snapshots-1) * T_chirp;
total_time = t_axis(end);
total_rotation_angle_deg = 10; % Let's rotate 10 degrees total
omega_dps = total_rotation_angle_deg / total_time; % deg/sec

% --- 1. Static Circular Array Definition ---
fprintf('1. Evaluating Static Circular Array...\n');
circumference = num_elements * spacing;
radius = circumference / (2 * pi);
angles = (0:num_elements-1).' * (2*pi/num_elements);
circ_elements = [radius * cos(angles), radius * sin(angles), zeros(num_elements, 1)];
array_circ_static = ArrayPlatform(circ_elements, 1, 1:num_elements);
array_circ_static = array_circ_static.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

sig_gen_circ = SignalGenerator(radar_params, array_circ_static, targets);
snapshots_circ = sig_gen_circ.generate_snapshots(t_axis, inf);
estimator_circ = DoaEstimator(array_circ_static, radar_params);
spectrum_circ = estimator_circ.estimate_gmusic(snapshots_circ, t_axis, 2, search_grid);

% --- 2. Rotating Linear Array Definition ---
fprintf('2. Evaluating Rotating Linear Array...\n');
% Define ULA with one end at the origin to make it pivot there
lin_elements = [(0:num_elements-1).' * spacing, zeros(num_elements, 2)];
array_lin_rotating = ArrayPlatform(lin_elements, 1, 1:num_elements);
rotation_trajectory = @(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]);
array_lin_rotating = array_lin_rotating.set_trajectory(rotation_trajectory);

sig_gen_rot = SignalGenerator(radar_params, array_lin_rotating, targets);
snapshots_rot = sig_gen_rot.generate_snapshots(t_axis, inf);
estimator_rot = DoaEstimator(array_lin_rotating, radar_params);
spectrum_rot = estimator_rot.estimate_gmusic(snapshots_rot, t_axis, 2, search_grid);

% --- START DEBUG BLOCK ---
fprintf('\n--- [DEBUG] Starting Phase Verification for Rotating Array ---\n');
% Use a single target for simplicity
target_debug = targets{1}; 
sig_gen_debug = SignalGenerator(radar_params, array_lin_rotating, {target_debug});
snapshots_debug = sig_gen_debug.generate_snapshots(t_axis, inf);

% Stage 1: Verify the phase of the generated snapshots
t0_snapshot = snapshots_debug(:, 1);
t_end_snapshot = snapshots_debug(:, end);
actual_phase_diff = angle(t_end_snapshot) - angle(t0_snapshot);
actual_phase_diff = unwrap(actual_phase_diff); % Handle phase wraps

fprintf('[DEBUG] Stage 1: Actual Phase Difference from Signal Generator (rad):\n');
disp(actual_phase_diff.');

% Calculate theoretical phase difference based on geometry change
pos_t0 = array_lin_rotating.get_mimo_virtual_positions(t_axis(1));
pos_t_end = array_lin_rotating.get_mimo_virtual_positions(t_axis(end));
target_pos_debug = target_debug.get_position_at(0);
u_true = target_pos_debug(:) / norm(target_pos_debug);
k = 2 * pi * radar_params.fc / radar_params.c;

% The phase of the beat signal is exp(j*k*R_total). So we calculate the
% change in the total path length for each virtual element.
phase_t0 = k * (pos_t0 * u_true);
phase_t_end = k * (pos_t_end * u_true);
theoretical_phase_diff = phase_t_end - phase_t0;
theoretical_phase_diff = unwrap(theoretical_phase_diff);

fprintf('[DEBUG] Stage 1: Theoretical Phase Difference from Geometry (rad):\n');
disp(theoretical_phase_diff.');

phase_error_stage1 = theoretical_phase_diff - actual_phase_diff;
fprintf('[DEBUG] Stage 1: Error (Theoretical - Actual). Should be near zero.\n');
disp(phase_error_stage1.');

% Stage 2: Verify the DoaEstimator's steering vector matches the signal
[~, A_u_debug] = estimator_rot.estimate_gmusic(snapshots_debug, t_axis, 1, search_grid, u_true);
estimator_phase_t0 = angle(A_u_debug(:, 1));
estimator_phase_t_end = angle(A_u_debug(:, end));
estimator_phase_diff = unwrap(estimator_phase_t_end) - unwrap(estimator_phase_t0);

fprintf('\n[DEBUG] Stage 2: Phase Difference from DoaEstimator''s Model (rad):\n');
disp(estimator_phase_diff.');

phase_error_stage2 = actual_phase_diff - estimator_phase_diff;
fprintf('[DEBUG] Stage 2: Error (Actual Signal - Estimator Model). Should be near zero.\n');
disp(phase_error_stage2.');

fprintf('--- [DEBUG] End of Phase Verification ---\n\n');
% --- END DEBUG BLOCK ---


% --- 3. Static Linear Array Definition (for baseline comparison) ---
fprintf('3. Evaluating Static Linear Array...\n');
array_lin_static = ArrayPlatform(lin_elements, 1, 1:num_elements); % Uses same physical elements
array_lin_static = array_lin_static.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

sig_gen_stat = SignalGenerator(radar_params, array_lin_static, targets);
snapshots_stat = sig_gen_stat.generate_snapshots(t_axis, inf);
estimator_stat = DoaEstimator(array_lin_static, radar_params);
spectrum_stat = estimator_stat.estimate_gmusic(snapshots_stat, t_axis, 2, search_grid);

% --- Plotting Results ---
fig_handle = figure('Name', 'Array Type Resolution Comparison', 'Position', [100, 100, 1500, 450]);

% Get true DOAs for plotting
target1_pos = targets{1}.get_position_at(0);
[phi1_true_rad, theta1_true_rad, ~] = cart2sph(target1_pos(1), target1_pos(2), target1_pos(3));
theta1_true = 90 - rad2deg(theta1_true_rad);
phi1_true = rad2deg(phi1_true_rad);
target2_pos = targets{2}.get_position_at(0);
[phi2_true_rad, theta2_true_rad, ~] = cart2sph(target2_pos(1), target2_pos(2), target2_pos(3));
theta2_true = 90 - rad2deg(theta2_true_rad);
phi2_true = rad2deg(phi2_true_rad);

% Plot for Static Linear Array
subplot(1, 3, 1);
surf(search_grid.phi, search_grid.theta, spectrum_stat);
shading interp; view(2); colorbar; hold on;
plot(phi1_true, theta1_true, 'r+', 'MarkerSize', 12, 'LineWidth', 2);
plot(phi2_true, theta2_true, 'rx', 'MarkerSize', 12, 'LineWidth', 2);
title('Static Linear Array');
xlabel('Azimuth (phi) [deg]'); ylabel('Elevation (theta) [deg]');

% Plot for Static Circular Array
subplot(1, 3, 2);
surf(search_grid.phi, search_grid.theta, spectrum_circ);
shading interp; view(2); colorbar; hold on;
plot(phi1_true, theta1_true, 'r+', 'MarkerSize', 12, 'LineWidth', 2);
plot(phi2_true, theta2_true, 'rx', 'MarkerSize', 12, 'LineWidth', 2);
title('Static Circular Array');
xlabel('Azimuth (phi) [deg]');

% Plot for Rotating Linear Array
subplot(1, 3, 3);
surf(search_grid.phi, search_grid.theta, spectrum_rot);
shading interp; view(2); colorbar; hold on;
plot(phi1_true, theta1_true, 'r+', 'MarkerSize', 12, 'LineWidth', 2);
plot(phi2_true, theta2_true, 'rx', 'MarkerSize', 12, 'LineWidth', 2);
title('Rotating Linear Array');
xlabel('Azimuth (phi) [deg]');

sgtitle(sprintf('Resolution Comparison (%d Elements)', num_elements));

end
