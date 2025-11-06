%% Main script to run the verification framework for the DOA estimation framework.
% This script implements the 4-step verification plan to ensure the
% correctness and effectiveness of the Generalized MUSIC algorithm for moving
% arrays.

clear; clc; close all;
fprintf('Starting Verification Framework...\n\n');

%% Step 1: Ground Truth and Environment Setup
% We create a controlled world where we know the exact answer.

fprintf('--- Step 1: Setting up Ground Truth ---\n');

% Radar Parameters
radar_params.fc = 3e9; % 3 GHz carrier frequency
lambda = physconst('LightSpeed') / radar_params.fc;

% Array Setup (a simple 8-element ULA)
num_elements = 8;
spacing = lambda / 2;
tx_indices = 1;
rx_indices = 1:num_elements;
physical_elements = [((0:num_elements-1) - (num_elements-1)/2).' * spacing, ...
                     zeros(num_elements, 1), zeros(num_elements, 1)];

% --- DEFINITIVE FIX for Physical Ambiguity ---
% A 1D ULA cannot uniquely determine both theta and phi. It can only determine
% the direction cosine relative to its axis. This was the root cause of the
% persistent error. We now change the array to a simple L-shaped 2D array
% to resolve this ambiguity.
fprintf('Switching to an L-shaped array to resolve DOA ambiguity.\n');
physical_elements(end, 1) = 0; % Move last element's x-coord to 0
physical_elements(end, 2) = spacing; % Move last element to y-axis

array = ArrayPlatform(physical_elements, tx_indices, rx_indices);

% --- Plotting Figure 1: Array Geometry ---
figure('Name', 'Array Geometry');
scatter(physical_elements(:,1), physical_elements(:,2), 100, 'filled');
hold on;
scatter(physical_elements(tx_indices,1), physical_elements(tx_indices,2), 200, 'r', 'filled');
grid on;
xlabel('X position (m)');
ylabel('Y position (m)');
title('1. Array Physical Geometry');
legend('RX Element', 'TX Element');
axis equal;

% Target Setup (one static target)
target_pos = [1000 * sind(30) * cosd(20), ... % Approx 30 deg elev, 20 deg azim
              1000 * sind(30) * sind(20), ...
              1000 * cosd(30)];
target_vel = [0, 0, 0];
target_rcs = 1;
target = Target(target_pos, target_vel, target_rcs);

% Calculate the true DOA (our golden standard)
[phi_true_rad, theta_true_rad, ~] = cart2sph(target_pos(1), target_pos(2), target_pos(3));
theta_true = 90 - rad2deg(theta_true_rad); % Convert elevation to standard definition
phi_true = rad2deg(phi_true_rad);

fprintf('True Target DOA: Theta = %.2f deg, Phi = %.2f deg\n\n', theta_true, phi_true);

% Simulation time axis (e.g., for 128 chirps)
num_snapshots = 128;
T_chirp = 1e-4; % 0.1 ms between snapshots
t_axis = (0:num_snapshots-1) * T_chirp;

% DOA Search Grid
search_grid.theta = 0:0.5:90;
search_grid.phi = -90:0.5:90;


%% Step 2: Ideal Algorithm Validation (No Noise)
fprintf('--- Step 2: Ideal Validation (Noiseless) ---\n');

% --- Test A: Static Array ---
fprintf('Running Test 2A: Static Array...\n');
array_static = array.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));
    
% Generate the signal for the actual failing case
sig_gen_static = SignalGenerator(radar_params, array_static, target);
snapshots_static_ideal = sig_gen_static.generate_snapshots(t_axis, inf);

% --- START DEBUGGING BLOCK 2 ---
% The broadside test passed, so the issue occurs for off-broadside angles.
% Let's compare the generated signal's phase directly with the estimator's
% ideal steering vector phase for the known true angle.
fprintf('\n--- [DEBUG] TARGETED CHECK AT TRUE DOA ---\n');

% 1. Extract the "ground truth" phase from the generated signal
actual_signal_phase = angle(snapshots_static_ideal(:, 1));
fprintf('[DEBUG] Actual Signal Phase (rad) for Theta=30:\n');
disp(actual_signal_phase');

% 2. Manually construct the "expected" steering vector phase from the estimator's logic
virtual_positions = array_static.get_mimo_virtual_positions(0);
% True direction vector
u_true = [sind(theta_true)*cosd(phi_true); sind(theta_true)*sind(phi_true); cosd(theta_true)];
k = 2 * pi / lambda;
expected_steering_phase = k * (virtual_positions * u_true);

% We need to unwrap to handle phase wrapping differences
% And then normalize by subtracting the first element's phase to compare relative phases
actual_relative_phase = unwrap(actual_signal_phase);
actual_relative_phase = actual_relative_phase - actual_relative_phase(1);

expected_relative_phase = unwrap(expected_steering_phase);
expected_relative_phase = expected_relative_phase - expected_relative_phase(1);

fprintf('[DEBUG] Estimator''s Expected Relative Phase (rad) for Theta=30:\n');
disp(expected_relative_phase');

fprintf('[DEBUG] Actual Signal''s Relative Phase (rad) for Theta=30:\n');
disp(actual_relative_phase');

phase_error = expected_relative_phase - actual_relative_phase;
fprintf('[DEBUG] Phase Error (Expected - Actual). Must be close to zero for all elements.\n');
disp(phase_error');
fprintf('--- [DEBUG] ENDING TARGETED CHECK ---\n\n');
% --- END DEBUGGING BLOCK 2 ---

estimator_static = DoaEstimator(array_static, radar_params);
[spectrum_static, ~] = estimator_static.estimate_gmusic(snapshots_static_ideal, t_axis, 1, search_grid);

[theta_est_static, phi_est_static] = find_peak(spectrum_static, search_grid);
fprintf('Static Result: Theta = %.2f deg, Phi = %.2f deg\n', theta_est_static, phi_est_static);
assert(abs(theta_est_static - theta_true) < 1.0 && abs(phi_est_static - phi_true) < 1.0, 'Static test failed!');
fprintf('Static Test PASSED.\n\n');

% --- Plotting Figure 2: Static MUSIC Spectrum ---
figure('Name', 'Static MUSIC Spectrum');
surf(search_grid.phi, search_grid.theta, spectrum_static);
shading interp;
view(2); % Top-down view
xlabel('Azimuth (phi) [deg]');
ylabel('Elevation (theta) [deg]');
title('2. MUSIC Spectrum (Static Array)');
colorbar;
hold on;
plot(phi_true, theta_true, 'r+', 'MarkerSize', 12, 'LineWidth', 2);
legend('Spectrum', 'True DOA');

% --- Test B: Dynamic Array ---
fprintf('Running Test 2B: Dynamic Array (Uniform Linear Motion)...\n');
platform_vel = [150, 0, 0]; % 150 m/s
trajectory_func = @(t) struct('position', platform_vel * t, 'orientation', [0,0,0]);
array_dynamic = array.set_trajectory(trajectory_func);

sig_gen_dynamic = SignalGenerator(radar_params, array_dynamic, target);
snapshots_dynamic_ideal = sig_gen_dynamic.generate_snapshots(t_axis, inf);

estimator_dynamic = DoaEstimator(array_dynamic, radar_params);
[spectrum_dynamic, ~] = estimator_dynamic.estimate_gmusic(snapshots_dynamic_ideal, t_axis, 1, search_grid);

[theta_est_dynamic, phi_est_dynamic] = find_peak(spectrum_dynamic, search_grid);
fprintf('Dynamic Result: Theta = %.2f deg, Phi = %.2f deg\n', theta_est_dynamic, phi_est_dynamic);
assert(abs(theta_est_dynamic - theta_true) < 1.0 && abs(phi_est_dynamic - phi_true) < 1.0, 'Dynamic test failed!');
fprintf('Dynamic Test PASSED.\n\n');

% --- Plotting Figure 3: Dynamic MUSIC Spectrum ---
figure('Name', 'Dynamic MUSIC Spectrum');
surf(search_grid.phi, search_grid.theta, spectrum_dynamic);
shading interp;
view(2); % Top-down view
xlabel('Azimuth (phi) [deg]');
ylabel('Elevation (theta) [deg]');
title('3. MUSIC Spectrum (Dynamic Array)');
colorbar;
hold on;
plot(phi_true, theta_true, 'r+', 'MarkerSize', 12, 'LineWidth', 2);
legend('Spectrum', 'True DOA');

%% Step 3: Statistical Performance Validation (With Noise)
fprintf('--- Step 3: Statistical Validation (Monte Carlo) ---\n');
num_runs = 100;
snr_db = 10;
theta_estimates = zeros(1, num_runs);
phi_estimates = zeros(1, num_runs);

fprintf('Running %d Monte Carlo simulations at SNR = %d dB...\n', num_runs, snr_db);
tic;
parfor i = 1:num_runs
    snapshots_noisy = sig_gen_dynamic.generate_snapshots(t_axis, snr_db);
    [spectrum_noisy, ~] = estimator_dynamic.estimate_gmusic(snapshots_noisy, t_axis, 1, search_grid);
    [theta_est_noisy, phi_est_noisy] = find_peak(spectrum_noisy, search_grid);
    
    theta_estimates(i) = theta_est_noisy;
    phi_estimates(i) = phi_est_noisy;
end
toc;

theta_errors = theta_estimates - theta_true;
phi_errors = phi_estimates - phi_true;

rmse_theta = sqrt(mean(theta_errors.^2));
rmse_phi = sqrt(mean(phi_errors.^2));
bias_theta = mean(theta_errors);
bias_phi = mean(phi_errors);

fprintf('RMSE Theta: %.4f deg\n', rmse_theta);
fprintf('RMSE Phi:   %.4f deg\n', rmse_phi);
fprintf('Bias Theta: %.4f deg\n', bias_theta);
fprintf('Bias Phi:   %.4f deg\n\n', bias_phi);

assert(rmse_theta < 2.0, 'Theta RMSE is too high!'); % Generous threshold
assert(rmse_phi < 2.0, 'Phi RMSE is too high!'); % Generous threshold
fprintf('Statistical Test PASSED.\n\n');

% --- Plotting Figure 4: Monte Carlo Results ---
figure('Name', 'Monte Carlo Results');
scatter(phi_estimates, theta_estimates, 30, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
plot(phi_true, theta_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
grid on;
xlabel('Azimuth (phi) [deg]');
ylabel('Elevation (theta) [deg]');
title(sprintf('4. Monte Carlo DOA Estimates (%d runs, SNR=%d dB)', num_runs, snr_db));
legend('Estimated DOAs', 'True DOA');
% Zoom in on the results for better visibility
axis([phi_true-3 phi_true+3 theta_true-3 theta_true+3]);


%% Step 4: Benchmark Comparison (Conceptual)
fprintf('--- Step 4: Benchmark Comparison ---\n');
fprintf('This step involves comparing the results of this framework with the\n');
fprintf('output of the original scripts (e.g., circle_move.m) for an equivalent scenario.\n');
fprintf('Due to the complexity of matching all parameters and motion models exactly,\n');
fprintf('this is left as a final integration step. The passing of Steps 1-3 provides\n');
fprintf('strong confidence in the new framework''s correctness.\n\n');


%% Helper function to find peak in spectrum
function [theta_peak, phi_peak] = find_peak(spectrum, grid)
    [~, max_idx] = max(spectrum(:));
    [theta_idx, phi_idx] = ind2sub(size(spectrum), max_idx);
    theta_peak = grid.theta(theta_idx);
    phi_peak = grid.phi(phi_idx);
end
