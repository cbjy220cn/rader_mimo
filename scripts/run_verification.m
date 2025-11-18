%% Main script to run the verification and performance evaluation framework.
% This script sets up the common simulation scene and then calls
% dedicated functions to perform specific tests.

clear; clc; close all;
fprintf('Starting Verification & Evaluation Framework...\n\n');

% --- Setup the Data Logger ---
logger = data();

%% Step 1: Common Scene Setup
fprintf('--- Step 1: Setting up Common Simulation Scene ---\n');

% Radar Parameters
radar_params.fc = 3e9; % 3 GHz carrier frequency
lambda = physconst('LightSpeed') / radar_params.fc;

% Array Setup (L-shaped to resolve DOA ambiguity)
num_elements = 8;
spacing = lambda / 2;
tx_indices = 1;
rx_indices = 1:num_elements;
physical_elements = [((0:num_elements-1) - (num_elements-1)/2).' * spacing, ...
                     zeros(num_elements, 1), zeros(num_elements, 1)];
physical_elements(end, 1) = 0;
physical_elements(end, 2) = spacing;
array_platform = ArrayPlatform(physical_elements, tx_indices, rx_indices);

% Dynamic Trajectory Setup
platform_vel = [150, 0, 0]; % 150 m/s
trajectory_func = @(t) struct('position', platform_vel * t, 'orientation', [0,0,0]);
array_platform = array_platform.set_trajectory(trajectory_func);

% DOA Search Grid
search_grid.theta = 0:0.5:90;
search_grid.phi = -90:0.5:90;

fprintf('Scene setup complete.\n\n');

%% Step 2: Run Performance Evaluations

% --- Test 1: Dual-Target Resolution Test ---
% Create two targets with a small angular separation.
theta1_true = 30;
phi1_true = 20;
target1_pos = [1000 * sind(theta1_true) * cosd(phi1_true), ...
               1000 * sind(theta1_true) * sind(phi1_true), ...
               1000 * cosd(theta1_true)];
target1 = Target(target1_pos, [0,0,0], 1);

theta2_true = 32; % 2 degree separation in elevation
phi2_true = 20;
target2_pos = [1000 * sind(theta2_true) * cosd(phi2_true), ...
               1000 * sind(theta2_true) * sind(phi2_true), ...
               1000 * cosd(theta2_true)];
target2 = Target(target2_pos, [0,0,0], 1);
targets_for_res_test = {target1, target2};

% Call the dedicated test function
fig_res = run_resolution_test(radar_params, array_platform, targets_for_res_test, search_grid);
logger.save_figure(fig_res, 'resolution_test');


% --- Test 2: RMSE vs. SNR Test ---
fprintf('\n--- Running RMSE vs. SNR Test ---\n');

% We can reuse target1 from the resolution test as the single target for this test.
target_for_rmse_test = target1;

% Call the dedicated test function
% NOTE: This test can be time-consuming due to the nested loops.
[fig_rmse, data_rmse] = run_rmse_vs_snr_test(radar_params, array_platform, target_for_rmse_test, search_grid);
logger.save_figure(fig_rmse, 'rmse_vs_snr');
logger.save_data('rmse_vs_snr_results', data_rmse);
