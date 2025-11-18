%% Baseline Uniform Rectangular Array (URA) Verification Script
% This script serves as a "golden standard" test to verify that the core
% FMCW signal generation and DOA estimation framework is working correctly.
% It uses a large, static Uniform Rectangular Array (URA) that should have
% no trouble resolving the two closely spaced targets.

clear; clc; close all;
fprintf('Starting Baseline URA Verification Test...\n\n');

% --- Setup the Data Logger ---
logger = data();

%% Step 1: Experiment-Specific Scene Setup
fprintf('--- Step 1: Setting up URA Scene ---\n');

% Radar Parameters (using the same FMCW setup)
radar_params.fc = 77e9;
radar_params.c = physconst('LightSpeed');
radar_params.lambda = radar_params.c / radar_params.fc;
radar_params.fs = 20e6;
radar_params.T_chirp = 50e-6;
radar_params.slope = 100e12;
radar_params.BW = radar_params.slope * radar_params.T_chirp;
radar_params.num_samples = radar_params.T_chirp * radar_params.fs;
radar_params.range_res = radar_params.c / (2 * radar_params.BW);

% --- Array Definition: 8x8 Uniform Rectangular Array (URA) ---
Nx = 8; % Number of elements in X
Ny = 8; % Number of elements in Y
num_elements = Nx * Ny;
spacing = radar_params.lambda / 2;

[x, y] = meshgrid( (-(Nx-1)/2:(Nx-1)/2) * spacing, ...
                   (-(Ny-1)/2:(Ny-1)/2) * spacing );
ura_elements = [x(:), y(:), zeros(num_elements, 1)];

% For simplicity, use a single transmitter at the corner and all as receivers
tx_indices = 1;
rx_indices = 1:num_elements;
array_ura_static = ArrayPlatform(ura_elements, tx_indices, rx_indices);
array_ura_static = array_ura_static.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

% Common DOA Search Grid
search_grid.theta = 0:0.5:90;
search_grid.phi = -90:0.5:90;

% Dual-Target Setup (at a close, valid range)
target_range = 20;
theta1_true = 30; phi1_true = 20;
target1_pos = [target_range * sind(theta1_true) * cosd(phi1_true), ...
               target_range * sind(theta1_true) * sind(phi1_true), ...
               target_range * cosd(theta1_true)];
% --- FIX: Introduce small velocity to ensure signals are not perfectly coherent ---
target1 = Target(target1_pos, [5, 0, 0], 1); % 5 m/s velocity for target 1

theta2_true = 32; phi2_true = 20;
target2_pos = [target_range * sind(theta2_true) * cosd(phi2_true), ...
               target_range * sind(theta2_true) * sind(phi2_true), ...
               target_range * cosd(theta2_true)];
target2 = Target(target2_pos, [-5, 0, 0], 1); % -5 m/s velocity for target 2
targets = {target1, target2};

fprintf('Scene setup complete: 8x8 URA with 2 targets.\n\n');

%% Step 2: Run Simulation and Estimation
fprintf('--- Running Simulation ---\n');

num_snapshots = 128;
T_chirp_interval = 200e-6; % Time between chirps
t_axis = (0:num_snapshots-1) * T_chirp_interval;

sig_gen = SignalGenerator(radar_params, array_ura_static, targets);
snapshots = sig_gen.generate_snapshots(t_axis, inf); % Noiseless

estimator = DoaEstimator(array_ura_static, radar_params);
spectrum = estimator.estimate_gmusic(snapshots, t_axis, 2, search_grid);

fprintf('Simulation finished.\n\n');

%% Step 3: Plotting Results
fprintf('--- Plotting Results ---\n');

fig_handle = figure('Name', 'Baseline URA Resolution Test');
surf(search_grid.phi, search_grid.theta, spectrum);
shading interp;
view(2);
xlabel('Azimuth (phi) [deg]');
ylabel('Elevation (theta) [deg]');
title('Resolution of 8x8 Static URA (Baseline)');
colorbar;
hold on;
plot(phi1_true, theta1_true, 'r+', 'MarkerSize', 12, 'LineWidth', 2);
plot(phi2_true, theta2_true, 'rx', 'MarkerSize', 12, 'LineWidth', 2);
legend('Spectrum', 'True DOA 1', 'True DOA 2');

% Save the results
logger.save_figure(fig_handle, 'baseline_ura_test');
fprintf('Results saved to output directory.\n');
