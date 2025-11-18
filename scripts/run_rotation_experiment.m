%% Main script for the Rotation vs. Circular Array Resolution Experiment.
% This script sets up a specific dual-target scenario and calls the
% dedicated test function to compare the resolution capabilities of
% a static circular array vs. a rotating linear array.

clear; clc; close all;
fprintf('Starting Rotation vs. Circular Array Experiment...\n\n');

% --- Setup the Data Logger ---
logger = data();

%% Step 1: Experiment-Specific Scene Setup
fprintf('--- Step 1: Setting up Experiment Scene ---\n');

% Radar Parameters
radar_params.fc = 77e9; % Carrier frequency (77 GHz automotive radar)
radar_params.c = physconst('LightSpeed');
radar_params.lambda = radar_params.c / radar_params.fc;

% FMCW Waveform Parameters (based on typical automotive radar)
radar_params.fs = 20e6; % Sampling rate (20 MHz)
radar_params.T_chirp = 50e-6; % Chirp time (50 us)
radar_params.slope = 100e12; % Chirp slope (100 MHz/us)
radar_params.BW = radar_params.slope * radar_params.T_chirp; % Bandwidth
radar_params.num_samples = radar_params.T_chirp * radar_params.fs; % Samples per chirp
radar_params.range_res = radar_params.c / (2 * radar_params.BW);

% Common DOA Search Grid
search_grid.theta = 0:0.5:90;
search_grid.phi = -90:0.5:90;

% Dual-Target Setup for the test
% --- DEFINITIVE FIX: Moved targets closer to be within the radar's max range ---
target_range = 20; % meters
theta1_true = 30;
phi1_true = 20;
target1_pos = [target_range * sind(theta1_true) * cosd(phi1_true), ...
               target_range * sind(theta1_true) * sind(phi1_true), ...
               target_range * cosd(theta1_true)];
target1 = Target(target1_pos, [5, 0, 0], 1); % Add small velocity for decorrelation

% theta2_true = 32; % 2 degree separation
% phi2_true = 20;
% target2_pos = [target_range * sind(theta2_true) * cosd(phi2_true), ...
%                target_range * sind(theta2_true) * sind(phi2_true), ...
%                target_range * cosd(theta2_true)];
% target2 = Target(target2_pos, [-5, 0, 0], 1); % Add small opposing velocity
targets = {target1};

% Number of array elements for a fair comparison
num_elements = 8;

fprintf('Scene setup complete.\n\n');

%% Step 2: Run the Test
fprintf('--- Running Rotation vs. Circular Array Test ---\n');

% Call the dedicated test function
fig_handle = run_rotation_vs_circular_test(radar_params, num_elements, targets, search_grid);

% Save the resulting figure
logger.save_figure(fig_handle, 'rotation_vs_circular_comparison');

fprintf('\nExperiment finished. Results saved to output directory.\n');
