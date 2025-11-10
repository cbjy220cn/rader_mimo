function [fig_handle, results] = run_rmse_vs_snr_test(radar_params, array_platform_dynamic, target_static, search_grid)
%RUN_RMSE_VS_SNR_TEST performs a Monte Carlo simulation to evaluate DOA
%estimation accuracy vs. Signal-to-Noise Ratio (SNR).
%
%   fig_handle = RUN_RMSE_VS_SNR_TEST(radar_params, array_platform_dynamic, target_static, search_grid)
%
%   Inputs:
%       radar_params           - Struct with radar parameters.
%       array_platform_dynamic - An ArrayPlatform object with a dynamic trajectory.
%       target_static          - A single static Target object.
%       search_grid            - Struct defining the theta and phi search space.
%
%   Outputs:
%       fig_handle             - A handle to the figure with the RMSE vs. SNR plot.

fprintf('--- Running RMSE vs. SNR Performance Test ---\n\n');

% --- Simulation Parameters ---
snr_db_range = -5:3:16; % SNR range from -5 dB to 16 dB
num_runs = 100; % Monte Carlo runs per SNR point
num_snapshots = 128;
T_chirp = 1e-4;
t_axis = (0:num_snapshots-1) * T_chirp;

% Get true DOA
target_pos = target_static.get_position_at(0);
[phi_true_rad, theta_true_rad, ~] = cart2sph(target_pos(1), target_pos(2), target_pos(3));
theta_true = 90 - rad2deg(theta_true_rad);
phi_true = rad2deg(phi_true_rad);

fprintf('Using single target at Theta=%.2f, Phi=%.2f\n', theta_true, phi_true);
fprintf('SNR Range: %s dB\n', mat2str(snr_db_range));
fprintf('Monte Carlo runs per SNR: %d\n\n', num_runs);

% --- Array Configurations ---
array_static = array_platform_dynamic.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

% --- Initialize Results Storage ---
rmse_theta_dynamic = zeros(size(snr_db_range));
rmse_phi_dynamic = zeros(size(snr_db_range));
rmse_theta_static = zeros(size(snr_db_range));
rmse_phi_static = zeros(size(snr_db_range));

% --- Signal Generators ---
sig_gen_dynamic = SignalGenerator(radar_params, array_platform_dynamic, {target_static});
sig_gen_static = SignalGenerator(radar_params, array_static, {target_static});

% --- Estimators ---
estimator_dynamic = DoaEstimator(array_platform_dynamic, radar_params);
estimator_static = DoaEstimator(array_static, radar_params);

% --- Main Simulation Loop ---
tic;
for i = 1:length(snr_db_range)
    snr_db = snr_db_range(i);
    fprintf('Running simulations for SNR = %d dB...\n', snr_db);
    
    theta_estimates_dynamic = zeros(1, num_runs);
    phi_estimates_dynamic = zeros(1, num_runs);
    theta_estimates_static = zeros(1, num_runs);
    phi_estimates_static = zeros(1, num_runs);

    parfor j = 1:num_runs
        % Dynamic Case
        snapshots_noisy_dyn = sig_gen_dynamic.generate_snapshots(t_axis, snr_db);
        spectrum_noisy_dyn = estimator_dynamic.estimate_gmusic(snapshots_noisy_dyn, t_axis, 1, search_grid);
        [theta_est_dyn, phi_est_dyn, ~] = DoaEstimator.find_peaks(spectrum_noisy_dyn, search_grid, 1);
        theta_estimates_dynamic(j) = theta_est_dyn;
        phi_estimates_dynamic(j) = phi_est_dyn;
        
        % Static Case
        snapshots_noisy_stat = sig_gen_static.generate_snapshots(t_axis, snr_db);
        spectrum_noisy_stat = estimator_static.estimate_gmusic(snapshots_noisy_stat, t_axis, 1, search_grid);
        [theta_est_stat, phi_est_stat, ~] = DoaEstimator.find_peaks(spectrum_noisy_stat, search_grid, 1);
        theta_estimates_static(j) = theta_est_stat;
        phi_estimates_static(j) = phi_est_stat;
    end
    
    % Calculate RMSE for this SNR point
    rmse_theta_dynamic(i) = sqrt(mean((theta_estimates_dynamic - theta_true).^2));
    rmse_phi_dynamic(i) = sqrt(mean((phi_estimates_dynamic - phi_true).^2));
    rmse_theta_static(i) = sqrt(mean((theta_estimates_static - theta_true).^2));
    rmse_phi_static(i) = sqrt(mean((phi_estimates_static - phi_true).^2));
end
toc;

fprintf('\nSimulation finished.\n');

% --- Package results for saving ---
results.snr_db_range = snr_db_range;
results.rmse_theta_dynamic = rmse_theta_dynamic;
results.rmse_phi_dynamic = rmse_phi_dynamic;
results.rmse_theta_static = rmse_theta_static;
results.rmse_phi_static = rmse_phi_static;

% --- Plotting Results ---
fig_handle = figure('Name', 'RMSE vs. SNR Performance');

subplot(1, 2, 1);
semilogy(snr_db_range, rmse_theta_dynamic, 'o-', 'LineWidth', 2, 'DisplayName', 'Dynamic Array');
hold on;
semilogy(snr_db_range, rmse_theta_static, 's--', 'LineWidth', 2, 'DisplayName', 'Static Array');
grid on;
xlabel('SNR (dB)');
ylabel('RMSE Theta (degrees)');
title('RMSE of Elevation Angle (\theta)');
legend;
ylim([1e-2, 1e2]);

subplot(1, 2, 2);
semilogy(snr_db_range, rmse_phi_dynamic, 'o-', 'LineWidth', 2, 'DisplayName', 'Dynamic Array');
hold on;
semilogy(snr_db_range, rmse_phi_static, 's--', 'LineWidth', 2, 'DisplayName', 'Static Array');
grid on;
xlabel('SNR (dB)');
ylabel('RMSE Phi (degrees)');
title('RMSE of Azimuth Angle (\phi)');
legend;
ylim([1e-2, 1e2]);

sgtitle('DOA Estimation Performance Comparison');

end
