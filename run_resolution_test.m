function fig_handle = run_resolution_test(radar_params, array_platform, targets, search_grid)
%RUN_RESOLUTION_TEST Performs a dual-target resolution test for a given radar setup.
%
%   fig_handle = RUN_RESOLUTION_TEST(radar_params, array_platform, targets, search_grid)
%
%   Inputs:
%       radar_params   - Struct with radar parameters (e.g., fc for frequency).
%       array_platform - An ArrayPlatform object, pre-configured.
%       targets        - A cell array of two Target objects.
%       search_grid    - Struct defining the theta and phi search space.
%
%   Outputs:
%       fig_handle     - A handle to the figure containing the comparison plots.

fprintf('--- Running Dual-Target Resolution Test ---\n\n');

if numel(targets) ~= 2
    error('This test is designed for exactly two targets.');
end

% Extract true DOAs for plotting
target1_pos = targets{1}.get_position_at(0);
[phi1_true_rad, theta1_true_rad, ~] = cart2sph(target1_pos(1), target1_pos(2), target1_pos(3));
theta1_true = 90 - rad2deg(theta1_true_rad);
phi1_true = rad2deg(phi1_true_rad);

target2_pos = targets{2}.get_position_at(0);
[phi2_true_rad, theta2_true_rad, ~] = cart2sph(target2_pos(1), target2_pos(2), target2_pos(3));
theta2_true = 90 - rad2deg(theta2_true_rad);
phi2_true = rad2deg(phi2_true_rad);

fprintf('Target 1 True DOA: Theta=%.2f, Phi=%.2f\n', theta1_true, phi1_true);
fprintf('Target 2 True DOA: Theta=%.2f, Phi=%.2f\n\n', theta2_true, phi2_true);

% Simulation time axis
num_snapshots = 128;
T_chirp = 1e-4;
t_axis = (0:num_snapshots-1) * T_chirp;

% --- Test A: Static Array ---
fprintf('Running Test A: Static Array...\n');
array_static = array_platform.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));
sig_gen_static = SignalGenerator(radar_params, array_static, targets);
snapshots_static_ideal = sig_gen_static.generate_snapshots(t_axis, inf);
estimator_static = DoaEstimator(array_static, radar_params);
[spectrum_static, ~] = estimator_static.estimate_gmusic(snapshots_static_ideal, t_axis, 2, search_grid);
fprintf('Static array test finished.\n');

% --- Test B: Dynamic Array ---
fprintf('Running Test B: Dynamic Array...\n');
% Assuming the platform's trajectory is already set from the outside
sig_gen_dynamic = SignalGenerator(radar_params, array_platform, targets);
snapshots_dynamic_ideal = sig_gen_dynamic.generate_snapshots(t_axis, inf);
estimator_dynamic = DoaEstimator(array_platform, radar_params);
[spectrum_dynamic, ~] = estimator_dynamic.estimate_gmusic(snapshots_dynamic_ideal, t_axis, 2, search_grid);
fprintf('Dynamic array test finished.\n\n');


% --- Plotting Results ---
fig_handle = figure('Name', 'Resolution Test Results', 'Position', [100, 100, 1200, 500]);

% Plot for Static Array
subplot(1, 2, 1);
surf(search_grid.phi, search_grid.theta, spectrum_static);
shading interp;
view(2);
xlabel('Azimuth (phi) [deg]');
ylabel('Elevation (theta) [deg]');
title('Static Array - Resolution Test');
colorbar;
hold on;
plot(phi1_true, theta1_true, 'r+', 'MarkerSize', 12, 'LineWidth', 2);
plot(phi2_true, theta2_true, 'rx', 'MarkerSize', 12, 'LineWidth', 2);
legend('Spectrum', 'True DOA 1', 'True DOA 2');
axis([search_grid.phi(1) search_grid.phi(end) search_grid.theta(1) search_grid.theta(end)]);


% Plot for Dynamic Array
subplot(1, 2, 2);
surf(search_grid.phi, search_grid.theta, spectrum_dynamic);
shading interp;
view(2);
xlabel('Azimuth (phi) [deg]');
ylabel('Elevation (theta) [deg]');
title('Dynamic Array - Resolution Test');
colorbar;
hold on;
plot(phi1_true, theta1_true, 'r+', 'MarkerSize', 12, 'LineWidth', 2);
plot(phi2_true, theta2_true, 'rx', 'MarkerSize', 12, 'LineWidth', 2);
legend('Spectrum', 'True DOA 1', 'True DOA 2');
axis([search_grid.phi(1) search_grid.phi(end) search_grid.theta(1) search_grid.theta(end)]);

sgtitle('Dual-Target Resolution Comparison');

end
