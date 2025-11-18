%% 使用与原始实验完全相同的参数
clear; clc;

fprintf('=== 复现原始实验参数 (180度旋转) ===\n\n');

% 雷达参数（与原始实验完全相同）
radar_params.fc = 77e9;
radar_params.c = physconst('LightSpeed');
radar_params.lambda = radar_params.c / radar_params.fc;
radar_params.fs = 20e6;
radar_params.T_chirp = 50e-6;
radar_params.slope = 100e12;
radar_params.BW = radar_params.slope * radar_params.T_chirp;
radar_params.num_samples = radar_params.T_chirp * radar_params.fs;
radar_params.range_res = radar_params.c / (2 * radar_params.BW);

% 阵列参数（与原始实验完全相同）
num_elements = 8;
spacing = radar_params.lambda / 2;
Nx = 4;
Ny = 2;
x_coords = (-(Nx-1)/2:(Nx-1)/2) * spacing;
y_coords = (-(Ny-1)/2:(Ny-1)/2) * spacing;
[X, Y] = meshgrid(x_coords, y_coords);
rect_elements = [X(:), Y(:), zeros(num_elements, 1)];

% 目标参数（与原始实验完全相同）
target_range = 20;
theta1_true = 30;
phi1_true = 20;
target1_pos = [target_range * sind(theta1_true) * cosd(phi1_true), ...
               target_range * sind(theta1_true) * sind(phi1_true), ...
               target_range * cosd(theta1_true)];
target1 = Target(target1_pos, [5, 0, 0], 1);

fprintf('目标位置: [%.2f, %.2f, %.2f] m\n', target1_pos(1), target1_pos(2), target1_pos(3));
fprintf('真实角度: theta=%.1f°, phi=%.1f°\n', theta1_true, phi1_true);
fprintf('目标速度: [5, 0, 0] m/s\n\n');

% 时间轴参数（与原始实验完全相同）
num_snapshots = 128;
T_chirp = 1e-4;
t_axis = (0:num_snapshots-1) * T_chirp;
total_rotation_angle_deg = 180;
omega_dps = total_rotation_angle_deg / t_axis(end);

fprintf('旋转参数:\n');
fprintf('  快拍数: %d\n', num_snapshots);
fprintf('  快拍间隔: %.2e s\n', T_chirp);
fprintf('  总时间: %.4f s\n', t_axis(end));
fprintf('  总旋转角度: %.1f°\n', total_rotation_angle_deg);
fprintf('  角速度: %.1f °/s\n\n', omega_dps);

% 搜索网格
search_grid.theta = 0:0.5:90;
search_grid.phi = -90:0.5:90;

% ============ 静态阵列 ============
fprintf('--- 静态4x2 URA ---\n');
array_static = ArrayPlatform(rect_elements, 1, 1:num_elements);
array_static = array_static.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

sig_gen_static = SignalGenerator(radar_params, array_static, {target1});
snapshots_static = sig_gen_static.generate_snapshots(t_axis, inf);

estimator_static = DoaEstimator(array_static, radar_params);
spectrum_static = estimator_static.estimate_gmusic(snapshots_static, t_axis, 1, search_grid);
[theta_est_static, phi_est_static, peak_val_static] = DoaEstimator.find_peaks(spectrum_static, search_grid, 1);

fprintf('静态URA估算: theta=%.2f°, phi=%.2f° (峰值=%.2f)\n', theta_est_static, phi_est_static, peak_val_static);
fprintf('误差: Δtheta=%.2f°, Δphi=%.2f°\n\n', theta_est_static-theta1_true, phi_est_static-phi1_true);

% ============ 旋转阵列 (180度) ============
fprintf('--- 旋转4x2 URA (180度旋转) ---\n');
array_rotating = ArrayPlatform(rect_elements, 1, 1:num_elements);
rotation_trajectory = @(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]);
array_rotating = array_rotating.set_trajectory(rotation_trajectory);

sig_gen_rotating = SignalGenerator(radar_params, array_rotating, {target1});
snapshots_rotating = sig_gen_rotating.generate_snapshots(t_axis, inf);

estimator_rotating = DoaEstimator(array_rotating, radar_params);
spectrum_rotating = estimator_rotating.estimate_gmusic(snapshots_rotating, t_axis, 1, search_grid);
[theta_est_rotating, phi_est_rotating, peak_val_rotating] = DoaEstimator.find_peaks(spectrum_rotating, search_grid, 1);

fprintf('旋转URA估算: theta=%.2f°, phi=%.2f° (峰值=%.2f)\n', theta_est_rotating, phi_est_rotating, peak_val_rotating);
fprintf('误差: Δtheta=%.2f°, Δphi=%.2f°\n\n', theta_est_rotating-theta1_true, phi_est_rotating-phi1_true);

% ============ 结果对比 ============
fprintf('=== 最终结果对比 ===\n');
fprintf('%-20s | %-12s | %-12s | %-15s | %-15s\n', '阵列类型', 'Theta估算', 'Phi估算', 'Theta误差', 'Phi误差');
fprintf('%s\n', repmat('-', 1, 85));
fprintf('%-20s | %-12.2f | %-12.2f | %-15.2f | %-15.2f\n', '静态URA', theta_est_static, phi_est_static, ...
    theta_est_static-theta1_true, phi_est_static-phi1_true);
fprintf('%-20s | %-12.2f | %-12.2f | %-15.2f | %-15.2f\n', '旋转URA (180°)', theta_est_rotating, phi_est_rotating, ...
    theta_est_rotating-theta1_true, phi_est_rotating-phi1_true);

if abs(phi_est_rotating - phi1_true) > 30
    fprintf('\n❌ 检测到严重的phi角估算错误！\n');
    fprintf('   旋转阵列给出phi=%.1f°，与真实值%.1f°相差%.1f°\n', ...
        phi_est_rotating, phi1_true, abs(phi_est_rotating - phi1_true));
    fprintf('   这与原始实验中的86.5°问题一致。\n');
else
    fprintf('\n✅ phi角估算误差可接受（<30°）。\n');
end

% 可视化
figure('Position', [100, 100, 1400, 500]);

subplot(1, 3, 1);
surf(search_grid.phi, search_grid.theta, spectrum_static);
shading interp; view(2); colorbar;
hold on;
plot(phi1_true, theta1_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
hold off;
title('静态4x2 URA');
xlabel('Phi (度)');
ylabel('Theta (度)');

subplot(1, 3, 2);
surf(search_grid.phi, search_grid.theta, spectrum_rotating);
shading interp; view(2); colorbar;
hold on;
plot(phi1_true, theta1_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
hold off;
title(sprintf('旋转4x2 URA (%d°)', total_rotation_angle_deg));
xlabel('Phi (度)');
ylabel('Theta (度)');

subplot(1, 3, 3);
% 对比1D切片（在真实theta附近）
[~, theta_idx] = min(abs(search_grid.theta - theta1_true));
plot(search_grid.phi, spectrum_static(theta_idx, :), 'b-', 'LineWidth', 2); hold on;
plot(search_grid.phi, spectrum_rotating(theta_idx, :), 'r-', 'LineWidth', 2);
xline(phi1_true, 'k--', 'LineWidth', 2);
legend('静态URA', '旋转URA', '真实phi', 'Location', 'best');
xlabel('Phi (度)');
ylabel('MUSIC谱值');
title(sprintf('Phi切片对比 (theta≈%.0f°)', theta1_true));
grid on;

