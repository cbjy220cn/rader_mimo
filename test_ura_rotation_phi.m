%% 测试4x2 URA旋转时对非零phi角的估算
clear; clc;

fprintf('=== 4x2 URA旋转阵列 phi角估算测试 ===\n\n');

% 雷达参数
radar_params.fc = 77e9;
radar_params.c = physconst('LightSpeed');
radar_params.lambda = radar_params.c / radar_params.fc;
radar_params.fs = 20e6;
radar_params.T_chirp = 50e-6;
radar_params.slope = 100e12;
radar_params.BW = radar_params.slope * radar_params.T_chirp;
radar_params.num_samples = radar_params.T_chirp * radar_params.fs;
radar_params.range_res = radar_params.c / (2 * radar_params.BW);

% 创建4x2 URA（与原始实验相同）
spacing = radar_params.lambda / 2;
Nx = 4;
Ny = 2;
x_coords = (-(Nx-1)/2:(Nx-1)/2) * spacing;
y_coords = (-(Ny-1)/2:(Ny-1)/2) * spacing;
[X, Y] = meshgrid(x_coords, y_coords);
rect_elements = [X(:), Y(:), zeros(Nx*Ny, 1)];

fprintf('4x2 URA阵列元素位置 (xy平面):\n');
for i = 1:Nx*Ny
    fprintf('  元素%d: [%.6f, %.6f, %.6f]\n', i, rect_elements(i,1), rect_elements(i,2), rect_elements(i,3));
end
fprintf('\n');

% 目标设置：使用与原始实验相同的角度
range_target = 20;
theta_true = 30;
phi_true = 20;  % ← 关键：非零phi角
target_pos = [range_target * sind(theta_true) * cosd(phi_true), ...
              range_target * sind(theta_true) * sind(phi_true), ...
              range_target * cosd(theta_true)];
target = Target(target_pos, [0,0,0], 1);

fprintf('目标位置: [%.2f, %.2f, %.2f] m\n', target_pos(1), target_pos(2), target_pos(3));
fprintf('真实角度: theta=%.1f°, phi=%.1f°\n\n', theta_true, phi_true);

% 搜索网格（使用较粗的网格以加快速度）
search_grid.theta = 0:1:90;
search_grid.phi = -90:1:90;

% 时间轴设置（与原始实验接近）
num_snapshots = 64;  % 减少到64以加快测试
T_chirp = 1e-4;
t_axis = (0:num_snapshots-1) * T_chirp;
total_rotation = 90;  % 旋转90度（减少旋转角度）
omega_dps = total_rotation / t_axis(end);

fprintf('旋转设置:\n');
fprintf('  快拍数: %d\n', num_snapshots);
fprintf('  总旋转角度: %.1f°\n', total_rotation);
fprintf('  角速度: %.1f °/s\n\n', omega_dps);

% ============ 测试1: 静态4x2 URA ============
fprintf('--- 测试1: 静态4x2 URA ---\n');
array_static = ArrayPlatform(rect_elements, 1, 1:Nx*Ny);
array_static = array_static.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

sig_gen_static = SignalGenerator(radar_params, array_static, {target});
snapshots_static = sig_gen_static.generate_snapshots(t_axis, inf);

estimator_static = DoaEstimator(array_static, radar_params);
spectrum_static = estimator_static.estimate_gmusic(snapshots_static, t_axis, 1, search_grid);
[theta_est_static, phi_est_static, peak_val_static] = DoaEstimator.find_peaks(spectrum_static, search_grid, 1);

fprintf('静态URA估算: theta=%.1f°, phi=%.1f° (峰值=%.2f)\n', theta_est_static, phi_est_static, peak_val_static);
fprintf('误差: Δtheta=%.1f°, Δphi=%.1f°\n\n', theta_est_static-theta_true, phi_est_static-phi_true);

% ============ 测试2: 旋转4x2 URA ============
fprintf('--- 测试2: 旋转4x2 URA (绕z轴) ---\n');
array_rotating = ArrayPlatform(rect_elements, 1, 1:Nx*Ny);
rotation_traj = @(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]);
array_rotating = array_rotating.set_trajectory(rotation_traj);

sig_gen_rotating = SignalGenerator(radar_params, array_rotating, {target});
snapshots_rotating = sig_gen_rotating.generate_snapshots(t_axis, inf);

% 打印几个时刻的虚拟元素位置
fprintf('虚拟元素位置随时间变化(前2个):\n');
for k = [1, ceil(num_snapshots/2), num_snapshots]
    virtual_pos = array_rotating.get_mimo_virtual_positions(t_axis(k));
    fprintf('  t=%.4fs (yaw=%.1f°):\n', t_axis(k), omega_dps * t_axis(k));
    for i = 1:min(2, size(virtual_pos, 1))
        fprintf('    元素%d: [%.6f, %.6f, %.6f]\n', i, virtual_pos(i,1), virtual_pos(i,2), virtual_pos(i,3));
    end
end
fprintf('\n');

estimator_rotating = DoaEstimator(array_rotating, radar_params);
spectrum_rotating = estimator_rotating.estimate_gmusic(snapshots_rotating, t_axis, 1, search_grid);
[theta_est_rotating, phi_est_rotating, peak_val_rotating] = DoaEstimator.find_peaks(spectrum_rotating, search_grid, 1);

fprintf('旋转URA估算: theta=%.1f°, phi=%.1f° (峰值=%.2f)\n', theta_est_rotating, phi_est_rotating, peak_val_rotating);
fprintf('误差: Δtheta=%.1f°, Δphi=%.1f°\n\n', theta_est_rotating-theta_true, phi_est_rotating-phi_true);

% ============ 结果对比 ============
fprintf('=== 结果对比 (4x2 URA, phi=20°) ===\n');
fprintf('%-15s | %-12s | %-12s | %-15s | %-15s\n', '阵列类型', 'Theta估算', 'Phi估算', 'Theta误差', 'Phi误差');
fprintf('%s\n', repmat('-', 1, 80));
fprintf('%-15s | %-12.1f | %-12.1f | %-15.1f | %-15.1f\n', '静态URA', theta_est_static, phi_est_static, ...
    theta_est_static-theta_true, phi_est_static-phi_true);
fprintf('%-15s | %-12.1f | %-12.1f | %-15.1f | %-15.1f\n', '旋转URA', theta_est_rotating, phi_est_rotating, ...
    theta_est_rotating-theta_true, phi_est_rotating-phi_true);

if abs(phi_est_rotating - phi_true) > 30
    fprintf('\n❌ 检测到严重的phi角估算错误！\n');
    fprintf('   旋转阵列给出phi=%.1f°，与真实值%.1f°相差%.1f°\n', ...
        phi_est_rotating, phi_true, abs(phi_est_rotating - phi_true));
    fprintf('   这可能是旋转引起的方位角模糊问题。\n');
else
    fprintf('\n✅ phi角估算正常。\n');
end

% 可视化
figure('Position', [100, 100, 1400, 500]);

subplot(1, 3, 1);
surf(search_grid.phi, search_grid.theta, spectrum_static);
shading interp; view(2); colorbar;
hold on;
plot(phi_true, theta_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
hold off;
title('静态4x2 URA');
xlabel('Phi (度)');
ylabel('Theta (度)');

subplot(1, 3, 2);
surf(search_grid.phi, search_grid.theta, spectrum_rotating);
shading interp; view(2); colorbar;
hold on;
plot(phi_true, theta_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
hold off;
title('旋转4x2 URA');
xlabel('Phi (度)');
ylabel('Theta (度)');

subplot(1, 3, 3);
% 对比1D切片
[~, theta_idx] = min(abs(search_grid.theta - theta_true));
plot(search_grid.phi, spectrum_static(theta_idx, :), 'b-', 'LineWidth', 2); hold on;
plot(search_grid.phi, spectrum_rotating(theta_idx, :), 'r-', 'LineWidth', 2);
xline(phi_true, 'k--', 'LineWidth', 2);
legend('静态URA', '旋转URA', '真实phi', 'Location', 'best');
xlabel('Phi (度)');
ylabel('MUSIC谱值');
title(sprintf('Phi切片对比 (theta=%.0f°)', search_grid.theta(theta_idx)));
grid on;

