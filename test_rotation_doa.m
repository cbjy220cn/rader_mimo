%% 专门测试旋转阵列DOA估算的简化脚本
clear; clc;

fprintf('=== 旋转阵列DOA估算简化测试 ===\n\n');

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

% 创建一个简单的ULA (4个元素沿x轴)
num_elements = 4;
spacing = radar_params.lambda / 2;
x_coords = (-(num_elements-1)/2:(num_elements-1)/2) * spacing;
elements = [x_coords', zeros(num_elements, 1), zeros(num_elements, 1)];

fprintf('阵列元素位置:\n');
for i = 1:num_elements
    fprintf('  元素%d: [%.6f, %.6f, %.6f]\n', i, elements(i,1), elements(i,2), elements(i,3));
end
fprintf('\n');

% 目标设置：在xz平面内，theta=30度，phi=0度
range_target = 30;
theta_true = 30;  % 俯仰角
phi_true = 0;     % 方位角（在xz平面内）
target_pos = [range_target * sind(theta_true) * cosd(phi_true), ...
              range_target * sind(theta_true) * sind(phi_true), ...
              range_target * cosd(theta_true)];
target = Target(target_pos, [0,0,0], 1);

fprintf('目标位置: [%.2f, %.2f, %.2f] m\n', target_pos(1), target_pos(2), target_pos(3));
fprintf('真实角度: theta=%.1f°, phi=%.1f°\n\n', theta_true, phi_true);

% 搜索网格
search_grid.theta = 0:1:90;
search_grid.phi = -45:1:45;

% ============ 测试1: 静态阵列 ============
fprintf('--- 测试1: 静态阵列 ---\n');
array_static = ArrayPlatform(elements, 1, 1:num_elements);
array_static = array_static.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

t_axis = 0:1e-4:1e-3;  % 短时间窗口
sig_gen_static = SignalGenerator(radar_params, array_static, {target});
snapshots_static = sig_gen_static.generate_snapshots(t_axis, inf);

estimator_static = DoaEstimator(array_static, radar_params);
spectrum_static = estimator_static.estimate_gmusic(snapshots_static, t_axis, 1, search_grid);
[theta_est_static, phi_est_static, ~] = DoaEstimator.find_peaks(spectrum_static, search_grid, 1);

fprintf('静态阵列估算: theta=%.1f°, phi=%.1f°\n', theta_est_static, phi_est_static);
fprintf('误差: Δtheta=%.1f°, Δphi=%.1f°\n\n', theta_est_static-theta_true, phi_est_static-phi_true);

% ============ 测试2: 旋转阵列 (小角度) ============
fprintf('--- 测试2: 旋转阵列 (旋转10度) ---\n');
total_rotation = 10;  % 总旋转角度（度）
T_total = t_axis(end);
omega_dps = total_rotation / T_total;  % 度/秒

array_rotating = ArrayPlatform(elements, 1, 1:num_elements);
rotation_traj = @(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]);
array_rotating = array_rotating.set_trajectory(rotation_traj);

sig_gen_rotating = SignalGenerator(radar_params, array_rotating, {target});
snapshots_rotating = sig_gen_rotating.generate_snapshots(t_axis, inf);

fprintf('旋转阵列信息:\n');
fprintf('  初始orientation: [0, 0, %.1f]°\n', omega_dps * t_axis(1));
fprintf('  最终orientation: [0, 0, %.1f]°\n', omega_dps * t_axis(end));
fprintf('  快拍数: %d\n\n', length(t_axis));

% 打印前3个虚拟元素的位置变化
fprintf('虚拟元素位置随时间变化(仅显示前3个):\n');
for k = [1, ceil(length(t_axis)/2), length(t_axis)]
    virtual_pos = array_rotating.get_mimo_virtual_positions(t_axis(k));
    fprintf('  t=%.4fs (yaw=%.1f°):\n', t_axis(k), omega_dps * t_axis(k));
    for i = 1:min(3, size(virtual_pos, 1))
        fprintf('    元素%d: [%.6f, %.6f, %.6f]\n', i, virtual_pos(i,1), virtual_pos(i,2), virtual_pos(i,3));
    end
end
fprintf('\n');

estimator_rotating = DoaEstimator(array_rotating, radar_params);
spectrum_rotating = estimator_rotating.estimate_gmusic(snapshots_rotating, t_axis, 1, search_grid);
[theta_est_rotating, phi_est_rotating, ~] = DoaEstimator.find_peaks(spectrum_rotating, search_grid, 1);

fprintf('旋转阵列估算: theta=%.1f°, phi=%.1f°\n', theta_est_rotating, phi_est_rotating);
fprintf('误差: Δtheta=%.1f°, Δphi=%.1f°\n\n', theta_est_rotating-theta_true, phi_est_rotating-phi_true);

% ============ 结果对比 ============
fprintf('=== 结果对比 ===\n');
fprintf('%-15s | %-12s | %-12s | %-15s | %-15s\n', '阵列类型', 'Theta估算', 'Phi估算', 'Theta误差', 'Phi误差');
fprintf('%s\n', repmat('-', 1, 80));
fprintf('%-15s | %-12.1f | %-12.1f | %-15.1f | %-15.1f\n', '静态', theta_est_static, phi_est_static, ...
    theta_est_static-theta_true, phi_est_static-phi_true);
fprintf('%-15s | %-12.1f | %-12.1f | %-15.1f | %-15.1f\n', '旋转', theta_est_rotating, phi_est_rotating, ...
    theta_est_rotating-theta_true, phi_est_rotating-phi_true);

% 可视化对比
figure('Position', [100, 100, 1200, 400]);

subplot(1, 2, 1);
surf(search_grid.phi, search_grid.theta, spectrum_static);
shading interp; view(2); colorbar;
hold on;
plot(phi_true, theta_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
hold off;
title('静态阵列MUSIC谱');
xlabel('Phi (度)');
ylabel('Theta (度)');

subplot(1, 2, 2);
surf(search_grid.phi, search_grid.theta, spectrum_rotating);
shading interp; view(2); colorbar;
hold on;
plot(phi_true, theta_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
hold off;
title('旋转阵列MUSIC谱');
xlabel('Phi (度)');
ylabel('Theta (度)');

