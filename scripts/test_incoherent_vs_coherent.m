%% 对比相干GMUSIC vs 非相干MUSIC
clear; clc;

fprintf('=== 相干GMUSIC vs 非相干MUSIC 性能对比 ===\n\n');

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

% 8元URA (4x2)
num_elements = 8;
spacing = radar_params.lambda / 2;
Nx = 4;
Ny = 2;
x_coords = (-(Nx-1)/2:(Nx-1)/2) * spacing;
y_coords = (-(Ny-1)/2:(Ny-1)/2) * spacing;
[X, Y] = meshgrid(x_coords, y_coords);
rect_elements = [X(:), Y(:), zeros(num_elements, 1)];

% 目标
target_range = 20;
theta_true = 30;
phi_true = 20;
target_pos = [target_range * sind(theta_true) * cosd(phi_true), ...
              target_range * sind(theta_true) * sind(phi_true), ...
              target_range * cosd(theta_true)];
target = Target(target_pos, [0, 0, 0], 1);

fprintf('目标参数:\n');
fprintf('  位置: [%.2f, %.2f, %.2f] m\n', target_pos(1), target_pos(2), target_pos(3));
fprintf('  真实角度: theta=%.1f°, phi=%.1f°\n\n', theta_true, phi_true);

% 搜索网格
search_grid.theta = 0:0.5:90;
search_grid.phi = -90:0.5:90;

% 测试不同旋转角度
rotation_angles = [0, 30, 90, 180, 360];

fprintf('%-12s | %-25s | %-25s\n', '旋转角度', '相干GMUSIC', '非相干MUSIC');
fprintf('%-12s | %-10s %-10s %-4s | %-10s %-10s %-4s\n', ...
    '', 'Theta', 'Phi', '峰值', 'Theta', 'Phi', '峰值');
fprintf('%s\n', repmat('-', 1, 80));

for total_rotation = rotation_angles
    % 时间轴
    num_snapshots = 64;
    T_chirp = 1e-4;
    t_axis = (0:num_snapshots-1) * T_chirp;
    
    if total_rotation == 0
        omega_dps = 0;
    else
        omega_dps = total_rotation / t_axis(end);
    end
    
    % 创建旋转阵列
    array_rotating = ArrayPlatform(rect_elements, 1, 1:num_elements);
    rotation_traj = @(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]);
    array_rotating = array_rotating.set_trajectory(rotation_traj);
    
    % 生成信号
    sig_gen = SignalGenerator(radar_params, array_rotating, {target});
    snapshots = sig_gen.generate_snapshots(t_axis, inf);
    
    % 方法1: 相干GMUSIC
    estimator_coherent = DoaEstimator(array_rotating, radar_params);
    spectrum_coherent = estimator_coherent.estimate_gmusic(snapshots, t_axis, 1, search_grid);
    [theta_coh, phi_coh, peak_coh] = DoaEstimator.find_peaks(spectrum_coherent, search_grid, 1);
    
    % 方法2: 非相干MUSIC
    estimator_incoherent = DoaEstimatorIncoherent(array_rotating, radar_params);
    options.verbose = false;
    options.weighting = 'uniform';
    spectrum_incoh = estimator_incoherent.estimate_incoherent_music(snapshots, t_axis, 1, search_grid, options);
    [theta_incoh, phi_incoh, peak_incoh] = DoaEstimatorIncoherent.find_peaks(spectrum_incoh, search_grid, 1);
    
    % 打印结果
    fprintf('%-12.0f | %-10.1f %-10.1f %.2f | %-10.1f %-10.1f %.2f\n', ...
        total_rotation, theta_coh, phi_coh, peak_coh, ...
        theta_incoh, phi_incoh, peak_incoh);
end

fprintf('\n=== 详细分析：180度旋转 ===\n\n');

% 重新计算180度的详细结果
total_rotation = 180;
num_snapshots = 64;
t_axis = (0:num_snapshots-1) * T_chirp;
omega_dps = total_rotation / t_axis(end);

array_rotating = ArrayPlatform(rect_elements, 1, 1:num_elements);
rotation_traj = @(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]);
array_rotating = array_rotating.set_trajectory(rotation_traj);

sig_gen = SignalGenerator(radar_params, array_rotating, {target});
snapshots = sig_gen.generate_snapshots(t_axis, inf);

% 相干GMUSIC
estimator_coherent = DoaEstimator(array_rotating, radar_params);
spectrum_coherent = estimator_coherent.estimate_gmusic(snapshots, t_axis, 1, search_grid);
[theta_coh, phi_coh, peak_coh] = DoaEstimator.find_peaks(spectrum_coherent, search_grid, 1);

% 非相干MUSIC
estimator_incoherent = DoaEstimatorIncoherent(array_rotating, radar_params);
spectrum_incoh = estimator_incoherent.estimate_incoherent_music(snapshots, t_axis, 1, search_grid, options);
[theta_incoh, phi_incoh, peak_incoh] = DoaEstimatorIncoherent.find_peaks(spectrum_incoh, search_grid, 1);

fprintf('相干GMUSIC结果:\n');
fprintf('  估算: theta=%.1f°, phi=%.1f°\n', theta_coh, phi_coh);
fprintf('  误差: Δtheta=%.1f°, Δphi=%.1f°\n', theta_coh-theta_true, phi_coh-phi_true);
fprintf('  峰值: %.2f\n\n', peak_coh);

fprintf('非相干MUSIC结果:\n');
fprintf('  估算: theta=%.1f°, phi=%.1f°\n', theta_incoh, phi_incoh);
fprintf('  误差: Δtheta=%.1f°, Δphi=%.1f°\n', theta_incoh-theta_true, phi_incoh-phi_true);
fprintf('  峰值: %.2f\n\n', peak_incoh);

% 可视化对比
figure('Position', [100, 100, 1400, 500]);

subplot(1, 2, 1);
surf(search_grid.phi, search_grid.theta, spectrum_coherent);
shading interp; view(2); colorbar;
hold on;
plot(phi_true, theta_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
hold off;
title(sprintf('相干GMUSIC (180°旋转)\nPhi误差=%.1f°', phi_coh-phi_true));
xlabel('Phi (度)');
ylabel('Theta (度)');

subplot(1, 2, 2);
surf(search_grid.phi, search_grid.theta, spectrum_incoh);
shading interp; view(2); colorbar;
hold on;
plot(phi_true, theta_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
hold off;
title(sprintf('非相干MUSIC (180°旋转)\nPhi误差=%.1f°', phi_incoh-phi_true));
xlabel('Phi (度)');
ylabel('Theta (度)');

sgtitle('旋转阵列DOA估算：相干 vs 非相干方法');

fprintf('结论：\n');
if abs(phi_incoh - phi_true) < abs(phi_coh - phi_true)
    fprintf('✅ 非相干MUSIC在旋转阵列上性能显著优于相干GMUSIC\n');
    fprintf('   推荐使用DoaEstimatorIncoherent类进行运动阵列的DOA估算\n');
else
    fprintf('⚠️ 两种方法性能相近，可能需要进一步调优\n');
end

