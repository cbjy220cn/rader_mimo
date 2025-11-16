%% 单目标测试 - 排查智能搜索的DOA估计问题
clear; clc; close all;

fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║        单目标测试 - 验证DOA估计精度                   ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

%% 雷达参数
c = physconst('LightSpeed');
f0 = 3e9;
lambda = c/f0;

radar_params.fc = f0;
radar_params.c = c;
radar_params.lambda = lambda;
radar_params.fs = 36100;
radar_params.T_chirp = 10e-3;
radar_params.slope = 5e12;
radar_params.BW = 50e6;
radar_params.num_samples = 361;
radar_params.range_res = c / (2 * radar_params.BW);

fprintf('📡 雷达参数: f₀=%.2f GHz\n\n', f0/1e9);

%% 测试多个单目标场景
test_cases = [
    30, 60;   % theta, phi
    30, 30;
    45, 60;
    30, 90;
];

fprintf('测试 %d 个单目标场景\n\n', size(test_cases, 1));

%% 阵列配置
num_elements = 8;
R_rx = 0.05;
theta_rx = linspace(0, 2*pi, num_elements+1); 
theta_rx(end) = [];
rx_elements = zeros(num_elements, 3);
for i = 1:num_elements
    rx_elements(i,:) = [R_rx*cos(theta_rx(i)), R_rx*sin(theta_rx(i)), 0];
end

num_snapshots = 64;
t_axis = (0:num_snapshots-1) * radar_params.T_chirp;
omega_dps = 360 / t_axis(end);

%% 搜索网格
smart_grid.coarse_res = 5.0;
smart_grid.fine_res = 0.2;
smart_grid.roi_margin = 10.0;
smart_grid.theta_range = [0, 90];
smart_grid.phi_range = [0, 180];

search_grid_full.theta = 0:0.2:90;
search_grid_full.phi = 0:0.2:180;

%% 运行测试
for test_idx = 1:size(test_cases, 1)
    theta_true = test_cases(test_idx, 1);
    phi_true = test_cases(test_idx, 2);
    
    fprintf('═══════════════════════════════════════════════════════\n');
    fprintf('测试 %d: 目标 theta=%.1f°, phi=%.1f°\n', test_idx, theta_true, phi_true);
    fprintf('═══════════════════════════════════════════════════════\n');
    
    % 创建目标
    target_range = 600;
    target_pos = [target_range * sind(theta_true) * cosd(phi_true), ...
                  target_range * sind(theta_true) * sind(phi_true), ...
                  target_range * cosd(theta_true)];
    targets = {Target(target_pos, [0,0,0], 1)};
    
    % 创建旋转阵列
    array_rotating = ArrayPlatform(rx_elements, 1, 1:num_elements);
    array_rotating = array_rotating.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]));
    
    % 生成信号
    sig_gen = SignalGenerator(radar_params, array_rotating, targets);
    snapshots = sig_gen.generate_snapshots(t_axis, inf);
    
    % 创建估计器
    estimator = DoaEstimatorIncoherent(array_rotating, radar_params);
    
    % 方法1: 传统全域搜索
    fprintf('\n  方法1: 传统全域搜索 ... ');
    tic;
    options.verbose = false;
    options.weighting = 'uniform';
    spectrum_full = estimator.estimate_incoherent_music(snapshots, t_axis, 1, search_grid_full, options);
    time_full = toc;
    [theta_est_full, phi_est_full, peak_val_full] = DoaEstimatorIncoherent.find_peaks(spectrum_full, search_grid_full, 1);
    
    fprintf('完成 (%.1fs)\n', time_full);
    fprintf('     估计: theta=%.2f°, phi=%.2f°\n', theta_est_full, phi_est_full);
    fprintf('     误差: Δθ=%.2f°, Δφ=%.2f°\n', abs(theta_est_full - theta_true), abs(phi_est_full - phi_true));
    fprintf('     峰值: %.2e\n', peak_val_full);
    
    % 方法2: 智能搜索
    fprintf('\n  方法2: 智能两步搜索 ... ');
    tic;
    [spectrum_smart, grid_smart] = smart_doa_search(estimator, snapshots, t_axis, 1, smart_grid, ...
        struct('verbose', false, 'weighting', 'uniform'));
    time_smart = toc;
    [theta_est_smart, phi_est_smart, peak_val_smart] = DoaEstimatorIncoherent.find_peaks(spectrum_smart, grid_smart, 1);
    
    fprintf('完成 (%.1fs, %.1fx加速)\n', time_smart, time_full/time_smart);
    fprintf('     估计: theta=%.2f°, phi=%.2f°\n', theta_est_smart, phi_est_smart);
    fprintf('     误差: Δθ=%.2f°, Δφ=%.2f°\n', abs(theta_est_smart - theta_true), abs(phi_est_smart - phi_true));
    fprintf('     峰值: %.2e\n', peak_val_smart);
    
    % 对比
    fprintf('\n  📊 对比:\n');
    fprintf('     角度误差差异: Δθ=%.3f°, Δφ=%.3f°\n', ...
        abs(theta_est_smart - theta_est_full), abs(phi_est_smart - phi_est_full));
    fprintf('     峰值差异: %.2f%%\n', abs(peak_val_smart - peak_val_full) / peak_val_full * 100);
    
    % 可视化对比
    figure('Position', [50 + test_idx*50, 50 + test_idx*50, 1400, 500]);
    
    subplot(1,2,1);
    surf(search_grid_full.phi, search_grid_full.theta, spectrum_full / max(spectrum_full(:)));
    shading interp; view(2); colorbar;
    caxis([0, 1]);
    hold on;
    plot(phi_true, theta_true, 'r+', 'MarkerSize', 20, 'LineWidth', 3);
    plot(phi_est_full, theta_est_full, 'go', 'MarkerSize', 12, 'LineWidth', 2);
    xlabel('Phi (°)');
    ylabel('Theta (°)');
    title(sprintf('传统搜索 (误差: θ=%.2f°, φ=%.2f°)', ...
        abs(theta_est_full - theta_true), abs(phi_est_full - phi_true)));
    xlim([max(0, phi_true-20), min(180, phi_true+20)]);
    ylim([max(0, theta_true-20), min(90, theta_true+20)]);
    
    subplot(1,2,2);
    surf(grid_smart.phi, grid_smart.theta, spectrum_smart / max(spectrum_smart(:)));
    shading interp; view(2); colorbar;
    caxis([0, 1]);
    hold on;
    plot(phi_true, theta_true, 'r+', 'MarkerSize', 20, 'LineWidth', 3);
    plot(phi_est_smart, theta_est_smart, 'go', 'MarkerSize', 12, 'LineWidth', 2);
    xlabel('Phi (°)');
    ylabel('Theta (°)');
    title(sprintf('智能搜索 (误差: θ=%.2f°, φ=%.2f°)', ...
        abs(theta_est_smart - theta_true), abs(phi_est_smart - phi_true)));
    xlim([max(0, phi_true-20), min(180, phi_true+20)]);
    ylim([max(0, theta_true-20), min(90, theta_true+20)]);
    
    sgtitle(sprintf('测试%d: 真实值 θ=%.1f°, φ=%.1f°', test_idx, theta_true, phi_true), ...
        'FontSize', 14, 'FontWeight', 'bold');
    
    fprintf('\n');
end

fprintf('═══════════════════════════════════════════════════════\n');
fprintf('✅ 测试完成\n');
fprintf('═══════════════════════════════════════════════════════\n');

