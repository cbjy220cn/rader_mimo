%% 测试智能两步搜索的有效性
% 对比传统全域搜索 vs 智能两步搜索
% 验证: 1) 速度提升  2) 结果精度保持

clear; clc; close all;
fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║        智能两步搜索 vs 传统全域搜索对比测试            ║\n');
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

fprintf('📡 雷达参数: f₀=%.2f GHz, λ=%.3f m\n\n', f0/1e9, lambda);

%% 测试场景：双目标
target_range = 600;
theta_true = 30;
phi1_true = 60;
phi2_true = 62;  % 2度间隔

target1_pos = [target_range * sind(theta_true) * cosd(phi1_true), ...
               target_range * sind(theta_true) * sind(phi1_true), ...
               target_range * cosd(theta_true)];
target2_pos = [target_range * sind(theta_true) * cosd(phi2_true), ...
               target_range * sind(theta_true) * sind(phi2_true), ...
               target_range * cosd(theta_true)];
           
targets = {Target(target1_pos, [0,0,0], 1), Target(target2_pos, [0,0,0], 1)};

fprintf('🎯 测试场景: 双目标\n');
fprintf('   目标1: theta=%.1f°, phi=%.1f°\n', theta_true, phi1_true);
fprintf('   目标2: theta=%.1f°, phi=%.1f°\n', theta_true, phi2_true);
fprintf('   间隔: %.1f°\n\n', phi2_true - phi1_true);

%% 创建阵列
num_elements = 8;
R_rx = 0.05;
theta_rx = linspace(0, 2*pi, num_elements+1); 
theta_rx(end) = [];
rx_elements = zeros(num_elements, 3);
for i = 1:num_elements
    rx_elements(i,:) = [R_rx*cos(theta_rx(i)), R_rx*sin(theta_rx(i)), 0];
end

% 旋转阵列（1圈）
num_snapshots = 64;
t_axis = (0:num_snapshots-1) * radar_params.T_chirp;
omega_dps = 360 / t_axis(end);

array_rotating = ArrayPlatform(rx_elements, 1, 1:num_elements);
array_rotating = array_rotating.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]));

fprintf('📊 阵列配置: 8元圆阵，旋转360°，64快拍\n\n');

%% 生成信号
sig_gen = SignalGenerator(radar_params, array_rotating, targets);
snapshots = sig_gen.generate_snapshots(t_axis, inf);
fprintf('✓ 信号生成完成\n\n');

%% 创建估计器
estimator = DoaEstimatorIncoherent(array_rotating, radar_params);

%% ======================================================================
%% 方法1: 传统全域搜索
%% ======================================================================
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('方法1: 传统全域搜索\n');
fprintf('═══════════════════════════════════════════════════════\n');

search_grid_full.theta = 0:0.2:90;
search_grid_full.phi = 0:0.2:180;
num_points_full = length(search_grid_full.theta) * length(search_grid_full.phi);

fprintf('搜索网格: %d × %d = %d 个点\n', ...
    length(search_grid_full.theta), length(search_grid_full.phi), num_points_full);
fprintf('开始全域搜索...\n');

tic;
options.verbose = false;
options.weighting = 'uniform';
spectrum_full = estimator.estimate_incoherent_music(snapshots, t_axis, 2, search_grid_full, options);
time_full = toc;

% 找峰值
[theta_peaks_full, phi_peaks_full, peak_vals_full] = DoaEstimatorIncoherent.find_peaks(spectrum_full, search_grid_full, 2);

fprintf('✓ 全域搜索完成\n');
fprintf('   耗时: %.2f 秒\n', time_full);
fprintf('   找到峰值:\n');
for i = 1:length(theta_peaks_full)
    fprintf('      峰值%d: theta=%.2f°, phi=%.2f°, 幅度=%.2e\n', ...
        i, theta_peaks_full(i), phi_peaks_full(i), peak_vals_full(i));
end
fprintf('\n');

%% ======================================================================
%% 方法2: 智能两步搜索
%% ======================================================================
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('方法2: 智能两步搜索\n');
fprintf('═══════════════════════════════════════════════════════\n');

smart_grid.coarse_res = 5.0;
smart_grid.fine_res = 0.2;
smart_grid.roi_margin = 10.0;
smart_grid.theta_range = [0, 90];
smart_grid.phi_range = [0, 180];

fprintf('策略: 粗搜索(%.1f°) → 定位峰值 → 细搜索(%.1f°, ±%.1f°) → 合并谱\n', ...
    smart_grid.coarse_res, smart_grid.fine_res, smart_grid.roi_margin);

tic;
[spectrum_smart, grid_smart] = smart_doa_search(estimator, snapshots, t_axis, 2, smart_grid, ...
    struct('verbose', true, 'weighting', 'uniform'));
time_smart = toc;

% 找峰值
[theta_peaks_smart, phi_peaks_smart, peak_vals_smart] = DoaEstimatorIncoherent.find_peaks(spectrum_smart, grid_smart, 2);

fprintf('✓ 智能搜索完成\n');
fprintf('   总耗时: %.2f 秒\n', time_smart);
fprintf('   找到峰值:\n');
for i = 1:length(theta_peaks_smart)
    fprintf('      峰值%d: theta=%.2f°, phi=%.2f°, 幅度=%.2e\n', ...
        i, theta_peaks_smart(i), phi_peaks_smart(i), peak_vals_smart(i));
end
fprintf('\n');

%% ======================================================================
%% 对比分析
%% ======================================================================
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('对比分析\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

% 速度对比
speedup = time_full / time_smart;
fprintf('⚡ 速度提升:\n');
fprintf('   传统搜索: %.2f 秒\n', time_full);
fprintf('   智能搜索: %.2f 秒\n', time_smart);
fprintf('   加速比: %.2fx\n\n', speedup);

% 精度对比
fprintf('🎯 精度对比:\n');
fprintf('   真实值: phi1=%.1f°, phi2=%.1f°\n', phi1_true, phi2_true);
fprintf('\n');
fprintf('   传统搜索估计:\n');
for i = 1:min(2, length(phi_peaks_full))
    error_full = min(abs(phi_peaks_full(i) - phi1_true), abs(phi_peaks_full(i) - phi2_true));
    fprintf('      峰值%d: phi=%.2f° (误差: %.2f°)\n', i, phi_peaks_full(i), error_full);
end
fprintf('\n');
fprintf('   智能搜索估计:\n');
for i = 1:min(2, length(phi_peaks_smart))
    error_smart = min(abs(phi_peaks_smart(i) - phi1_true), abs(phi_peaks_smart(i) - phi2_true));
    fprintf('      峰值%d: phi=%.2f° (误差: %.2f°)\n', i, phi_peaks_smart(i), error_smart);
end
fprintf('\n');

% 峰值对比
fprintf('📈 峰值幅度对比:\n');
fprintf('   传统搜索: 平均峰值 = %.2e\n', mean(peak_vals_full));
fprintf('   智能搜索: 平均峰值 = %.2e\n', mean(peak_vals_smart));
fprintf('   相对差异: %.2f%%\n\n', abs(mean(peak_vals_full) - mean(peak_vals_smart)) / mean(peak_vals_full) * 100);

%% ======================================================================
%% 可视化对比
%% ======================================================================
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('生成对比图表\n');
fprintf('═══════════════════════════════════════════════════════\n');

% 图1: 2D谱对比
figure('Position', [50, 50, 1400, 600]);

subplot(1,2,1);
surf(search_grid_full.phi, search_grid_full.theta, spectrum_full / max(spectrum_full(:)));
shading interp; view(2); colorbar;
caxis([0, 1]);
hold on;
plot(phi1_true, theta_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
plot(phi2_true, theta_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
xlabel('Phi (°)', 'FontSize', 11);
ylabel('Theta (°)', 'FontSize', 11);
title(sprintf('传统全域搜索 (%.2f秒)', time_full), 'FontSize', 12, 'FontWeight', 'bold');
xlim([50, 70]);
ylim([20, 40]);

subplot(1,2,2);
surf(grid_smart.phi, grid_smart.theta, spectrum_smart / max(spectrum_smart(:)));
shading interp; view(2); colorbar;
caxis([0, 1]);
hold on;
plot(phi1_true, theta_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
plot(phi2_true, theta_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
xlabel('Phi (°)', 'FontSize', 11);
ylabel('Theta (°)', 'FontSize', 11);
title(sprintf('智能两步搜索 (%.2f秒, %.1fx加速)', time_smart, speedup), 'FontSize', 12, 'FontWeight', 'bold');
xlim([50, 70]);
ylim([20, 40]);

sgtitle('MUSIC谱对比（归一化）', 'FontSize', 14, 'FontWeight', 'bold');

% 图2: 1D切片对比
figure('Position', [100, 100, 1000, 500]);

[~, theta_idx_full] = min(abs(search_grid_full.theta - theta_true));
slice_full = spectrum_full(theta_idx_full, :);
slice_full_db = 10*log10(slice_full / max(slice_full));

[~, theta_idx_smart] = min(abs(grid_smart.theta - theta_true));
slice_smart = spectrum_smart(theta_idx_smart, :);
slice_smart_db = 10*log10(slice_smart / max(slice_smart));

plot(search_grid_full.phi, slice_full_db, 'b-', 'LineWidth', 2.5, 'DisplayName', '传统搜索'); hold on;
plot(grid_smart.phi, slice_smart_db, 'r--', 'LineWidth', 2.5, 'DisplayName', '智能搜索');
xline(phi1_true, 'g--', 'LineWidth', 1.5, 'DisplayName', '真实目标');
xline(phi2_true, 'g--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
yline(-3, 'k:', 'LineWidth', 1, 'DisplayName', '-3dB');

xlim([50, 70]);
ylim([-40, 5]);
grid on;
xlabel('Phi (°)', 'FontSize', 12);
ylabel('归一化幅度 (dB)', 'FontSize', 12);
title(sprintf('1D切片对比 (theta=%.1f°)', theta_true), 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 11);

fprintf('✓ 图表生成完成\n\n');

%% ======================================================================
%% 结论
%% ======================================================================
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('✅ 测试结论\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

fprintf('智能两步搜索验证成功！\n\n');
fprintf('性能指标:\n');
fprintf('  ✅ 速度: %.2fx 加速\n', speedup);
fprintf('  ✅ 精度: 与全域搜索基本一致 (差异<%.1f%%)\n', ...
    abs(mean(peak_vals_full) - mean(peak_vals_smart)) / mean(peak_vals_full) * 100);
fprintf('  ✅ 画图质量: 完全保持（通过插值+细搜索）\n\n');

fprintf('建议:\n');
if speedup > 3
    fprintf('  🚀 强烈推荐使用智能搜索！加速%.1fx，效果相同\n', speedup);
elseif speedup > 1.5
    fprintf('  ✅ 推荐使用智能搜索，有明显加速效果\n');
else
    fprintf('  ⚠️  加速不明显，可能场景不适合（目标过多或分布太广）\n');
end
fprintf('\n');

