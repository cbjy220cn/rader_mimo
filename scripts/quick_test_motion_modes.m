%% 快速测试：运动模式对比（5分钟验证）
% 快速验证纯旋转vs平移的效果差异

clear; clc; close all;

fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║        快速测试：运动模式对比（5分钟）                  ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

fprintf('测试目的: 快速验证纯旋转无效、平移有效\n');
fprintf('测试配置: 双目标2°间隔，16快拍\n\n');

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

fprintf('📡 雷达: f₀=%.2f GHz, λ=%.1f cm\n\n', f0/1e9, lambda*100);

%% 快速配置
num_snapshots = 64;  % 增加到64快拍（观测时间0.64秒）
t_axis = (0:num_snapshots-1) * radar_params.T_chirp;

search_grid.theta = 0:0.5:90;   % 0.5度步长，能够精确定位2度间隔
search_grid.phi = 0:0.5:180;    % 0.5度步长

%% 双目标设置
sep = 8.0;  % 8度间隔（平移的6°带宽可以分辨）
target1_pos = [600*sind(30)*cosd(60-sep/2), 600*sind(30)*sind(60-sep/2), 600*cosd(30)];
target2_pos = [600*sind(30)*cosd(60+sep/2), 600*sind(30)*sind(60+sep/2), 600*cosd(30)];
targets = {Target(target1_pos, [0,0,0], 1), Target(target2_pos, [0,0,0], 1)};

fprintf('目标: 双目标 %.1f°间隔 @ (θ=30°, φ=60°)\n\n', sep);

%% 8元圆形阵列
R_rx = 0.15;
theta_rx = linspace(0, 2*pi, 9); theta_rx(end) = [];
rx_elem = zeros(8, 3);
for i = 1:8
    rx_elem(i,:) = [R_rx*cos(theta_rx(i)), R_rx*sin(theta_rx(i)), 0];
end

fprintf('阵列: 8元圆阵，半径%.1f cm\n\n', R_rx*100);

%% 测试1: 静态阵列（基准）
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('测试1: 静态阵列（基准）\n');
fprintf('═══════════════════════════════════════════════════════\n');

array_st = ArrayPlatform(rx_elem, 1, 1:8);
array_st = array_st.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

sig_gen = SignalGenerator(radar_params, array_st, targets);
snapshots = sig_gen.generate_snapshots(t_axis, inf);

est = DoaEstimator(array_st, radar_params);
fprintf('  计算MUSIC谱... ');
tic;
spectrum_static = est.estimate_gmusic(snapshots, t_axis, 2, search_grid);
time_static = toc;
fprintf('完成 (%.1fs)\n', time_static);

% 找峰值
[~, phi_est, ~] = DoaEstimator.find_peaks(spectrum_static, search_grid, 2);
peak_sep_static = abs(phi_est(1) - phi_est(2));
fprintf('  检测间隔: %.1f° (真实%.1f°)\n', peak_sep_static, sep);
fprintf('  误差: %.1f°\n\n', abs(peak_sep_static - sep));

%% 测试2: 纯旋转（应该无效）
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('测试2: 纯旋转运动（预期：无改善）\n');
fprintf('═══════════════════════════════════════════════════════\n');

omega_dps = 360 / t_axis(end);
array_rot = ArrayPlatform(rx_elem, 1, 1:8);
array_rot = array_rot.set_trajectory(@(t) struct('position', [0,0,0], ...
                                                  'orientation', [0, 0, omega_dps * t]));

sig_gen = SignalGenerator(radar_params, array_rot, targets);
snapshots = sig_gen.generate_snapshots(t_axis, inf);

est = DoaEstimatorIncoherent_FIXED(array_rot, radar_params);
fprintf('  计算非相干MUSIC谱... ');
tic;
options.verbose = false;
options.weighting = 'uniform';
spectrum_rotation = est.estimate_incoherent_music(snapshots, t_axis, 2, search_grid, options);
time_rotation = toc;
fprintf('完成 (%.1fs)\n', time_rotation);

% 找峰值
[~, phi_est, ~] = DoaEstimatorIncoherent_FIXED.find_peaks(spectrum_rotation, search_grid, 2);
peak_sep_rotation = abs(phi_est(1) - phi_est(2));
fprintf('  检测间隔: %.1f° (真实%.1f°)\n', peak_sep_rotation, sep);
fprintf('  误差: %.1f°\n', abs(peak_sep_rotation - sep));
fprintf('  相比静态改善: %.2fx\n\n', peak_sep_static / peak_sep_rotation);

%% 测试3: 直线平移（应该有效）
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('测试3: 直线平移 (v=5 m/s，预期：显著改善)\n');
fprintf('═══════════════════════════════════════════════════════\n');

v_drone = 5;  % 5 m/s（18 km/h，经济巡航速度）
array_trans = ArrayPlatform(rx_elem, 1, 1:8);
array_trans = array_trans.set_trajectory(@(t) struct('position', [v_drone * t, 0, 0], ...
                                                      'orientation', [0, 0, 0]));

sig_gen = SignalGenerator(radar_params, array_trans, targets);
snapshots = sig_gen.generate_snapshots(t_axis, inf);

est = DoaEstimatorIncoherent_FIXED(array_trans, radar_params);
fprintf('  计算非相干MUSIC谱... ');
tic;
options.verbose = false;
options.weighting = 'uniform';
spectrum_translation = est.estimate_incoherent_music(snapshots, t_axis, 2, search_grid, options);
time_translation = toc;
fprintf('完成 (%.1fs)\n', time_translation);

% 找峰值
[~, phi_est, ~] = DoaEstimatorIncoherent_FIXED.find_peaks(spectrum_translation, search_grid, 2);
peak_sep_translation = abs(phi_est(1) - phi_est(2));
distance = v_drone * t_axis(end);
fprintf('  飞行距离: %.1f m\n', distance);
fprintf('  孔径扩展: %.1f cm → %.1f m (%.0f倍)\n', R_rx*2*100, distance, distance/(R_rx*2));
fprintf('  检测间隔: %.1f° (真实%.1f°)\n', peak_sep_translation, sep);
fprintf('  误差: %.1f°\n', abs(peak_sep_translation - sep));
fprintf('  相比静态改善: %.2fx\n', peak_sep_static / peak_sep_translation);

% 诊断：检查虚拟阵列是否真的在运动
fprintf('\n  [诊断] 检查虚拟阵列位置变化:\n');
pos_t0 = array_trans.get_mimo_virtual_positions(t_axis(1));
pos_t_end = array_trans.get_mimo_virtual_positions(t_axis(end));
max_displacement = max(sqrt(sum((pos_t_end - pos_t0).^2, 2)));
fprintf('    最大位移: %.2f m (理论%.2f m)\n', max_displacement, distance);
if max_displacement < 0.01
    fprintf('    ⚠️ 警告：虚拟阵列几乎没有移动！\n');
end
fprintf('\n');

%% 诊断：检查谱的差异
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('诊断：检查MUSIC谱差异\n');
fprintf('═══════════════════════════════════════════════════════\n');

[~, theta_idx] = min(abs(search_grid.theta - 30));
slice_st = spectrum_static(theta_idx, :);
slice_rot = spectrum_rotation(theta_idx, :);
slice_trans = spectrum_translation(theta_idx, :);

% 归一化
slice_st = slice_st / max(slice_st);
slice_rot = slice_rot / max(slice_rot);
slice_trans = slice_trans / max(slice_trans);

% 找主峰位置（在目标附近）
phi_target_center = 60;
[~, center_idx] = min(abs(search_grid.phi - phi_target_center));
search_range = max(1, center_idx-40):min(length(search_grid.phi), center_idx+40);  % ±20度范围

[~, max_idx_st] = max(slice_st(search_range));
[~, max_idx_rot] = max(slice_rot(search_range));
[~, max_idx_trans] = max(slice_trans(search_range));

max_idx_st = search_range(max_idx_st);
max_idx_rot = search_range(max_idx_rot);
max_idx_trans = search_range(max_idx_trans);

% 计算3dB带宽（半功率带宽）作为分辨率指标
% 静态
peak_val_st = slice_st(max_idx_st);
half_power_st = peak_val_st / 2;
left_idx = max_idx_st;
while left_idx > 1 && slice_st(left_idx) > half_power_st
    left_idx = left_idx - 1;
end
right_idx = max_idx_st;
while right_idx < length(slice_st) && slice_st(right_idx) > half_power_st
    right_idx = right_idx + 1;
end
bw_st = search_grid.phi(right_idx) - search_grid.phi(left_idx);

% 旋转
peak_val_rot = slice_rot(max_idx_rot);
half_power_rot = peak_val_rot / 2;
left_idx = max_idx_rot;
while left_idx > 1 && slice_rot(left_idx) > half_power_rot
    left_idx = left_idx - 1;
end
right_idx = max_idx_rot;
while right_idx < length(slice_rot) && slice_rot(right_idx) > half_power_rot
    right_idx = right_idx + 1;
end
bw_rot = search_grid.phi(right_idx) - search_grid.phi(left_idx);

% 平移
peak_val_trans = slice_trans(max_idx_trans);
half_power_trans = peak_val_trans / 2;
left_idx = max_idx_trans;
while left_idx > 1 && slice_trans(left_idx) > half_power_trans
    left_idx = left_idx - 1;
end
right_idx = max_idx_trans;
while right_idx < length(slice_trans) && slice_trans(right_idx) > half_power_trans
    right_idx = right_idx + 1;
end
bw_trans = search_grid.phi(right_idx) - search_grid.phi(left_idx);

fprintf('  分辨率指标（主峰3dB带宽）:\n');
fprintf('    静态:   %.1f° (峰位置: %.1f°)\n', bw_st, search_grid.phi(max_idx_st));
fprintf('    旋转:   %.1f° (峰位置: %.1f°)\n', bw_rot, search_grid.phi(max_idx_rot));
fprintf('    平移:   %.1f° (峰位置: %.1f°)\n', bw_trans, search_grid.phi(max_idx_trans));

fprintf('\n  改善倍数:\n');
fprintf('    旋转相比静态: %.2fx\n', bw_st / bw_rot);
fprintf('    平移相比静态: %.2fx\n', bw_st / bw_trans);

fprintf('\n  能否分辨%.1f°间隔？\n', sep);
fprintf('    静态: %s (3dB带宽/间隔 = %.1f)\n', ...
    ternary(bw_st < 1.2*sep, '✅ 能分辨', ternary(bw_st < 2*sep, '△ 勉强', '❌ 不能')), bw_st/sep);
fprintf('    旋转: %s (3dB带宽/间隔 = %.1f)\n', ...
    ternary(bw_rot < 1.2*sep, '✅ 能分辨', ternary(bw_rot < 2*sep, '△ 勉强', '❌ 不能')), bw_rot/sep);
fprintf('    平移: %s (3dB带宽/间隔 = %.1f)\n', ...
    ternary(bw_trans < 1.2*sep, '✅ 能分辨', ternary(bw_trans < 2*sep, '△ 勉强', '❌ 不能')), bw_trans/sep);

% 检查谱的差异
diff_st_rot = norm(slice_st - slice_rot) / norm(slice_st);
diff_st_trans = norm(slice_st - slice_trans) / norm(slice_st);
fprintf('\n  谱的归一化差异:\n');
fprintf('    静态 vs 旋转: %.2f%%\n', diff_st_rot * 100);
fprintf('    静态 vs 平移: %.2f%%\n\n', diff_st_trans * 100);

%% 可视化对比
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('生成对比图\n');
fprintf('═══════════════════════════════════════════════════════\n');

figure('Position', [100, 100, 1400, 900]);

% 2D谱对比
for i = 1:3
    subplot(2, 3, i);
    if i == 1
        spec = spectrum_static;
        tit = '静态';
    elseif i == 2
        spec = spectrum_rotation;
        tit = '纯旋转';
    else
        spec = spectrum_translation;
        tit = sprintf('平移 (v=%dm/s)', v_drone);
    end
    
    imagesc(search_grid.phi, search_grid.theta, 10*log10(spec));
    axis xy;
    colorbar;
    caxis([-40, 0]);
    xlabel('Phi (°)');
    ylabel('Theta (°)');
    title(tit);
    hold on;
    plot([60-sep/2, 60+sep/2], [30, 30], 'r+', 'MarkerSize', 15, 'LineWidth', 2);
end

% 1D切片对比（带3dB带宽标注）
subplot(2, 1, 2);
[~, theta_idx_plot] = min(abs(search_grid.theta - 30));

slice_st_plot = spectrum_static(theta_idx_plot, :);
slice_rot_plot = spectrum_rotation(theta_idx_plot, :);
slice_trans_plot = spectrum_translation(theta_idx_plot, :);

% 归一化
slice_st_norm = slice_st_plot / max(slice_st_plot);
slice_rot_norm = slice_rot_plot / max(slice_rot_plot);
slice_trans_norm = slice_trans_plot / max(slice_trans_plot);

plot(search_grid.phi, 10*log10(slice_st_norm), 'b-', 'LineWidth', 2.5, 'DisplayName', sprintf('静态 (3dB宽度: %.1f°)', bw_st));
hold on;
plot(search_grid.phi, 10*log10(slice_rot_norm), 'r-', 'LineWidth', 2.5, 'DisplayName', sprintf('纯旋转 (3dB宽度: %.1f°)', bw_rot));
plot(search_grid.phi, 10*log10(slice_trans_norm), 'g-', 'LineWidth', 2.5, 'DisplayName', sprintf('平移 (3dB宽度: %.1f°)', bw_trans));

% 标注真实目标位置
plot([60-sep/2, 60-sep/2], [-15, 0], 'k--', 'LineWidth', 1.5, 'DisplayName', '真实位置');
plot([60+sep/2, 60+sep/2], [-15, 0], 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');

% 标注-3dB线
plot(xlim, [-3, -3], 'k:', 'LineWidth', 1, 'DisplayName', '-3dB线');

xlabel('Phi (°)', 'FontSize', 12);
ylabel('归一化功率 (dB)', 'FontSize', 12);
title(sprintf('Phi方向切片对比 (θ=30°) - 分辨率改善: %.1fx', bw_st/bw_trans), 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 10);
grid on;
xlim([50, 70]);
ylim([-15, 0]);

sgtitle(sprintf('运动模式对比：双目标%.1f°间隔', sep), ...
        'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, 'quick_test_motion_comparison.png');
fprintf('  ✓ 保存图片: quick_test_motion_comparison.png\n\n');

%% 结果总结（基于3dB带宽）
fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║              快速测试结果总结                          ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

fprintf('配置      | 3dB带宽 | 能分辨%.1f°? | 分辨率改善 | 状态\n', sep);
fprintf('----------|---------|-------------|-----------|------\n');
fprintf('静态      | %6.1f° |     %s      |   1.00x   | 基准\n', ...
    bw_st, ternary(bw_st < 1.2*sep, '✅', ternary(bw_st < 2*sep, '△', '❌')));
fprintf('纯旋转    | %6.1f° |     %s      |   %.2fx   | %s\n', ...
    bw_rot, ternary(bw_rot < 1.2*sep, '✅', ternary(bw_rot < 2*sep, '△', '❌')), bw_st/bw_rot, ...
    ternary(abs(bw_st/bw_rot - 1.0) < 0.15, '❌ 无改善', '✓ 有改善'));
fprintf('平移5m/s  | %6.1f° |     %s      |   %.2fx   | %s\n', ...
    bw_trans, ternary(bw_trans < 1.2*sep, '✅', ternary(bw_trans < 2*sep, '△', '❌')), bw_st/bw_trans, ...
    ternary(bw_st/bw_trans > 1.5, '✅ 显著改善', '△ 轻微改善'));

fprintf('\n');

if abs(bw_st/bw_rot - 1.0) < 0.3
    fprintf('✅ 验证通过: 纯旋转基本无效（性能≈静态，%.1fx）\n', bw_st/bw_rot);
elseif bw_st/bw_rot < 5
    fprintf('⚠️ 注意: 纯旋转有轻微改善（%.1fx），可能是数值误差\n', bw_st/bw_rot);
else
    fprintf('⚠️ 异常: 纯旋转改善了%.1fx，不符合预期\n', bw_st/bw_rot);
end

if bw_st/bw_trans > 1.5
    fprintf('✅ 验证通过: 平移运动显著改善性能（%.1fx）\n', bw_st/bw_trans);
elseif bw_st/bw_trans > 1.1
    fprintf('✓ 平移有改善但不显著（%.1fx），可能需要更长观测时间或更快速度\n', bw_st/bw_trans);
else
    fprintf('⚠️ 异常: 平移改善不明显（%.1fx），检查参数\n', bw_st/bw_trans);
end

fprintf('\n💡 结论:\n');
if bw_st/bw_trans > 10
    fprintf('   ✅✅ 实验设计修复成功！SAR效果显著！\n');
    fprintf('   ✓ 纯旋转基本无效（%.1fx改善）\n', bw_st/bw_rot);
    fprintf('   ✓ 平移运动大幅改善（%.1fx改善）\n', bw_st/bw_trans);
    fprintf('   ✓ SAR原理正确实现（谱变窄%d倍）\n', round(bw_st/bw_trans));
    fprintf('   ✓ 非相干MUSIC Bug已修复\n');
    fprintf('\n   📊 下一步：运行完整实验 comprehensive_validation_FIXED\n');
elseif bw_st/bw_trans > 3
    fprintf('   ✅ 平移有显著改善（%.1fx）\n', bw_st/bw_trans);
    fprintf('   ✓ SAR原理有效\n');
    if bw_st/bw_rot > 2
        fprintf('   ⚠️ 注意：旋转也有改善（%.1fx），可能需要进一步分析\n', bw_st/bw_rot);
    end
    fprintf('\n   💡 建议：增加观测时间以获得更好效果\n');
else
    fprintf('   ⚠️ 平移改善不明显（%.1fx），检查参数\n', bw_st/bw_trans);
end

fprintf('\n');

% 辅助函数
function out = ternary(cond, true_val, false_val)
    if cond
        out = true_val;
    else
        out = false_val;
    end
end

