%% ═══════════════════════════════════════════════════════════════════════════
%  实验：双目标分辨能力测试 v4.0 (1D时间平滑MUSIC版本)
%  验证：运动阵列在分辨相近目标时的优势
%  算法：
%    - 静态阵列：标准MUSIC（多快拍协方差矩阵）
%    - 运动阵列：时间平滑MUSIC（解决合成孔径单快拍秩-1问题）
%  特性：
%    - 使用1D DOA估计（φ方位角），因为双目标只在φ方向分离
%    - 使用8元ULA + y平移，最大化合成孔径优势
%    - y平移方向与φ分辨方向匹配，确保孔径扩展有效
%% ═══════════════════════════════════════════════════════════════════════════
clear; clc; close all;

addpath('asset');

% 创建输出文件夹
script_name = 'experiment_dual_target_1d';
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
output_folder = fullfile('validation_results', [script_name '_' timestamp]);
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

log_file = fullfile(output_folder, 'experiment_log.txt');
diary(log_file);

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║       双目标分辨能力测试 v4.0 (1D时间平滑MUSIC)              ║\n');
fprintf('║  验证：运动阵列分辨相近目标的优势                              ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');
fprintf('输出目录: %s\n\n', output_folder);

%% 参数设置
c = physconst('LightSpeed');
fc = 3e9;
lambda = c / fc;
d = lambda / 2;
radar_params = struct('fc', fc, 'lambda', lambda);

target_range = 500;
snr_db = 10;

% 使用8元ULA（y方向排列）
% y平移会扩展y方向孔径，对sin(φ)方向敏感
% 阵列沿y方向排列，与平移方向一致，合成孔径效果最佳
num_elements = 8;
y_pos = ((0:num_elements-1) - (num_elements-1)/2) * d;
elements = [zeros(num_elements, 1), y_pos', zeros(num_elements, 1)];

v = 5;  % y方向平移速度
T_chirp = 50e-3;  % Chirp周期: 50ms (FMCW标准)
num_snapshots = 16;  % 快拍数
T_obs = num_snapshots * T_chirp;  % 总观测时间: 0.8s
t_axis = (0:num_snapshots-1) * T_chirp;

% 计算孔径
static_aperture = (num_elements - 1) * d;
synthetic_aperture = v * T_obs;
total_aperture = sqrt(static_aperture^2 + synthetic_aperture^2);

% 双目标角度间隔测试
angle_separations = [1, 2, 3, 5, 8, 10, 15, 20];  % 度
phi_center = 60;   % 中心方位角 - sin(60°)≈0.87，对y方向阵列最优
theta_fixed = 90;  % 水平面（θ=90°）- 简化为1D问题

fprintf('【实验设置】\n');
fprintf('  阵列: %d元ULA (y方向排列)\n', num_elements);
fprintf('  运动: y方向平移 v=%.1f m/s (与阵列方向一致)\n', v);
fprintf('  Chirp周期: %.0f ms (FMCW标准)\n', T_chirp*1000);
fprintf('  快拍数: %d, 观测时间: %.1f s\n', num_snapshots, T_obs);
fprintf('  静态孔径: %.2f λ\n', static_aperture / lambda);
fprintf('  合成孔径: %.1f λ (平移%.2fm)\n', synthetic_aperture / lambda, synthetic_aperture);
fprintf('  总孔径: %.1f λ\n', total_aperture / lambda);
fprintf('  目标中心: φ=%.0f°, θ=%.0f° (水平面，1D问题)\n', phi_center, theta_fixed);
fprintf('  测试角度间隔: [%s]°\n', num2str(angle_separations));
fprintf('  SNR: %d dB\n\n', snr_db);

% 理论分辨率
static_resolution = asind(lambda / static_aperture);
synthetic_resolution = asind(lambda / total_aperture);
fprintf('【理论分辨率 (瑞利极限)】\n');
fprintf('  静态: %.1f°\n', static_resolution);
fprintf('  合成: %.2f°\n', synthetic_resolution);
fprintf('  改善: %.1f 倍\n\n', static_resolution / synthetic_resolution);

%% 搜索网格 - 1D (细网格)
phi_search = 30:0.1:90;  % 覆盖φ=60°为中心的范围
search_grid = struct('phi', phi_search);

%% 运行实验
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('开始实验\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

results = struct();
results.separations = angle_separations;
results.static_resolved = zeros(size(angle_separations));
results.motion_resolved = zeros(size(angle_separations));
results.static_peaks = cell(size(angle_separations));
results.motion_peaks = cell(size(angle_separations));
results.static_spectra = cell(size(angle_separations));
results.motion_spectra = cell(size(angle_separations));

fprintf('间隔   | 静态阵列 [峰值φ]     | 运动阵列 [峰值φ]     | 结论\n');
fprintf('-------|----------------------|----------------------|----------\n');

est_options.search_mode = '1d';

for sep_idx = 1:length(angle_separations)
    sep = angle_separations(sep_idx);
    
    % 双目标角度（方位角方向分离）
    phi1 = phi_center - sep/2;
    phi2 = phi_center + sep/2;
    
    % 水平面上的目标位置
    target1_pos = target_range * [cosd(phi1), sind(phi1), 0];
    target2_pos = target_range * [cosd(phi2), sind(phi2), 0];
    
    target1 = Target(target1_pos, [0,0,0], 1);
    target2 = Target(target2_pos, [0,0,0], 1);
    
    % ===== 静态阵列 =====
    array_static = ArrayPlatform(elements, 1, 1:num_elements);
    array_static.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));
    
    sig_gen_static = SignalGeneratorSimple(radar_params, array_static, {target1, target2});
    rng(sep_idx);
    snapshots_static = sig_gen_static.generate_snapshots(t_axis, snr_db);
    
    % 静态MUSIC (1D)
    positions_static = array_static.get_mimo_virtual_positions(0);
    spectrum_static = music_standard_1d(snapshots_static, positions_static, phi_search, lambda, 2);
    
    results.static_spectra{sep_idx} = spectrum_static;
    
    % 找两个最大峰值
    min_sep_search = max(0.5, sep * 0.4);
    [peaks_static_phi] = find_1d_peaks(spectrum_static, phi_search, 2, min_sep_search);
    results.static_peaks{sep_idx} = peaks_static_phi;
    
    % 判断是否分辨
    [static_resolved, ~] = check_resolution(peaks_static_phi, [phi1, phi2], sep);
    results.static_resolved(sep_idx) = static_resolved;
    
    % ===== 运动阵列 (y平移) =====
    array_motion = ArrayPlatform(elements, 1, 1:num_elements);
    array_motion.set_trajectory(@(t) struct('position', [0, v*t, 0], 'orientation', [0,0,0]));
    
    sig_gen_motion = SignalGeneratorSimple(radar_params, array_motion, {target1, target2});
    rng(sep_idx);
    snapshots_motion = sig_gen_motion.generate_snapshots(t_axis, snr_db);
    
    % 运动阵列：时间平滑MUSIC
    estimator_motion = DoaEstimatorSynthetic(array_motion, radar_params);
    [spectrum_motion, ~, ~] = estimator_motion.estimate(snapshots_motion, t_axis, search_grid, 2, est_options);
    
    results.motion_spectra{sep_idx} = spectrum_motion;
    
    % 手动从谱中检测峰值（更准确）
    peaks_motion_phi = find_1d_peaks(spectrum_motion, phi_search, 2, min_sep_search);
    results.motion_peaks{sep_idx} = peaks_motion_phi;
    
    [motion_resolved, ~] = check_resolution(peaks_motion_phi, [phi1, phi2], sep);
    results.motion_resolved(sep_idx) = motion_resolved;
    
    % 输出详细信息
    static_str = ternary(static_resolved, '✓', '✗');
    motion_str = ternary(motion_resolved, '✓', '✗');
    
    if motion_resolved && ~static_resolved
        conclusion = '★运动优势';
    elseif motion_resolved && static_resolved
        conclusion = '均可分辨';
    elseif ~motion_resolved && ~static_resolved
        conclusion = '均不可分辨';
    else
        conclusion = '静态优势';
    end
    
    % 输出峰值位置
    if length(peaks_static_phi) >= 2
        static_peaks_str = sprintf('[%.1f,%.1f]', peaks_static_phi(1), peaks_static_phi(2));
    else
        static_peaks_str = sprintf('[%.1f,-]', peaks_static_phi(1));
    end
    
    if length(peaks_motion_phi) >= 2
        motion_peaks_str = sprintf('[%.1f,%.1f]', peaks_motion_phi(1), peaks_motion_phi(2));
    else
        motion_peaks_str = sprintf('[%.1f,-]', peaks_motion_phi(1));
    end
    
    fprintf('  %2d°  | %s %s | %s %s | %s\n', sep, static_str, static_peaks_str, motion_str, motion_peaks_str, conclusion);
end

%% 绘图
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('生成结果图表\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% 图1: 分辨能力对比柱状图
fig1 = figure('Position', [100, 100, 900, 400], 'Color', 'white');
set(gcf, 'DefaultAxesFontName', 'SimHei');

subplot(1, 2, 1);
bar([results.static_resolved; results.motion_resolved]', 'grouped');
set(gca, 'XTick', 1:length(angle_separations), 'XTickLabel', arrayfun(@(x) sprintf('%d°', x), angle_separations, 'UniformOutput', false));
xlabel('双目标角度间隔', 'FontWeight', 'bold');
ylabel('分辨成功 (1=是, 0=否)', 'FontWeight', 'bold');
title('(a) 分辨能力对比', 'FontSize', 12, 'FontWeight', 'bold');
legend({'静态阵列', '运动阵列'}, 'Location', 'southeast');
grid on;
ylim([0, 1.2]);

% 添加理论分辨率线
hold on;
theoretical_static = find(angle_separations >= static_resolution, 1);
theoretical_motion = find(angle_separations >= synthetic_resolution, 1);
if ~isempty(theoretical_static)
    plot([theoretical_static-0.5, theoretical_static-0.5], [0, 1.2], 'k--', 'LineWidth', 1.5);
    text(theoretical_static-0.3, 1.1, sprintf('静态理论\n%.1f°', static_resolution), 'FontSize', 8);
end
hold off;

subplot(1, 2, 2);
% 找到最小可分辨间隔
static_min = min(angle_separations(results.static_resolved == 1));
motion_min = min(angle_separations(results.motion_resolved == 1));
if isempty(static_min), static_min = max(angle_separations); end
if isempty(motion_min), motion_min = max(angle_separations); end

bar_data = [static_min, motion_min];
b = bar(1:2, bar_data, 0.5);
b.FaceColor = 'flat';
b.CData(1,:) = [0.3, 0.3, 0.3];
b.CData(2,:) = [0.0, 0.45, 0.74];
set(gca, 'XTick', 1:2, 'XTickLabel', {'静态阵列', '运动阵列'});
ylabel('最小可分辨角度 (°)', 'FontWeight', 'bold');
title('(b) 分辨率对比', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% 添加数值标签
text(1, bar_data(1)+0.5, sprintf('%.0f°', bar_data(1)), 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
text(2, bar_data(2)+0.5, sprintf('%.0f°', bar_data(2)), 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');

% 改善倍数
if motion_min > 0
    improvement = static_min / motion_min;
    text(1.5, max(bar_data)*0.6, sprintf('分辨率提升\n%.1f倍', improvement), ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', 'Color', [0 0.5 0]);
end

sgtitle(sprintf('双目标分辨能力测试 (8元ULA + y平移, SNR=%ddB)', snr_db), 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig1, fullfile(output_folder, 'fig1_分辨能力对比.png'));
saveas(fig1, fullfile(output_folder, 'fig1_分辨能力对比.eps'), 'epsc');

%% 图2: MUSIC谱对比（选择典型间隔）
% 选择一个能体现运动优势的间隔
typical_sep_idx = 0;
for i = 1:length(angle_separations)
    if results.motion_resolved(i) && ~results.static_resolved(i)
        typical_sep_idx = i;
        break;
    end
end
if typical_sep_idx == 0
    % 如果没有找到运动优势的，选择5°或最接近的
    typical_sep_idx = find(angle_separations == 5, 1);
    if isempty(typical_sep_idx), typical_sep_idx = 4; end
end

fig2 = figure('Position', [100, 100, 1000, 400], 'Color', 'white');
set(gcf, 'DefaultAxesFontName', 'SimHei');

sep = angle_separations(typical_sep_idx);
phi1 = phi_center - sep/2;
phi2 = phi_center + sep/2;

subplot(1, 2, 1);
spectrum_db = 10*log10(results.static_spectra{typical_sep_idx} / max(results.static_spectra{typical_sep_idx}));
plot(phi_search, spectrum_db, 'b-', 'LineWidth', 2);
hold on;
xline(phi1, 'r--', 'LineWidth', 2);
xline(phi2, 'r--', 'LineWidth', 2);
hold off;
xlabel('方位角 φ (°)', 'FontWeight', 'bold');
ylabel('归一化功率 (dB)', 'FontWeight', 'bold');
title(sprintf('(a) 静态阵列 (孔径=%.1fλ)', static_aperture/lambda), 'FontWeight', 'bold');
xlim([phi_center-25, phi_center+25]);
ylim([-40, 5]);
grid on;
legend({'MUSIC谱', '真实目标'}, 'Location', 'south');

% 标注分辨结果
if results.static_resolved(typical_sep_idx)
    text(phi_center, -35, '✓ 可分辨', 'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'g', 'FontWeight', 'bold');
else
    text(phi_center, -35, '✗ 不可分辨', 'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'r', 'FontWeight', 'bold');
end

subplot(1, 2, 2);
spectrum_db = 10*log10(results.motion_spectra{typical_sep_idx} / max(results.motion_spectra{typical_sep_idx}));
plot(phi_search, spectrum_db, 'b-', 'LineWidth', 2);
hold on;
xline(phi1, 'r--', 'LineWidth', 2);
xline(phi2, 'r--', 'LineWidth', 2);
hold off;
xlabel('方位角 φ (°)', 'FontWeight', 'bold');
ylabel('归一化功率 (dB)', 'FontWeight', 'bold');
title(sprintf('(b) 运动阵列 (合成孔径=%.1fλ)', total_aperture/lambda), 'FontWeight', 'bold');
xlim([phi_center-25, phi_center+25]);
ylim([-40, 5]);
grid on;
legend({'时间平滑MUSIC谱', '真实目标'}, 'Location', 'south');

% 标注分辨结果
if results.motion_resolved(typical_sep_idx)
    text(phi_center, -35, '✓ 可分辨', 'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'g', 'FontWeight', 'bold');
else
    text(phi_center, -35, '✗ 不可分辨', 'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'r', 'FontWeight', 'bold');
end

sgtitle(sprintf('1D MUSIC谱对比 (间隔=%d°)', sep), 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig2, fullfile(output_folder, 'fig2_MUSIC谱对比.png'));
saveas(fig2, fullfile(output_folder, 'fig2_MUSIC谱对比.eps'), 'epsc');

%% 图3: 多间隔MUSIC谱对比
fig3 = figure('Position', [100, 100, 1200, 600], 'Color', 'white');
set(gcf, 'DefaultAxesFontName', 'SimHei');

% 选择4个代表性间隔
selected_seps = [2, 5, 10, 15];
selected_idx = [];
for s = selected_seps
    idx = find(angle_separations == s, 1);
    if ~isempty(idx)
        selected_idx = [selected_idx, idx];
    end
end

for i = 1:length(selected_idx)
    idx = selected_idx(i);
    sep = angle_separations(idx);
    phi1 = phi_center - sep/2;
    phi2 = phi_center + sep/2;
    
    % 静态
    subplot(2, 4, i);
    spectrum_db = 10*log10(results.static_spectra{idx} / max(results.static_spectra{idx}));
    plot(phi_search, spectrum_db, 'k-', 'LineWidth', 1.5);
    hold on;
    xline(phi1, 'r--', 'LineWidth', 1.5);
    xline(phi2, 'r--', 'LineWidth', 1.5);
    hold off;
    xlim([max(30, phi_center-25), min(90, phi_center+25)]);
    ylim([-30, 5]);
    grid on;
    if results.static_resolved(idx)
        title(sprintf('静态 %d° ✓', sep), 'Color', 'g', 'FontWeight', 'bold');
    else
        title(sprintf('静态 %d° ✗', sep), 'Color', 'r', 'FontWeight', 'bold');
    end
    if i == 1
        ylabel('归一化功率 (dB)', 'FontWeight', 'bold');
    end
    
    % 运动
    subplot(2, 4, i + 4);
    spectrum_db = 10*log10(results.motion_spectra{idx} / max(results.motion_spectra{idx}));
    plot(phi_search, spectrum_db, 'b-', 'LineWidth', 1.5);
    hold on;
    xline(phi1, 'r--', 'LineWidth', 1.5);
    xline(phi2, 'r--', 'LineWidth', 1.5);
    hold off;
    xlim([max(30, phi_center-25), min(90, phi_center+25)]);
    ylim([-30, 5]);
    grid on;
    xlabel('φ (°)', 'FontWeight', 'bold');
    if results.motion_resolved(idx)
        title(sprintf('运动 %d° ✓', sep), 'Color', 'g', 'FontWeight', 'bold');
    else
        title(sprintf('运动 %d° ✗', sep), 'Color', 'r', 'FontWeight', 'bold');
    end
    if i == 1
        ylabel('归一化功率 (dB)', 'FontWeight', 'bold');
    end
end

sgtitle('不同间隔下的MUSIC谱对比 (上:静态, 下:运动)', 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig3, fullfile(output_folder, 'fig3_多间隔对比.png'));
saveas(fig3, fullfile(output_folder, 'fig3_多间隔对比.eps'), 'epsc');

%% 统计
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                        实验结论                                   \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

fprintf('【阵列配置】\n');
fprintf('  静态孔径: %.1f λ\n', static_aperture / lambda);
fprintf('  合成孔径: %.1f λ\n', total_aperture / lambda);
fprintf('  孔径扩展: %.1f 倍\n\n', total_aperture / static_aperture);

fprintf('【理论分辨率】\n');
fprintf('  静态: %.1f°\n', static_resolution);
fprintf('  合成: %.2f°\n\n', synthetic_resolution);

fprintf('【实测分辨率】\n');
fprintf('  静态阵列最小可分辨间隔: %d°\n', static_min);
fprintf('  运动阵列最小可分辨间隔: %d°\n', motion_min);
if motion_min > 0
    fprintf('  分辨率改善: %.1f 倍\n\n', static_min / motion_min);
end

fprintf('【核心结论】\n');
if motion_min < static_min
    fprintf('  ✅ 运动阵列通过时间平滑MUSIC，显著提升角度分辨率\n');
    fprintf('  ✅ 可分辨更近的双目标 (%d° vs %d°)\n', motion_min, static_min);
else
    fprintf('  ⚠️ 运动阵列未显示出分辨率优势，需要检查参数\n');
end

%% 保存
save(fullfile(output_folder, 'experiment_results.mat'), 'results', 'static_aperture', 'synthetic_aperture', 'total_aperture');
fprintf('\n实验完成！结果保存在: %s\n', output_folder);
diary off;

%% ═══════════════════════════════════════════════════════════════════════════
%  辅助函数
%% ═══════════════════════════════════════════════════════════════════════════

function spectrum = music_standard_1d(snapshots, positions, phi_search, lambda, num_targets)
    num_elements = size(snapshots, 1);
    num_snapshots = size(snapshots, 2);
    
    Rxx = (snapshots * snapshots') / num_snapshots;
    [V, D] = eig(Rxx);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    Qn = V(:, (num_targets+1):end);
    
    spectrum = zeros(size(phi_search));
    
    for phi_idx = 1:length(phi_search)
        phi = phi_search(phi_idx);
        % 水平面上的方向矢量
        u = [cosd(phi); sind(phi); 0];
        
        a = zeros(num_elements, 1);
        for i = 1:num_elements
            phase = 4 * pi / lambda * (positions(i, :) * u);
            a(i) = exp(-1j * phase);
        end
        
        spectrum(phi_idx) = 1 / abs(a' * (Qn * Qn') * a);
    end
end

function peaks_phi = find_1d_peaks(spectrum, phi_search, num_peaks, min_separation)
    % 使用MATLAB的findpeaks函数进行更鲁棒的峰值检测
    if nargin < 4
        min_separation = 0.5;
    end
    
    % 计算最小峰值间隔对应的样本数
    dphi = phi_search(2) - phi_search(1);
    min_samples = max(1, floor(min_separation / dphi));
    
    % 使用findpeaks检测峰值
    [pks, locs] = findpeaks(spectrum, 'MinPeakDistance', min_samples, 'SortStr', 'descend');
    
    % 取前num_peaks个峰值
    if length(locs) >= num_peaks
        peaks_phi = phi_search(locs(1:num_peaks));
    elseif length(locs) > 0
        % 如果找到的峰值数量不足，用找到的峰值填充
        peaks_phi = zeros(1, num_peaks);
        peaks_phi(1:length(locs)) = phi_search(locs);
        % 剩余位置用最大值位置填充
        [~, max_idx] = max(spectrum);
        peaks_phi(length(locs)+1:end) = phi_search(max_idx);
    else
        % 如果没找到峰值，用最大值位置
        [~, sorted_idx] = sort(spectrum, 'descend');
        peaks_phi = phi_search(sorted_idx(1:min(num_peaks, length(sorted_idx))));
    end
    
    % 确保输出长度正确
    if length(peaks_phi) < num_peaks
        peaks_phi(end+1:num_peaks) = peaks_phi(end);
    end
end

function [resolved, details] = check_resolution(estimated_peaks, true_angles, sep)
    % 分辨判断逻辑
    % 核心标准：两个峰值是否分别接近两个真实目标位置
    
    details = struct();
    details.peaks = estimated_peaks;
    details.true_angles = true_angles;
    
    if length(estimated_peaks) < 2
        resolved = false;
        details.reason = '峰值数量不足';
        return;
    end
    
    % 估计的峰值间隔
    est_peaks_sorted = sort(estimated_peaks);
    est_separation = abs(est_peaks_sorted(2) - est_peaks_sorted(1));
    details.est_separation = est_separation;
    
    % 分辨标准1: 检查估计的峰值间隔是否显著（大于间隔的50%）
    sep_significant = est_separation > sep * 0.5;
    
    % 分辨标准2: 两个峰值是否分别在两个目标附近
    % 容差策略：
    %   - 小间隔（<=5°）：允许更大相对误差，容差 = sep*0.4 + 0.5
    %   - 大间隔（>5°）：固定小容差 = 2.0°，因为大间隔更容易分辨，要求更准确
    if sep <= 5
        tolerance = sep * 0.4 + 0.5;  % 2°间隔时容差=1.3°, 5°间隔时容差=2.5°
    else
        tolerance = 2.0;  % 大间隔时固定2°容差
    end
    
    true_sorted = sort(true_angles);
    est_sorted = sort(estimated_peaks);
    
    % 检查第一个估计峰是否接近第一个目标，第二个估计峰是否接近第二个目标
    error1 = abs(est_sorted(1) - true_sorted(1));
    error2 = abs(est_sorted(2) - true_sorted(2));
    
    match1 = error1 < tolerance;
    match2 = error2 < tolerance;
    
    details.match1 = match1;
    details.match2 = match2;
    details.sep_significant = sep_significant;
    details.error1 = error1;
    details.error2 = error2;
    details.tolerance = tolerance;
    
    % 只要峰值间隔显著，且两个峰值都在各自目标附近，就算分辨成功
    resolved = sep_significant && match1 && match2;
    
    if resolved
        details.reason = '分辨成功';
    else
        if ~sep_significant
            details.reason = sprintf('峰值间隔不显著(%.1f°<%.1f°)', est_separation, sep*0.5);
        elseif ~match1
            details.reason = sprintf('第一峰偏差过大(%.1f°>%.1f°)', error1, tolerance);
        elseif ~match2
            details.reason = sprintf('第二峰偏差过大(%.1f°>%.1f°)', error2, tolerance);
        end
    end
end

function out = ternary(cond, true_val, false_val)
    if cond
        out = true_val;
    else
        out = false_val;
    end
end
