%% ═══════════════════════════════════════════════════════════════════════════
%  实验：双目标分辨能力测试 v5.0 (蒙特卡洛版本)
%  验证：运动阵列在分辨相近目标时的优势
%  算法：
%    - 静态阵列：标准MUSIC（多快拍协方差矩阵）
%    - 运动阵列：时间平滑MUSIC（解决合成孔径单快拍秩-1问题）
%  改进：
%    - 每个角度间隔进行多次蒙特卡洛测试
%    - 统计分辨成功率，而非单次二值结果
%    - 更有统计意义，结果更可靠
%% ═══════════════════════════════════════════════════════════════════════════
clear; clc; close all;

addpath('asset');

% 创建输出文件夹
script_name = 'experiment_dual_target_mc';
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
output_folder = fullfile('validation_results', [script_name '_' timestamp]);
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

log_file = fullfile(output_folder, 'experiment_log.txt');
diary(log_file);

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║     双目标分辨能力测试 v5.0 (蒙特卡洛统计版本)              ║\n');
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
% 注意：静态8元ULA孔径=3.5λ，理论分辨率≈16.6°
% 测试范围应覆盖静态能分辨和不能分辨的区间
angle_separations = [3, 5, 8, 10, 15, 20, 25, 30];  % 度
phi_center = 60;   % 中心方位角 - sin(60°)≈0.87，对y方向阵列最优
theta_fixed = 90;  % 水平面（θ=90°）- 简化为1D问题

% 蒙特卡洛参数
num_trials = 20;   % 每个间隔测试次数 (减少以加快速度)
snr_values = [5, 10, 15];  % 测试多个SNR

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
fprintf('  蒙特卡洛次数: %d 次/间隔\n', num_trials);
fprintf('  测试SNR: [%s] dB\n\n', num2str(snr_values));

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

%% 运行蒙特卡洛实验
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('开始蒙特卡洛实验 (共 %d × %d × %d = %d 次测试)\n', ...
    length(angle_separations), length(snr_values), num_trials, ...
    length(angle_separations) * length(snr_values) * num_trials);
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

results = struct();
results.separations = angle_separations;
results.snr_values = snr_values;
results.num_trials = num_trials;

% 分辨成功率矩阵 [间隔 × SNR]
results.static_success_rate = zeros(length(angle_separations), length(snr_values));
results.motion_success_rate = zeros(length(angle_separations), length(snr_values));

% 保存一组典型的谱（用于绘图，使用中间SNR）
typical_snr_idx = ceil(length(snr_values) / 2);
results.static_spectra = cell(size(angle_separations));
results.motion_spectra = cell(size(angle_separations));

est_options.search_mode = '1d';

for snr_idx = 1:length(snr_values)
    snr_db = snr_values(snr_idx);
    fprintf('【SNR = %d dB】\n', snr_db);
    fprintf('间隔   | 静态成功率 | 运动成功率 | 差异\n');
    fprintf('-------|------------|------------|--------\n');
    
    for sep_idx = 1:length(angle_separations)
        sep = angle_separations(sep_idx);
        
        % 双目标角度（方位角方向分离）
        phi1 = phi_center - sep/2;
        phi2 = phi_center + sep/2;
        
        static_success_count = 0;
        motion_success_count = 0;
        
        for trial = 1:num_trials
            % 随机种子
            rng(sep_idx * 1000 + snr_idx * 100 + trial);
            
            % 水平面上的目标位置
            target1_pos = target_range * [cosd(phi1), sind(phi1), 0];
            target2_pos = target_range * [cosd(phi2), sind(phi2), 0];
            
            target1 = Target(target1_pos, [0,0,0], 1);
            target2 = Target(target2_pos, [0,0,0], 1);
            
            % ===== 静态阵列 =====
            % 信号生成器已包含目标波动（每快拍独立幅度）
            % 这模拟真实环境，静态和运动阵列使用相同信号模型
            array_static = ArrayPlatform(elements, 1, 1:num_elements);
            array_static.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));
            
            sig_gen_static = SignalGeneratorSimple(radar_params, array_static, {target1, target2});
            snapshots_static = sig_gen_static.generate_snapshots(t_axis, snr_db);
            
            % 静态MUSIC (1D)
            positions_static = array_static.get_mimo_virtual_positions(0);
            spectrum_static = music_standard_1d(snapshots_static, positions_static, phi_search, lambda, 2);
            
            % 保存典型谱（第一次试验）
            if trial == 1 && snr_idx == typical_snr_idx
                results.static_spectra{sep_idx} = spectrum_static;
            end
            
            % 找两个最大峰值
            min_sep_search = max(0.5, sep * 0.4);
            peaks_static_phi = find_1d_peaks(spectrum_static, phi_search, 2, min_sep_search);
            
            % 判断是否分辨
            [static_resolved, ~] = check_resolution(peaks_static_phi, [phi1, phi2], sep);
            if static_resolved
                static_success_count = static_success_count + 1;
            end
            
            % ===== 运动阵列 (y平移) =====
            array_motion = ArrayPlatform(elements, 1, 1:num_elements);
            array_motion.set_trajectory(@(t) struct('position', [0, v*t, 0], 'orientation', [0,0,0]));
            
            sig_gen_motion = SignalGeneratorSimple(radar_params, array_motion, {target1, target2});
            snapshots_motion = sig_gen_motion.generate_snapshots(t_axis, snr_db);
            
            % 运动阵列：时间平滑MUSIC
            estimator_motion = DoaEstimatorSynthetic(array_motion, radar_params);
            [spectrum_motion, ~, ~] = estimator_motion.estimate(snapshots_motion, t_axis, search_grid, 2, est_options);
            
            % 保存典型谱
            if trial == 1 && snr_idx == typical_snr_idx
                results.motion_spectra{sep_idx} = spectrum_motion;
            end
            
            % 峰值检测
            peaks_motion_phi = find_1d_peaks(spectrum_motion, phi_search, 2, min_sep_search);
            
            [motion_resolved, ~] = check_resolution(peaks_motion_phi, [phi1, phi2], sep);
            if motion_resolved
                motion_success_count = motion_success_count + 1;
            end
        end
        
        % 计算成功率
        static_rate = static_success_count / num_trials * 100;
        motion_rate = motion_success_count / num_trials * 100;
        
        results.static_success_rate(sep_idx, snr_idx) = static_rate;
        results.motion_success_rate(sep_idx, snr_idx) = motion_rate;
        
        % 输出
        diff_str = '';
        if motion_rate > static_rate + 10
            diff_str = sprintf('+%.0f%%', motion_rate - static_rate);
        elseif static_rate > motion_rate + 10
            diff_str = sprintf('-%.0f%%', static_rate - motion_rate);
        else
            diff_str = '≈';
        end
        
        fprintf('  %2d°  |   %5.1f%%   |   %5.1f%%   | %s\n', sep, static_rate, motion_rate, diff_str);
    end
    fprintf('\n');
end

%% 绘图
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('生成结果图表\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% 定义颜色和标记
color_static = [0.3, 0.3, 0.3];  % 深灰
color_motion = [0.0, 0.45, 0.74];  % 蓝色

% 图1: 分辨成功率对比柱状图（选择中间SNR）
fig1 = figure('Position', [100, 100, 900, 400], 'Color', 'white');
set(gcf, 'DefaultAxesFontName', 'SimHei');

subplot(1, 2, 1);
bar_data = [results.static_success_rate(:, typical_snr_idx), results.motion_success_rate(:, typical_snr_idx)];
b = bar(bar_data, 'grouped');
b(1).FaceColor = color_static;
b(1).EdgeColor = 'k';
b(1).LineWidth = 1.2;
b(2).FaceColor = color_motion;
b(2).EdgeColor = 'k';
b(2).LineWidth = 1.2;

set(gca, 'XTick', 1:length(angle_separations), 'XTickLabel', arrayfun(@(x) sprintf('%d°', x), angle_separations, 'UniformOutput', false));
xlabel('双目标角度间隔', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('分辨成功率 (%)', 'FontWeight', 'bold', 'FontSize', 11);
title(sprintf('(a) 分辨成功率对比 (SNR=%ddB, N=%d)', snr_values(typical_snr_idx), num_trials), 'FontSize', 12, 'FontWeight', 'bold');
legend({'静态阵列', '运动阵列'}, 'Location', 'southeast', 'FontSize', 10);
grid on;
ylim([0, 110]);

% 添加理论分辨率线
hold on;
theoretical_static = find(angle_separations >= static_resolution, 1);
if ~isempty(theoretical_static)
    xline(theoretical_static - 0.5, 'k--', 'LineWidth', 1.5);
    text(theoretical_static - 0.3, 105, sprintf('静态理论\n%.1f°', static_resolution), 'FontSize', 9);
end
hold off;

subplot(1, 2, 2);
% 找到90%成功率对应的最小间隔
threshold = 90;  % 90%成功率作为"可分辨"标准
static_min = find_min_resolvable_angle(angle_separations, results.static_success_rate(:, typical_snr_idx), threshold);
motion_min = find_min_resolvable_angle(angle_separations, results.motion_success_rate(:, typical_snr_idx), threshold);

bar_data = [static_min, motion_min];
b = bar(1:2, bar_data, 0.5);
b.FaceColor = 'flat';
b.CData(1,:) = color_static;
b.CData(2,:) = color_motion;
b.EdgeColor = 'k';
b.LineWidth = 1.2;
set(gca, 'XTick', 1:2, 'XTickLabel', {'静态阵列', '运动阵列'});
ylabel('最小可分辨角度 (°)', 'FontWeight', 'bold', 'FontSize', 11);
title(sprintf('(b) 最小分辨角度 (成功率≥%d%%)', threshold), 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% 添加数值标签
text(1, bar_data(1)+1, sprintf('%.0f°', bar_data(1)), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
text(2, bar_data(2)+1, sprintf('%.0f°', bar_data(2)), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');

% 改善倍数
if motion_min > 0 && bar_data(1) > bar_data(2)
    improvement = bar_data(1) / bar_data(2);
    text(1.5, max(bar_data)*0.6, sprintf('分辨率提升\n%.1f倍', improvement), ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', 'Color', [0 0.5 0]);
end

sgtitle(sprintf('双目标分辨能力测试 (8元ULA + y平移, %d次蒙特卡洛)', num_trials), 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig1, fullfile(output_folder, 'fig1_分辨成功率对比.png'));
saveas(fig1, fullfile(output_folder, 'fig1_分辨成功率对比.eps'), 'epsc');

%% 图2: 不同SNR下的成功率曲线
fig2 = figure('Position', [100, 100, 600, 450], 'Color', 'white');
set(gcf, 'DefaultAxesFontName', 'SimHei');

% 线型和标记
line_styles_static = {'--', '-.', ':'};
line_styles_motion = {'-', '-', '-'};
markers = {'o', 's', 'd'};

hold on;
legend_entries = {};
for snr_idx = 1:length(snr_values)
    snr = snr_values(snr_idx);
    
    % 静态阵列
    plot(angle_separations, results.static_success_rate(:, snr_idx), ...
        line_styles_static{snr_idx}, 'Color', color_static, 'LineWidth', 1.5, ...
        'Marker', markers{snr_idx}, 'MarkerSize', 7, 'MarkerFaceColor', 'w');
    legend_entries{end+1} = sprintf('静态 SNR=%ddB', snr);
    
    % 运动阵列
    plot(angle_separations, results.motion_success_rate(:, snr_idx), ...
        line_styles_motion{snr_idx}, 'Color', color_motion, 'LineWidth', 2, ...
        'Marker', markers{snr_idx}, 'MarkerSize', 8, 'MarkerFaceColor', color_motion);
    legend_entries{end+1} = sprintf('运动 SNR=%ddB', snr);
end

% 90%成功率参考线
yline(90, 'k--', 'LineWidth', 1, 'Alpha', 0.5);
text(max(angle_separations)-2, 92, '90%阈值', 'FontSize', 9, 'Color', [0.5,0.5,0.5]);

hold off;

xlabel('双目标角度间隔 (°)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('分辨成功率 (%)', 'FontWeight', 'bold', 'FontSize', 12);
title('不同SNR下的分辨成功率', 'FontSize', 14, 'FontWeight', 'bold');
legend(legend_entries, 'Location', 'southeast', 'FontSize', 9, 'NumColumns', 2);
grid on;
xlim([min(angle_separations)-1, max(angle_separations)+1]);
ylim([0, 105]);

saveas(fig2, fullfile(output_folder, 'fig2_SNR对比.png'));
saveas(fig2, fullfile(output_folder, 'fig2_SNR对比.eps'), 'epsc');

%% 图3: MUSIC谱对比（选择典型间隔）
% 选择一个能体现运动优势的间隔（成功率差异最大）
rate_diff = results.motion_success_rate(:, typical_snr_idx) - results.static_success_rate(:, typical_snr_idx);
[~, typical_sep_idx] = max(rate_diff);
if rate_diff(typical_sep_idx) < 10
    % 如果差异不明显，选择5°或最接近的
    typical_sep_idx = find(angle_separations == 5, 1);
    if isempty(typical_sep_idx), typical_sep_idx = 2; end
end

fig3 = figure('Position', [100, 100, 1000, 400], 'Color', 'white');
set(gcf, 'DefaultAxesFontName', 'SimHei');

sep = angle_separations(typical_sep_idx);
phi1 = phi_center - sep/2;
phi2 = phi_center + sep/2;

subplot(1, 2, 1);
if ~isempty(results.static_spectra{typical_sep_idx})
    spectrum_db = 10*log10(results.static_spectra{typical_sep_idx} / max(results.static_spectra{typical_sep_idx}));
    plot(phi_search, spectrum_db, 'k-', 'LineWidth', 2);
    hold on;
    xline(phi1, 'r--', 'LineWidth', 2);
    xline(phi2, 'r--', 'LineWidth', 2);
    hold off;
end
xlabel('方位角 φ (°)', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('归一化功率 (dB)', 'FontWeight', 'bold', 'FontSize', 11);
title(sprintf('(a) 静态阵列 (成功率=%.0f%%)', results.static_success_rate(typical_sep_idx, typical_snr_idx)), ...
    'FontWeight', 'bold', 'FontSize', 12);
xlim([phi_center-25, phi_center+25]);
ylim([-40, 5]);
grid on;
legend({'MUSIC谱', '真实目标'}, 'Location', 'south', 'FontSize', 10);

subplot(1, 2, 2);
if ~isempty(results.motion_spectra{typical_sep_idx})
    spectrum_db = 10*log10(results.motion_spectra{typical_sep_idx} / max(results.motion_spectra{typical_sep_idx}));
    plot(phi_search, spectrum_db, 'b-', 'LineWidth', 2);
    hold on;
    xline(phi1, 'r--', 'LineWidth', 2);
    xline(phi2, 'r--', 'LineWidth', 2);
    hold off;
end
xlabel('方位角 φ (°)', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('归一化功率 (dB)', 'FontWeight', 'bold', 'FontSize', 11);
title(sprintf('(b) 运动阵列 (成功率=%.0f%%)', results.motion_success_rate(typical_sep_idx, typical_snr_idx)), ...
    'FontWeight', 'bold', 'FontSize', 12);
xlim([phi_center-25, phi_center+25]);
ylim([-40, 5]);
grid on;
legend({'时间平滑MUSIC谱', '真实目标'}, 'Location', 'south', 'FontSize', 10);

sgtitle(sprintf('MUSIC谱对比 (间隔=%d°, SNR=%ddB)', sep, snr_values(typical_snr_idx)), 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig3, fullfile(output_folder, 'fig3_MUSIC谱对比.png'));
saveas(fig3, fullfile(output_folder, 'fig3_MUSIC谱对比.eps'), 'epsc');

%% 图4: 多间隔MUSIC谱对比
fig4 = figure('Position', [100, 100, 1200, 500], 'Color', 'white');
set(gcf, 'DefaultAxesFontName', 'SimHei');

% 选择4个代表性间隔
selected_seps = [3, 5, 10, 15];
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
    
    static_rate = results.static_success_rate(idx, typical_snr_idx);
    motion_rate = results.motion_success_rate(idx, typical_snr_idx);
    
    % 静态
    subplot(2, length(selected_idx), i);
    if ~isempty(results.static_spectra{idx})
        spectrum_db = 10*log10(results.static_spectra{idx} / max(results.static_spectra{idx}));
        plot(phi_search, spectrum_db, 'k-', 'LineWidth', 1.5);
        hold on;
        xline(phi1, 'r--', 'LineWidth', 1.5);
        xline(phi2, 'r--', 'LineWidth', 1.5);
        hold off;
    end
    xlim([max(30, phi_center-25), min(90, phi_center+25)]);
    ylim([-30, 5]);
    grid on;
    if static_rate >= 90
        title_color = [0, 0.6, 0];  % 绿色
    elseif static_rate >= 50
        title_color = [0.8, 0.5, 0];  % 橙色
    else
        title_color = [0.8, 0, 0];  % 红色
    end
    title(sprintf('静态 %d° (%.0f%%)', sep, static_rate), 'Color', title_color, 'FontWeight', 'bold', 'FontSize', 11);
    if i == 1
        ylabel('归一化功率 (dB)', 'FontWeight', 'bold');
    end
    
    % 运动
    subplot(2, length(selected_idx), i + length(selected_idx));
    if ~isempty(results.motion_spectra{idx})
        spectrum_db = 10*log10(results.motion_spectra{idx} / max(results.motion_spectra{idx}));
        plot(phi_search, spectrum_db, 'b-', 'LineWidth', 1.5);
        hold on;
        xline(phi1, 'r--', 'LineWidth', 1.5);
        xline(phi2, 'r--', 'LineWidth', 1.5);
        hold off;
    end
    xlim([max(30, phi_center-25), min(90, phi_center+25)]);
    ylim([-30, 5]);
    grid on;
    xlabel('φ (°)', 'FontWeight', 'bold');
    if motion_rate >= 90
        title_color = [0, 0.6, 0];
    elseif motion_rate >= 50
        title_color = [0.8, 0.5, 0];
    else
        title_color = [0.8, 0, 0];
    end
    title(sprintf('运动 %d° (%.0f%%)', sep, motion_rate), 'Color', title_color, 'FontWeight', 'bold', 'FontSize', 11);
    if i == 1
        ylabel('归一化功率 (dB)', 'FontWeight', 'bold');
    end
end

sgtitle(sprintf('MUSIC谱对比 (上:静态, 下:运动, SNR=%ddB)', snr_values(typical_snr_idx)), 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig4, fullfile(output_folder, 'fig4_多间隔对比.png'));
saveas(fig4, fullfile(output_folder, 'fig4_多间隔对比.eps'), 'epsc');

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

fprintf('【实测分辨率 (90%%成功率标准)】\n');
fprintf('  SNR    | 静态最小角 | 运动最小角 | 改善倍数\n');
fprintf('  -------|------------|------------|----------\n');
for snr_idx = 1:length(snr_values)
    static_min_snr = find_min_resolvable_angle(angle_separations, results.static_success_rate(:, snr_idx), 90);
    motion_min_snr = find_min_resolvable_angle(angle_separations, results.motion_success_rate(:, snr_idx), 90);
    if motion_min_snr > 0 && static_min_snr > motion_min_snr
        improvement = static_min_snr / motion_min_snr;
        fprintf('  %2ddB   |   %5.1f°   |   %5.1f°   |  %.1fx\n', snr_values(snr_idx), static_min_snr, motion_min_snr, improvement);
    else
        fprintf('  %2ddB   |   %5.1f°   |   %5.1f°   |   -\n', snr_values(snr_idx), static_min_snr, motion_min_snr);
    end
end
fprintf('\n');

fprintf('【成功率汇总 (SNR=%ddB)】\n', snr_values(typical_snr_idx));
fprintf('  间隔   | 静态成功率 | 运动成功率 | 提升\n');
fprintf('  -------|------------|------------|------\n');
for sep_idx = 1:length(angle_separations)
    static_rate = results.static_success_rate(sep_idx, typical_snr_idx);
    motion_rate = results.motion_success_rate(sep_idx, typical_snr_idx);
    diff = motion_rate - static_rate;
    if diff > 0
        diff_str = sprintf('+%.0f%%', diff);
    elseif diff < 0
        diff_str = sprintf('%.0f%%', diff);
    else
        diff_str = '=';
    end
    fprintf('  %3d°   |   %5.1f%%   |   %5.1f%%   | %s\n', angle_separations(sep_idx), static_rate, motion_rate, diff_str);
end
fprintf('\n');

fprintf('【核心结论】\n');
% 计算中间SNR的最小可分辨角度
static_min_typical = find_min_resolvable_angle(angle_separations, results.static_success_rate(:, typical_snr_idx), 90);
motion_min_typical = find_min_resolvable_angle(angle_separations, results.motion_success_rate(:, typical_snr_idx), 90);

if motion_min_typical < static_min_typical
    fprintf('  ✅ 运动阵列通过时间平滑MUSIC，显著提升角度分辨率\n');
    fprintf('  ✅ 90%%成功率最小角: 运动 %.0f° vs 静态 %.0f° (提升%.1f倍)\n', motion_min_typical, static_min_typical, static_min_typical/motion_min_typical);
else
    fprintf('  ⚠️ 运动阵列未显示出分辨率优势，需要检查参数\n');
end

% 计算平均成功率提升
avg_improvement = mean(results.motion_success_rate(:, typical_snr_idx) - results.static_success_rate(:, typical_snr_idx));
fprintf('  📊 平均成功率提升: %.1f%%\n', avg_improvement);

%% 保存
results.static_aperture = static_aperture;
results.synthetic_aperture = synthetic_aperture;
results.total_aperture = total_aperture;
results.phi_center = phi_center;
results.phi_search = phi_search;
results.lambda = lambda;
results.static_resolution = static_resolution;
results.synthetic_resolution = synthetic_resolution;

save(fullfile(output_folder, 'experiment_results.mat'), 'results');
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
    % 容差策略：容差 = 间隔的30% + 1°，但最小2°，最大5°
    tolerance = min(5, max(2, sep * 0.3 + 1));
    
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

function min_angle = find_min_resolvable_angle(angles, success_rates, threshold)
    % 找到成功率达到阈值的最小角度间隔
    % 使用插值来获得更精确的值
    
    above_threshold = success_rates >= threshold;
    if ~any(above_threshold)
        min_angle = max(angles);  % 都不能分辨，返回最大值
        return;
    end
    
    first_above = find(above_threshold, 1);
    if first_above == 1
        min_angle = angles(1);  % 最小角度就已经能分辨
        return;
    end
    
    % 在相邻两点之间插值
    x1 = angles(first_above - 1);
    x2 = angles(first_above);
    y1 = success_rates(first_above - 1);
    y2 = success_rates(first_above);
    
    % 线性插值找到阈值对应的角度
    if y2 > y1
        min_angle = x1 + (threshold - y1) / (y2 - y1) * (x2 - x1);
    else
        min_angle = x2;
    end
end
