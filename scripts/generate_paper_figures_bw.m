%% ═══════════════════════════════════════════════════════════════════════════
%  论文图像生成 - 带颜色+线型+标记区分版 v2.0
%  特点:
%    1. 每个图像单独保存
%    2. 保留颜色 + 添加不同线型、标记区分（黑白打印友好）
%    3. 保持原有图像尺寸
%% ═══════════════════════════════════════════════════════════════════════════
clear; clc; close all;

addpath('asset');

%% ═══════════════════════════════════════════════════════════════════════════
%  配置
%% ═══════════════════════════════════════════════════════════════════════════

% 数据文件夹路径
data_folders = struct();
data_folders.comprehensive = 'E:\code\matlab\rader_mimo\big_paper\yongjin_paper\images\chapter4\comprehensive_motion_array_test_20251209_191404';
data_folders.dual_target = 'E:\code\matlab\rader_mimo\big_paper\yongjin_paper\images\chapter4\experiment_dual_target_mc_20251222_030219';
data_folders.vibration = 'E:\code\matlab\rader_mimo\big_paper\yongjin_paper\images\chapter4\experiment_vibration_multiband_20251210_022255';

% 输出文件夹
output_folder = fullfile('validation_results', 'paper_figures_color_bw');
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('  论文图像生成 - 颜色+线型+标记区分版\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');
fprintf('输出目录: %s\n\n', output_folder);

%% ═══════════════════════════════════════════════════════════════════════════
%  样式定义 (颜色 + 线型 + 标记)
%% ═══════════════════════════════════════════════════════════════════════════

% ===== 运动模式配色方案 (保留原颜色) =====
color_static = [0.2, 0.2, 0.2];       % 深灰 - 静态
color_x_trans = [0.85, 0.33, 0.1];    % 橙色 - x平移
color_y_trans = [0.0, 0.45, 0.74];    % 蓝色 - y平移
color_rotation = [0.47, 0.67, 0.19];  % 绿色 - 旋转
color_combined = [0.49, 0.18, 0.56];  % 紫色 - 平移+旋转
motion_colors = [color_static; color_x_trans; color_y_trans; color_rotation; color_combined];

% ===== 线型 (不同模式用不同线型) =====
motion_line_styles = {'-', '--', '-', ':', '-.'};

% ===== 标记形状 (不同模式用不同标记) =====
motion_markers = {'o', 's', 'd', '^', 'v'};

% ===== 频段配色 =====
band_colors = [0.2, 0.4, 0.8; 0.8, 0.3, 0.2];  % 蓝色, 红色
band_line_styles = {'-', '--'};
band_markers = {'o', 's'};

% ===== 柱状图配色 =====
bar_color_static = [0.3, 0.3, 0.3];   % 深灰
bar_color_motion = [0.0, 0.45, 0.74]; % 蓝色

% 设置默认字体
set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultLineLineWidth', 1.5);

% ===== 图像导出设置 =====
dpi_resolution = 300;  % 高分辨率 300 DPI

% 辅助函数：高清保存图片
save_figure_hd = @(fig, filepath) print(fig, filepath, '-dpng', sprintf('-r%d', dpi_resolution));

%% ═══════════════════════════════════════════════════════════════════════════
%  第一部分: 综合运动阵列测试图像
%% ═══════════════════════════════════════════════════════════════════════════

fprintf('【加载综合测试数据】\n');
load(fullfile(data_folders.comprehensive, 'experiment_results.mat'));

% 获取数据
snr_range = results.snr_range;
array_names = results.array_names;
motion_names = results.motion_names;

% 运动模式的线型和标记配置 (使用全局定义的颜色)

%% 图1: 单个阵列 静态vs运动 RMSE对比 (分开保存8个子图)
fprintf('生成图1系列: 各阵列静态vs运动RMSE...\n');

for arr_idx = 1:length(array_names)
    fig = figure('Position', [100, 100, 450, 350], 'Color', 'white');
    hold on;
    
    for mot_idx = 1:min(4, length(motion_names))  % 只画前4个运动模式
        rmse_curve = squeeze(results.rmse(arr_idx, mot_idx, :));
        
        plot(snr_range, rmse_curve, ...
            'LineStyle', motion_line_styles{mot_idx}, ...
            'Marker', motion_markers{mot_idx}, ...
            'Color', motion_colors(mot_idx, :), ...
            'LineWidth', 2, ...
            'MarkerSize', 8, ...
            'MarkerFaceColor', motion_colors(mot_idx, :), ...
            'DisplayName', motion_names{mot_idx});
    end
    
    xlabel('SNR (dB)', 'FontWeight', 'bold');
    ylabel('RMSE (°)', 'FontWeight', 'bold');
    title(sprintf('%s: 不同运动模式RMSE对比', array_names{arr_idx}), 'FontWeight', 'bold');
    legend('Location', 'southwest', 'FontSize', 9);  % 移到左下角避免遮挡数据
    grid on;
    box on;
    
    % 对数Y轴
    set(gca, 'YScale', 'log');
    xlim([snr_range(1)-1, snr_range(end)+1]);
    
    % 保存 (高分辨率)
    save_figure_hd(fig, fullfile(output_folder, sprintf('fig1_%d_%s_RMSE.png', arr_idx, array_names{arr_idx})));
    saveas(fig, fullfile(output_folder, sprintf('fig1_%d_%s_RMSE.eps', arr_idx, array_names{arr_idx})), 'epsc');
    close(fig);
end
fprintf('  已保存 fig1_1 到 fig1_8 (300 DPI)\n');

%% 图2: 代表性阵列对比 (4个独立图)
fprintf('生成图2系列: 代表性阵列对比...\n');

selected_arrays = [1, 2, 6, 8];  % ULA-8, URA-3x3, 圆阵-8, Y阵列
arr_display_names = {'ULA-8', 'URA-3×3', '圆阵-8', 'Y阵列'};

for i = 1:length(selected_arrays)
    arr_idx = selected_arrays(i);
    
    fig = figure('Position', [100, 100, 500, 400], 'Color', 'white');
    hold on;
    
    for mot_idx = 1:length(motion_names)
        rmse_curve = squeeze(results.rmse(arr_idx, mot_idx, :));
        
        plot(snr_range, rmse_curve, ...
            'LineStyle', motion_line_styles{mot_idx}, ...
            'Marker', motion_markers{mot_idx}, ...
            'Color', motion_colors(mot_idx, :), ...
            'LineWidth', 2.5, ...
            'MarkerSize', 9, ...
            'MarkerFaceColor', motion_colors(mot_idx, :), ...
            'DisplayName', motion_names{mot_idx});
    end
    
    xlabel('信噪比 (dB)', 'FontWeight', 'bold', 'FontSize', 12);
    ylabel('RMSE (°)', 'FontWeight', 'bold', 'FontSize', 12);
    title(sprintf('%s: 各运动模式RMSE', arr_display_names{i}), 'FontSize', 13, 'FontWeight', 'bold');
    legend('Location', 'southwest', 'FontSize', 10);  % 移到左下角避免遮挡数据
    grid on;
    box on;
    
    % 对数Y轴
    set(gca, 'YScale', 'log');
    xlim([snr_range(1)-1, snr_range(end)+1]);
    
    save_figure_hd(fig, fullfile(output_folder, sprintf('fig2_%d_%s_运动模式对比.png', i, arr_display_names{i})));
    saveas(fig, fullfile(output_folder, sprintf('fig2_%d_%s_运动模式对比.eps', i, arr_display_names{i})), 'epsc');
    close(fig);
end
fprintf('  已保存 fig2_1 到 fig2_4 (300 DPI)\n');

%% 图3: 孔径扩展对比 - 竖向水平条形图
fprintf('生成图3: 孔径扩展对比...\n');

fig = figure('Position', [100, 50, 1200, 1000], 'Color', 'white');
hold on;

aperture_data = results.aperture;  % [阵列 × 运动模式]
num_arrays = length(array_names);
num_motions = min(5, length(motion_names));

bar_height = 0.7;
bar_gap = 0.15;
group_gap = 1.8;

y_current = 0;

for arr_idx = num_arrays:-1:1
    for mot_idx = 1:num_motions
        y = y_current;
        x_val = aperture_data(arr_idx, mot_idx);
        
        barh(y, x_val, bar_height, 'FaceColor', motion_colors(mot_idx, :), ...
            'EdgeColor', 'k', 'LineWidth', 1.5);
        
        % 左侧标注运动模式
        text(-0.5, y, motion_names{mot_idx}, 'HorizontalAlignment', 'right', ...
            'FontSize', 11, 'FontWeight', 'normal');
        
        % 右侧数值标注
        text(x_val + 0.5, y, sprintf('%.1fλ', x_val), 'HorizontalAlignment', 'left', ...
            'FontSize', 11, 'FontWeight', 'bold');
        
        y_current = y_current + bar_height + bar_gap;
    end
    
    % 阵列名称标注
    y_center = y_current - num_motions * (bar_height + bar_gap) / 2 - bar_gap;
    text(34, y_center, array_names{arr_idx}, 'HorizontalAlignment', 'left', ...
        'FontSize', 13, 'FontWeight', 'bold', 'BackgroundColor', [0.95, 0.95, 0.95]);
    
    if arr_idx > 1
        y_current = y_current + group_gap;
        plot([0, 32], [y_current - group_gap/2, y_current - group_gap/2], ...
            'k-', 'LineWidth', 1, 'HandleVisibility', 'off');
    end
end

hold off;

set(gca, 'YTick', []);
xlabel('合成孔径 (λ)', 'FontWeight', 'bold', 'FontSize', 14);
title('各阵列配置的合成孔径对比', 'FontSize', 16, 'FontWeight', 'bold');
xlim([-8, 40]);
ylim([-0.5, y_current + 0.5]);
grid on;
box on;
set(gca, 'FontSize', 12);

save_figure_hd(fig, fullfile(output_folder, 'fig3_孔径扩展对比.png'));
saveas(fig, fullfile(output_folder, 'fig3_孔径扩展对比.eps'), 'epsc');
close(fig);
fprintf('  已保存 fig3 (300 DPI)\n');

%% 图4: 改善倍数对比 - 竖向水平条形图
fprintf('生成图4: 改善倍数对比...\n');

low_snr_idx = find(snr_range <= -10, 1, 'last');
if isempty(low_snr_idx), low_snr_idx = 1; end

improvement_ratio = zeros(length(array_names), length(motion_names)-1);
for arr_idx = 1:length(array_names)
    static_rmse = results.rmse(arr_idx, 1, low_snr_idx);
    for mot_idx = 2:length(motion_names)
        motion_rmse = results.rmse(arr_idx, mot_idx, low_snr_idx);
        improvement_ratio(arr_idx, mot_idx-1) = static_rmse / max(motion_rmse, 0.01);
    end
end

num_motion_types = min(4, size(improvement_ratio, 2));
motion_labels = motion_names(2:num_motion_types+1);  % 排除静态

fig = figure('Position', [100, 50, 1100, 900], 'Color', 'white');
hold on;

bar_height = 0.7;
bar_gap = 0.15;
group_gap = 1.8;

y_current = 0;
max_improvement = max(improvement_ratio(:));

for arr_idx = num_arrays:-1:1
    for mot_idx = 1:num_motion_types
        y = y_current;
        x_val = improvement_ratio(arr_idx, mot_idx);
        
        barh(y, x_val, bar_height, 'FaceColor', motion_colors(mot_idx+1, :), ...
            'EdgeColor', 'k', 'LineWidth', 1.5);
        
        % 左侧标注运动模式
        text(-0.1, y, motion_labels{mot_idx}, 'HorizontalAlignment', 'right', ...
            'FontSize', 11, 'FontWeight', 'normal');
        
        % 右侧数值标注
        text(x_val + 0.1, y, sprintf('%.2f×', x_val), 'HorizontalAlignment', 'left', ...
            'FontSize', 11, 'FontWeight', 'bold');
        
        y_current = y_current + bar_height + bar_gap;
    end
    
    % 阵列名称标注
    y_center = y_current - num_motion_types * (bar_height + bar_gap) / 2 - bar_gap;
    text(max_improvement + 0.8, y_center, array_names{arr_idx}, 'HorizontalAlignment', 'left', ...
        'FontSize', 13, 'FontWeight', 'bold', 'BackgroundColor', [0.95, 0.95, 0.95]);
    
    if arr_idx > 1
        y_current = y_current + group_gap;
        plot([0, max_improvement + 0.5], [y_current - group_gap/2, y_current - group_gap/2], ...
            'k-', 'LineWidth', 1, 'HandleVisibility', 'off');
    end
end

hold off;

set(gca, 'YTick', []);
xlabel('改善倍数 (×)', 'FontWeight', 'bold', 'FontSize', 14);
title(sprintf('SNR = %d dB 时相对静态阵列的RMSE改善', snr_range(low_snr_idx)), ...
    'FontSize', 16, 'FontWeight', 'bold');
xlim([-2, max_improvement + 2]);
ylim([-0.5, y_current + 0.5]);
grid on;
box on;
set(gca, 'FontSize', 12);

save_figure_hd(fig, fullfile(output_folder, 'fig4_改善倍数对比.png'));
saveas(fig, fullfile(output_folder, 'fig4_改善倍数对比.eps'), 'epsc');
close(fig);
fprintf('  已保存 fig4 (300 DPI)\n');

%% 图5: 阵列形状可视化 (4个独立图)
fprintf('生成图5系列: 阵列形状...\n');

c = physconst('LightSpeed');
fc = 3e9;
lambda = c / fc;
d = lambda / 2;

% 阵列创建函数
display_arrays = [1, 3, 4, 6];
display_arr_names = {'ULA-8', 'URA-4×2', 'L阵列', '圆阵-8'};

for i = 1:length(display_arrays)
    fig = figure('Position', [100, 100, 350, 350], 'Color', 'white');
    
    % 根据阵列类型创建
    switch display_arrays(i)
        case 1  % ULA-8
            x_pos = ((0:7) - 3.5) * d;
            pos = [x_pos', zeros(8,1), zeros(8,1)];
        case 3  % URA-4x2
            pos = [];
            for iy = 1:2
                for ix = 1:4
                    x = (ix - 1 - 1.5) * d;
                    y = (iy - 1 - 0.5) * d;
                    pos = [pos; x, y, 0];
                end
            end
        case 4  % L阵列
            pos = [];
            for ix = 1:4
                pos = [pos; (ix-1)*d, 0, 0];
            end
            for iy = 2:4
                pos = [pos; 0, (iy-1)*d, 0];
            end
        case 6  % 圆阵-8
            pos = [];
            for k = 1:8
                angle = 2*pi*(k-1)/8;
                pos = [pos; lambda*cos(angle), lambda*sin(angle), 0];
            end
    end
    
    % 使用蓝色填充的圆形标记（带黑色边框便于黑白识别）
    scatter(pos(:,1)*1000, pos(:,2)*1000, 150, 'filled', ...
        'MarkerFaceColor', [0.2, 0.4, 0.8], ...
        'MarkerEdgeColor', [0, 0, 0], 'LineWidth', 1.5);
    
    xlabel('x (mm)', 'FontWeight', 'bold');
    ylabel('y (mm)', 'FontWeight', 'bold');
    title(display_arr_names{i}, 'FontWeight', 'bold', 'FontSize', 13);
    axis equal;
    grid on;
    box on;
    
    ax = gca;
    max_range = max(abs([ax.XLim, ax.YLim])) * 1.4;
    if max_range < 1, max_range = 100; end
    xlim([-max_range, max_range]);
    ylim([-max_range, max_range]);
    
    save_figure_hd(fig, fullfile(output_folder, sprintf('fig5_%d_%s_形状.png', i, display_arr_names{i})));
    saveas(fig, fullfile(output_folder, sprintf('fig5_%d_%s_形状.eps', i, display_arr_names{i})), 'epsc');
    close(fig);
end
fprintf('  已保存 fig5_1 到 fig5_4 (300 DPI)\n');

%% 图6: 虚拟阵列轨迹 (3个独立图)
fprintf('生成图6系列: 虚拟阵列轨迹...\n');

v_demo = 5;
T_obs_demo = 0.5;
num_snaps_demo = 64;
t_demo = linspace(0, T_obs_demo, num_snaps_demo);

% 创建ULA
x_pos = ((0:7) - 3.5) * d;
ula_elements = [x_pos', zeros(8,1), zeros(8,1)];

% 6a: 静态
fig = figure('Position', [100, 100, 350, 500], 'Color', 'white');
scatter(ula_elements(:,1)/lambda, ula_elements(:,2)/lambda, 100, 'filled', ...
    'MarkerFaceColor', color_static, 'MarkerEdgeColor', [0, 0, 0]);
xlabel('x (λ)', 'FontWeight', 'bold');
ylabel('y (λ)', 'FontWeight', 'bold');
title('(a) 静态阵列', 'FontWeight', 'bold', 'FontSize', 13);
axis equal;
grid on;
box on;
xlim([-5, 5]);
ylim([-2, 28]);
aperture_static = (max(ula_elements(:,1)) - min(ula_elements(:,1))) / lambda;
text(0, -1.5, sprintf('孔径: %.1fλ', aperture_static), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
save_figure_hd(fig, fullfile(output_folder, 'fig6_1_静态轨迹.png'));
saveas(fig, fullfile(output_folder, 'fig6_1_静态轨迹.eps'), 'epsc');
close(fig);

% 6b: Y平移
fig = figure('Position', [100, 100, 350, 500], 'Color', 'white');
all_pos = [];
for k = 1:num_snaps_demo
    y_offset = v_demo * t_demo(k);
    pos_k = ula_elements + repmat([0, y_offset, 0], 8, 1);
    all_pos = [all_pos; pos_k];
end
scatter(all_pos(:,1)/lambda, all_pos(:,2)/lambda, 8, 'filled', ...
    'MarkerFaceColor', color_y_trans, 'MarkerFaceAlpha', 0.5);
hold on;
quiver(0, 0, 0, v_demo*T_obs_demo/lambda*0.9, 0, 'r', 'LineWidth', 2.5, 'MaxHeadSize', 1);
hold off;
xlabel('x (λ)', 'FontWeight', 'bold');
ylabel('y (λ)', 'FontWeight', 'bold');
title('(b) Y方向平移', 'FontWeight', 'bold', 'FontSize', 13);
axis equal;
grid on;
box on;
xlim([-5, 5]);
ylim([-2, 28]);
aperture_y = (max(all_pos(:,2)) - min(all_pos(:,2))) / lambda;
text(0, -1.5, sprintf('孔径: %.1fλ', aperture_y), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
save_figure_hd(fig, fullfile(output_folder, 'fig6_2_Y平移轨迹.png'));
saveas(fig, fullfile(output_folder, 'fig6_2_Y平移轨迹.eps'), 'epsc');
close(fig);

% 6c: 旋转
fig = figure('Position', [100, 100, 350, 350], 'Color', 'white');
all_pos = [];
for k = 1:num_snaps_demo
    rot_angle = 90 * t_demo(k) / T_obs_demo;
    rot_rad = deg2rad(rot_angle);
    R = [cos(rot_rad), -sin(rot_rad), 0; sin(rot_rad), cos(rot_rad), 0; 0, 0, 1];
    pos_k = (R * ula_elements')';
    all_pos = [all_pos; pos_k];
end
scatter(all_pos(:,1)/lambda, all_pos(:,2)/lambda, 8, 'filled', ...
    'MarkerFaceColor', color_rotation, 'MarkerFaceAlpha', 0.5);
xlabel('x (λ)', 'FontWeight', 'bold');
ylabel('y (λ)', 'FontWeight', 'bold');
title('(c) 旋转 (90°)', 'FontWeight', 'bold', 'FontSize', 13);
axis equal;
grid on;
box on;
xlim([-5, 5]);
ylim([-5, 5]);
aperture_rot = sqrt((max(all_pos(:,1))-min(all_pos(:,1)))^2 + (max(all_pos(:,2))-min(all_pos(:,2)))^2) / lambda;
text(0, -4.2, sprintf('孔径: %.1fλ', aperture_rot), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
save_figure_hd(fig, fullfile(output_folder, 'fig6_3_旋转轨迹.png'));
saveas(fig, fullfile(output_folder, 'fig6_3_旋转轨迹.eps'), 'epsc');
close(fig);

fprintf('  已保存 fig6_1 到 fig6_3 (300 DPI)\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  第二部分: 双目标分辨实验图像 (蒙特卡洛版本)
%% ═══════════════════════════════════════════════════════════════════════════

fprintf('\n【加载双目标分辨数据 (蒙特卡洛)】\n');
load(fullfile(data_folders.dual_target, 'experiment_results.mat'));

angle_separations = results.separations;
snr_values = results.snr_values;
phi_center = results.phi_center;
phi_search = results.phi_search;
num_trials = results.num_trials;

% 使用中间SNR (通常是10dB) 作为主要展示
typical_snr_idx = ceil(length(snr_values) / 2);
typical_snr = snr_values(typical_snr_idx);

% 提取该SNR下的成功率
static_success = results.static_success_rate(:, typical_snr_idx);  % [间隔 × 1]
motion_success = results.motion_success_rate(:, typical_snr_idx);

%% 图7: 分辨成功率对比柱状图 (百分比)
fprintf('生成图7: 分辨成功率对比...\n');

fig = figure('Position', [100, 100, 600, 400], 'Color', 'white');

bar_data = [static_success, motion_success];
b = bar(bar_data, 'grouped');
b(1).FaceColor = bar_color_static;  % 深灰 - 静态
b(1).EdgeColor = [0, 0, 0];
b(1).LineWidth = 1.5;
b(2).FaceColor = bar_color_motion;  % 蓝色 - 运动
b(2).EdgeColor = [0, 0, 0];
b(2).LineWidth = 1.5;

hold on;
% 添加静态理论分辨率参考线
if isfield(results, 'static_resolution')
    static_res = results.static_resolution;
    xline_pos = find(angle_separations >= static_res, 1);
    if ~isempty(xline_pos)
        xline(xline_pos - 0.5, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
        text(xline_pos - 0.3, 105, sprintf('静态理论\n%.1f°', static_res), ...
            'FontSize', 9, 'HorizontalAlignment', 'left');
    end
end
hold off;

set(gca, 'XTick', 1:length(angle_separations), ...
    'XTickLabel', arrayfun(@(x) sprintf('%d°', x), angle_separations, 'UniformOutput', false));
xlabel('双目标角度间隔', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('分辨成功率 (%)', 'FontWeight', 'bold', 'FontSize', 12);
title(sprintf('双目标分辨成功率对比 (SNR=%ddB, N=%d)', typical_snr, num_trials), ...
    'FontSize', 13, 'FontWeight', 'bold');
legend({'静态阵列', '运动阵列'}, 'Location', 'southeast', 'FontSize', 10);
grid on;
box on;
ylim([0, 110]);

save_figure_hd(fig, fullfile(output_folder, 'fig7_分辨成功率对比.png'));
saveas(fig, fullfile(output_folder, 'fig7_分辨成功率对比.eps'), 'epsc');
close(fig);
fprintf('  已保存 fig7 (300 DPI)\n');

%% 图8: 最小可分辨角度对比 - 水平条形图 (90%成功率标准)
fprintf('生成图8: 最小可分辨角度...\n');

threshold = 90;  % 90%成功率标准
static_min = find_min_angle_mc(angle_separations, static_success, threshold);
motion_min = find_min_angle_mc(angle_separations, motion_success, threshold);

fig = figure('Position', [100, 100, 600, 300], 'Color', 'white');
hold on;

% 水平条形图
bar_height = 0.5;
y_positions = [2, 1];  % 静态在上，运动在下

barh(y_positions(1), static_min, bar_height, 'FaceColor', bar_color_static, 'EdgeColor', 'k', 'LineWidth', 1.5);
barh(y_positions(2), motion_min, bar_height, 'FaceColor', bar_color_motion, 'EdgeColor', 'k', 'LineWidth', 1.5);

% 左侧标注
text(-1, y_positions(1), '静态阵列', 'HorizontalAlignment', 'right', 'FontSize', 12, 'FontWeight', 'bold');
text(-1, y_positions(2), '运动阵列', 'HorizontalAlignment', 'right', 'FontSize', 12, 'FontWeight', 'bold');

% 右侧数值标注
text(static_min + 0.5, y_positions(1), sprintf('%.1f°', static_min), 'HorizontalAlignment', 'left', 'FontSize', 13, 'FontWeight', 'bold');
text(motion_min + 0.5, y_positions(2), sprintf('%.1f°', motion_min), 'HorizontalAlignment', 'left', 'FontSize', 13, 'FontWeight', 'bold');

% 改善倍数标注
if motion_min > 0 && motion_min < static_min
    improvement = static_min / motion_min;
    text(max(static_min, motion_min) + 3, 1.5, sprintf('分辨率提升\n%.1f 倍', improvement), ...
        'HorizontalAlignment', 'left', 'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.1, 0.5, 0.1]);
end

hold off;

set(gca, 'YTick', []);
xlabel('最小可分辨角度 (°) [成功率≥90%]', 'FontWeight', 'bold', 'FontSize', 12);
title(sprintf('双目标分辨能力对比 (SNR=%ddB)', typical_snr), 'FontSize', 14, 'FontWeight', 'bold');
xlim([-5, max(static_min, motion_min) + 10]);
ylim([0.3, 2.7]);
grid on;
box on;

save_figure_hd(fig, fullfile(output_folder, 'fig8_最小可分辨角度.png'));
saveas(fig, fullfile(output_folder, 'fig8_最小可分辨角度.eps'), 'epsc');
close(fig);
fprintf('  已保存 fig8 (300 DPI)\n');

%% 图9: MUSIC谱对比 (找运动优势的间隔)
fprintf('生成图9: MUSIC谱对比...\n');

% 找一个运动成功率高，静态成功率低的间隔
typical_sep_idx = 0;
for i = 1:length(angle_separations)
    if motion_success(i) >= 90 && static_success(i) < 50
        typical_sep_idx = i;
        break;
    end
end
if typical_sep_idx == 0
    typical_sep_idx = find(angle_separations == 5, 1);
    if isempty(typical_sep_idx), typical_sep_idx = 2; end
end

sep = angle_separations(typical_sep_idx);
phi1 = phi_center - sep/2;
phi2 = phi_center + sep/2;

% 静态谱 (深灰色实线)
fig = figure('Position', [100, 100, 450, 350], 'Color', 'white');
spectrum_db = 10*log10(results.static_spectra{typical_sep_idx} / max(results.static_spectra{typical_sep_idx}));
plot(phi_search, spectrum_db, '-', 'Color', bar_color_static, 'LineWidth', 2);
hold on;
xline(phi1, 'r--', 'LineWidth', 1.5);
xline(phi2, 'r--', 'LineWidth', 1.5);
hold off;
xlabel('方位角 φ (°)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('归一化功率 (dB)', 'FontWeight', 'bold', 'FontSize', 12);
title(sprintf('静态阵列MUSIC谱 (间隔=%d°, 成功率=%.0f%%)', sep, static_success(typical_sep_idx)), ...
    'FontWeight', 'bold', 'FontSize', 13);
xlim([phi_center-25, phi_center+25]);
ylim([-40, 5]);
grid on;
box on;
legend({'MUSIC谱', '真实目标'}, 'Location', 'south');

save_figure_hd(fig, fullfile(output_folder, 'fig9_1_静态MUSIC谱.png'));
saveas(fig, fullfile(output_folder, 'fig9_1_静态MUSIC谱.eps'), 'epsc');
close(fig);

% 运动谱 (蓝色实线)
fig = figure('Position', [100, 100, 450, 350], 'Color', 'white');
spectrum_db = 10*log10(results.motion_spectra{typical_sep_idx} / max(results.motion_spectra{typical_sep_idx}));
plot(phi_search, spectrum_db, '-', 'Color', bar_color_motion, 'LineWidth', 2);
hold on;
xline(phi1, 'r--', 'LineWidth', 1.5);
xline(phi2, 'r--', 'LineWidth', 1.5);
hold off;
xlabel('方位角 φ (°)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('归一化功率 (dB)', 'FontWeight', 'bold', 'FontSize', 12);
title(sprintf('运动阵列时间平滑MUSIC谱 (间隔=%d°, 成功率=%.0f%%)', sep, motion_success(typical_sep_idx)), ...
    'FontWeight', 'bold', 'FontSize', 13);
xlim([phi_center-25, phi_center+25]);
ylim([-40, 5]);
grid on;
box on;
legend({'时间平滑MUSIC谱', '真实目标'}, 'Location', 'south');

save_figure_hd(fig, fullfile(output_folder, 'fig9_2_运动MUSIC谱.png'));
saveas(fig, fullfile(output_folder, 'fig9_2_运动MUSIC谱.eps'), 'epsc');
close(fig);
fprintf('  已保存 fig9_1, fig9_2 (300 DPI)\n');

%% 图10: 不同SNR下的成功率曲线
fprintf('生成图10: 不同SNR下的成功率曲线...\n');

fig = figure('Position', [100, 100, 600, 450], 'Color', 'white');
hold on;

% 静态阵列用虚线+空心标记，运动阵列用实线+实心标记
snr_markers = {'o', 's', 'd'};
snr_colors_static = [0.5, 0.5, 0.5];  % 灰色系
snr_colors_motion = [0.0, 0.45, 0.74];  % 蓝色

for snr_idx = 1:length(snr_values)
    % 静态
    plot(angle_separations, results.static_success_rate(:, snr_idx), ...
        '--', 'Color', snr_colors_static, 'LineWidth', 1.5, ...
        'Marker', snr_markers{snr_idx}, 'MarkerSize', 8, ...
        'MarkerFaceColor', 'none', ...
        'DisplayName', sprintf('静态 SNR=%ddB', snr_values(snr_idx)));
    
    % 运动
    plot(angle_separations, results.motion_success_rate(:, snr_idx), ...
        '-', 'Color', snr_colors_motion, 'LineWidth', 2, ...
        'Marker', snr_markers{snr_idx}, 'MarkerSize', 8, ...
        'MarkerFaceColor', snr_colors_motion, ...
        'DisplayName', sprintf('运动 SNR=%ddB', snr_values(snr_idx)));
end

% 90%阈值线
yline(90, 'k--', 'LineWidth', 1.5, 'DisplayName', '90%阈值');

hold off;
xlabel('双目标角度间隔 (°)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('分辨成功率 (%)', 'FontWeight', 'bold', 'FontSize', 12);
title('不同SNR下的分辨成功率', 'FontWeight', 'bold', 'FontSize', 13);
legend('Location', 'southeast', 'FontSize', 9, 'NumColumns', 2);
grid on;
box on;
xlim([min(angle_separations)-1, max(angle_separations)+1]);
ylim([0, 105]);

save_figure_hd(fig, fullfile(output_folder, 'fig10_SNR对比成功率曲线.png'));
saveas(fig, fullfile(output_folder, 'fig10_SNR对比成功率曲线.eps'), 'epsc');
close(fig);
fprintf('  已保存 fig10 (300 DPI)\n');

%% 图11: 多间隔MUSIC谱对比 (独立子图)
fprintf('生成图11系列: 多间隔MUSIC谱...\n');

selected_seps = [3, 5, 10, 15];
for s_idx = 1:length(selected_seps)
    sep = selected_seps(s_idx);
    idx = find(angle_separations == sep, 1);
    if isempty(idx), continue; end
    
    phi1 = phi_center - sep/2;
    phi2 = phi_center + sep/2;
    
    % 静态 (深灰色)
    fig = figure('Position', [100, 100, 400, 300], 'Color', 'white');
    spectrum_db = 10*log10(results.static_spectra{idx} / max(results.static_spectra{idx}));
    plot(phi_search, spectrum_db, '-', 'Color', bar_color_static, 'LineWidth', 2);
    hold on;
    xline(phi1, 'r--', 'LineWidth', 1.5);
    xline(phi2, 'r--', 'LineWidth', 1.5);
    hold off;
    xlim([max(35, phi_center-25), min(85, phi_center+25)]);
    ylim([-30, 5]);
    grid on;
    box on;
    xlabel('φ (°)', 'FontWeight', 'bold');
    ylabel('功率 (dB)', 'FontWeight', 'bold');
    title(sprintf('静态 %d° (成功率%.0f%%)', sep, static_success(idx)), 'FontWeight', 'bold');
    save_figure_hd(fig, fullfile(output_folder, sprintf('fig11_%d_静态_%ddeg.png', s_idx, sep)));
    saveas(fig, fullfile(output_folder, sprintf('fig11_%d_静态_%ddeg.eps', s_idx, sep)), 'epsc');
    close(fig);
    
    % 运动 (蓝色)
    fig = figure('Position', [100, 100, 400, 300], 'Color', 'white');
    spectrum_db = 10*log10(results.motion_spectra{idx} / max(results.motion_spectra{idx}));
    plot(phi_search, spectrum_db, '-', 'Color', bar_color_motion, 'LineWidth', 2);
    hold on;
    xline(phi1, 'r--', 'LineWidth', 1.5);
    xline(phi2, 'r--', 'LineWidth', 1.5);
    hold off;
    xlim([max(35, phi_center-25), min(85, phi_center+25)]);
    ylim([-30, 5]);
    grid on;
    box on;
    xlabel('φ (°)', 'FontWeight', 'bold');
    ylabel('功率 (dB)', 'FontWeight', 'bold');
    title(sprintf('运动 %d° (成功率%.0f%%)', sep, motion_success(idx)), 'FontWeight', 'bold');
    save_figure_hd(fig, fullfile(output_folder, sprintf('fig11_%d_运动_%ddeg.png', s_idx, sep)));
    saveas(fig, fullfile(output_folder, sprintf('fig11_%d_运动_%ddeg.eps', s_idx, sep)), 'epsc');
    close(fig);
end
fprintf('  已保存 fig11 系列 (300 DPI)\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  第三部分: 震动鲁棒性实验图像
%% ═══════════════════════════════════════════════════════════════════════════

fprintf('\n【加载震动鲁棒性数据】\n');
load(fullfile(data_folders.vibration, 'experiment_results.mat'));

vibration_cm_range = results_save.vibration_cm;
band_names = results_save.band_names;
band_lambda = results_save.band_lambda;
num_bands = length(band_names);

%% 图12: RMSE vs 震动幅度(cm)
fprintf('生成图12: RMSE vs 物理震动...\n');

fig = figure('Position', [100, 100, 500, 400], 'Color', 'white');
hold on;

% 不同频段用不同颜色+线型+标记
for b = 1:num_bands
    plot(vibration_cm_range, results_save.rmse(b,:), ...
        'LineStyle', band_line_styles{b}, ...
        'Marker', band_markers{b}, ...
        'Color', band_colors(b,:), ...
        'LineWidth', 2.5, ...
        'MarkerSize', 9, ...
        'MarkerFaceColor', band_colors(b,:), ...
        'DisplayName', band_names{b});
end

% 参考线
yline(1, 'k:', 'LineWidth', 1.5, 'DisplayName', '1° 精度线');
yline(2, 'k-.', 'LineWidth', 1.5, 'DisplayName', '2° 精度线');

hold off;
xlabel('震动标准差 (cm)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('RMSE (°)', 'FontWeight', 'bold', 'FontSize', 12);
title('RMSE vs 物理震动幅度', 'FontWeight', 'bold', 'FontSize', 13);
legend('Location', 'northwest', 'FontSize', 10);
grid on;
box on;
xlim([0, max(vibration_cm_range)*1.05]);
ylim([0, 12]);

save_figure_hd(fig, fullfile(output_folder, 'fig12_RMSE_vs_物理震动.png'));
saveas(fig, fullfile(output_folder, 'fig12_RMSE_vs_物理震动.eps'), 'epsc');
close(fig);
fprintf('  已保存 fig12 (300 DPI)\n');

%% 图13: RMSE vs 震动幅度(lambda)
fprintf('生成图13: RMSE vs 电相位震动...\n');

fig = figure('Position', [100, 100, 500, 400], 'Color', 'white');
hold on;

for b = 1:num_bands
    plot(results_save.vib_lambda(b,:), results_save.rmse(b,:), ...
        'LineStyle', band_line_styles{b}, ...
        'Marker', band_markers{b}, ...
        'Color', band_colors(b,:), ...
        'LineWidth', 2.5, ...
        'MarkerSize', 9, ...
        'MarkerFaceColor', band_colors(b,:), ...
        'DisplayName', band_names{b});
end

yline(1, 'k:', 'LineWidth', 1.5, 'DisplayName', '1° 精度线');
yline(2, 'k-.', 'LineWidth', 1.5, 'DisplayName', '2° 精度线');

hold off;
xlabel('震动标准差 (λ)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('RMSE (°)', 'FontWeight', 'bold', 'FontSize', 12);
title('RMSE vs 电相位震动', 'FontWeight', 'bold', 'FontSize', 13);
legend('Location', 'northwest', 'FontSize', 10);
grid on;
box on;
xlim([0, 0.15]);
ylim([0, 12]);

save_figure_hd(fig, fullfile(output_folder, 'fig13_RMSE_vs_电相位震动.png'));
saveas(fig, fullfile(output_folder, 'fig13_RMSE_vs_电相位震动.eps'), 'epsc');
close(fig);
fprintf('  已保存 fig13 (300 DPI)\n');

%% 图14: 容许震动对比柱状图
fprintf('生成图14: 容许震动对比...\n');

tolerance_1deg = zeros(1, num_bands);
tolerance_2deg = zeros(1, num_bands);

for b = 1:num_bands
    idx_1deg = find(results_save.rmse(b,:) < 1, 1, 'last');
    idx_2deg = find(results_save.rmse(b,:) < 2, 1, 'last');
    
    if ~isempty(idx_1deg)
        tolerance_1deg(b) = vibration_cm_range(idx_1deg);
    end
    if ~isempty(idx_2deg)
        tolerance_2deg(b) = vibration_cm_range(idx_2deg);
    end
end

fig = figure('Position', [100, 100, 700, 400], 'Color', 'white');
hold on;

% 颜色配置
tol_colors = [0.2, 0.6, 0.3; 0.9, 0.7, 0.2];  % 绿色, 黄色
tol_labels = {'RMSE < 1°', 'RMSE < 2°'};

bar_height = 0.6;
bar_gap = 0.15;
group_gap = 1.2;

y_current = 0;
max_tol = max([tolerance_1deg, tolerance_2deg]) + 1;

for band_idx = num_bands:-1:1
    for tol_idx = 1:2
        y = y_current;
        if tol_idx == 1
            x_val = tolerance_1deg(band_idx);
        else
            x_val = tolerance_2deg(band_idx);
        end
        
        barh(y, x_val, bar_height, 'FaceColor', tol_colors(tol_idx, :), ...
            'EdgeColor', 'k', 'LineWidth', 1.5);
        
        % 左侧标注精度等级
        text(-0.1, y, tol_labels{tol_idx}, 'HorizontalAlignment', 'right', ...
            'FontSize', 11, 'FontWeight', 'normal');
        
        % 右侧数值标注
        text(x_val + 0.1, y, sprintf('%.1f cm', x_val), 'HorizontalAlignment', 'left', ...
            'FontSize', 11, 'FontWeight', 'bold');
        
        y_current = y_current + bar_height + bar_gap;
    end
    
    % 频段名称标注
    y_center = y_current - 2 * (bar_height + bar_gap) / 2 - bar_gap/2;
    text(max_tol + 0.5, y_center, band_names{band_idx}, 'HorizontalAlignment', 'left', ...
        'FontSize', 13, 'FontWeight', 'bold', 'BackgroundColor', [0.95, 0.95, 0.95]);
    
    if band_idx > 1
        y_current = y_current + group_gap;
        plot([0, max_tol], [y_current - group_gap/2, y_current - group_gap/2], ...
            'k-', 'LineWidth', 1, 'HandleVisibility', 'off');
    end
end

hold off;

set(gca, 'YTick', []);
xlabel('容许震动 (cm)', 'FontWeight', 'bold', 'FontSize', 12);
title('各频段最大容许震动对比', 'FontWeight', 'bold', 'FontSize', 14);
xlim([-2, max_tol + 2]);
ylim([-0.5, y_current + 0.8]);
grid on;
box on;
set(gca, 'FontSize', 11);

save_figure_hd(fig, fullfile(output_folder, 'fig14_容许震动对比.png'));
saveas(fig, fullfile(output_folder, 'fig14_容许震动对比.eps'), 'epsc');
close(fig);
fprintf('  已保存 fig14 (300 DPI)\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  完成
%% ═══════════════════════════════════════════════════════════════════════════

fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('所有图像生成完成！\n');
fprintf('输出目录: %s\n', output_folder);
fprintf('═══════════════════════════════════════════════════════════════════\n');

% 列出所有生成的文件
files = dir(fullfile(output_folder, '*.png'));
fprintf('\n生成的PNG文件 (%d个):\n', length(files));
for i = 1:min(10, length(files))
    fprintf('  %s\n', files(i).name);
end
if length(files) > 10
    fprintf('  ... 还有 %d 个文件\n', length(files) - 10);
end

%% ═══════════════════════════════════════════════════════════════════════════
%  辅助函数
%% ═══════════════════════════════════════════════════════════════════════════

function min_angle = find_min_angle_mc(angles, success_rates, threshold)
    % 从蒙特卡洛结果中找到达到阈值成功率的最小角度
    % 使用线性插值获取更精确的值
    
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

function draw_pattern_in_bar(x_left, x_right, y_bottom, y_top, pattern, spacing, line_width)
    % 在柱状图内绘制清晰的填充图案
    % pattern: 'slash'=斜线, 'backslash'=反斜线, 'horizontal'=水平线, 
    %          'vertical'=垂直线, 'cross'=交叉, 'dots'=点阵
    
    w = x_right - x_left;
    h = y_top - y_bottom;
    
    switch pattern
        case 'slash'  % 正斜线 /
            % 从左下到右上
            for offset = -h:spacing:w+h
                x1 = x_left + offset;
                y1 = y_bottom;
                x2 = x1 + h * 0.8;  % 斜率约45度
                y2 = y_top;
                
                % 裁剪到矩形内
                [xc, yc] = clip_line(x1, y1, x2, y2, x_left, x_right, y_bottom, y_top);
                if ~isempty(xc)
                    plot(xc, yc, 'k-', 'LineWidth', line_width);
                end
            end
            
        case 'backslash'  % 反斜线 \
            for offset = -h:spacing:w+h
                x1 = x_left + offset;
                y1 = y_top;
                x2 = x1 + h * 0.8;
                y2 = y_bottom;
                
                [xc, yc] = clip_line(x1, y1, x2, y2, x_left, x_right, y_bottom, y_top);
                if ~isempty(xc)
                    plot(xc, yc, 'k-', 'LineWidth', line_width);
                end
            end
            
        case 'horizontal'  % 水平线
            for y = y_bottom + spacing/2 : spacing : y_top
                plot([x_left, x_right], [y, y], 'k-', 'LineWidth', line_width);
            end
            
        case 'vertical'  % 垂直线
            for x = x_left + w*0.1 : spacing*w/10 : x_right - w*0.1
                plot([x, x], [y_bottom, y_top], 'k-', 'LineWidth', line_width);
            end
            
        case 'cross'  % 交叉网格
            draw_pattern_in_bar(x_left, x_right, y_bottom, y_top, 'slash', spacing, line_width * 0.8);
            draw_pattern_in_bar(x_left, x_right, y_bottom, y_top, 'backslash', spacing, line_width * 0.8);
            
        case 'dots'  % 点阵
            marker_size = line_width;
            x_step = w / 3;
            y_step = spacing;
            row = 0;
            for y = y_bottom + y_step/2 : y_step : y_top
                row = row + 1;
                x_offset = mod(row, 2) * x_step/2;  % 交错排列
                for x = x_left + x_step/2 + x_offset : x_step : x_right
                    if x > x_left && x < x_right
                        plot(x, y, 'ko', 'MarkerSize', marker_size, 'MarkerFaceColor', 'k');
                    end
                end
            end
    end
end

function [xc, yc] = clip_line(x1, y1, x2, y2, x_min, x_max, y_min, y_max)
    % 简单的线段裁剪
    xc = []; yc = [];
    
    % 计算线段与四边的交点
    points = [];
    
    % 检查线段斜率
    if abs(x2 - x1) < 1e-10
        % 垂直线
        if x1 >= x_min && x1 <= x_max
            y_start = max(y_min, min(y1, y2));
            y_end = min(y_max, max(y1, y2));
            if y_start < y_end
                xc = [x1, x1];
                yc = [y_start, y_end];
            end
        end
        return;
    end
    
    m = (y2 - y1) / (x2 - x1);
    b = y1 - m * x1;
    
    % 与左边界的交点
    y_at_xmin = m * x_min + b;
    if y_at_xmin >= y_min && y_at_xmin <= y_max
        t = (x_min - x1) / (x2 - x1);
        if t >= 0 && t <= 1
            points = [points; x_min, y_at_xmin];
        end
    end
    
    % 与右边界的交点
    y_at_xmax = m * x_max + b;
    if y_at_xmax >= y_min && y_at_xmax <= y_max
        t = (x_max - x1) / (x2 - x1);
        if t >= 0 && t <= 1
            points = [points; x_max, y_at_xmax];
        end
    end
    
    % 与下边界的交点
    if abs(m) > 1e-10
        x_at_ymin = (y_min - b) / m;
        if x_at_ymin >= x_min && x_at_ymin <= x_max
            t = (x_at_ymin - x1) / (x2 - x1);
            if t >= 0 && t <= 1
                points = [points; x_at_ymin, y_min];
            end
        end
    end
    
    % 与上边界的交点
    if abs(m) > 1e-10
        x_at_ymax = (y_max - b) / m;
        if x_at_ymax >= x_min && x_at_ymax <= x_max
            t = (x_at_ymax - x1) / (x2 - x1);
            if t >= 0 && t <= 1
                points = [points; x_at_ymax, y_max];
            end
        end
    end
    
    % 去重
    if size(points, 1) >= 2
        points = unique(round(points, 6), 'rows');
        if size(points, 1) >= 2
            xc = points(1:2, 1)';
            yc = points(1:2, 2)';
        end
    end
end

function add_hatch_to_bar(x_left, x_right, y_bottom, y_top, pattern, spacing, line_width)
    % 在柱状图区域内绘制填充图案（带边界剪裁）
    if nargin < 6, spacing = 2; end
    if nargin < 7, line_width = 0.5; end
    
    w = x_right - x_left;
    h = y_top - y_bottom;
    
    % 创建剪裁多边形
    clip_x = [x_left, x_right, x_right, x_left];
    clip_y = [y_bottom, y_bottom, y_top, y_top];
    
    switch pattern
        case '/'  % 正斜线 (45度)
            for offset = -h:spacing:w+h
                x1 = x_left + offset;
                y1 = y_bottom;
                x2 = x_left + offset + h;
                y2 = y_top;
                [xi, yi] = clip_line_to_rect(x1, y1, x2, y2, x_left, x_right, y_bottom, y_top);
                if ~isempty(xi)
                    plot(xi, yi, 'k-', 'LineWidth', line_width, 'Clipping', 'on');
                end
            end
            
        case '\'  % 反斜线 (-45度)
            for offset = -h:spacing:w+h
                x1 = x_left + offset;
                y1 = y_top;
                x2 = x_left + offset + h;
                y2 = y_bottom;
                [xi, yi] = clip_line_to_rect(x1, y1, x2, y2, x_left, x_right, y_bottom, y_top);
                if ~isempty(xi)
                    plot(xi, yi, 'k-', 'LineWidth', line_width, 'Clipping', 'on');
                end
            end
            
        case '-'  % 水平线
            for y = y_bottom+spacing/2:spacing:y_top
                plot([x_left, x_right], [y, y], 'k-', 'LineWidth', line_width);
            end
            
        case '|'  % 垂直线
            for x = x_left+spacing*w/10:spacing*w/10:x_right
                plot([x, x], [y_bottom, y_top], 'k-', 'LineWidth', line_width);
            end
            
        case 'x'  % 交叉线
            add_hatch_to_bar(x_left, x_right, y_bottom, y_top, '/', spacing, line_width);
            add_hatch_to_bar(x_left, x_right, y_bottom, y_top, '\', spacing, line_width);
            
        case '.'  % 点阵
            for x = x_left+w/8:w/4:x_right
                for y = y_bottom+spacing/2:spacing:y_top
                    plot(x, y, 'k.', 'MarkerSize', 3);
                end
            end
    end
end

function [xi, yi] = clip_line_to_rect(x1, y1, x2, y2, x_min, x_max, y_min, y_max)
    % Cohen-Sutherland 线段剪裁算法简化版
    xi = []; yi = [];
    
    % 检查线段是否在矩形内
    if x1 < x_min && x2 < x_min, return; end
    if x1 > x_max && x2 > x_max, return; end
    if y1 < y_min && y2 < y_min, return; end
    if y1 > y_max && y2 > y_max, return; end
    
    % 计算斜率
    if abs(x2 - x1) < 1e-10
        % 垂直线
        xi = [x1, x2];
        yi = [max(y_min, min(y1, y2)), min(y_max, max(y1, y2))];
        return;
    end
    
    m = (y2 - y1) / (x2 - x1);
    b = y1 - m * x1;
    
    % 与四条边的交点
    points = [];
    
    % 左边 x = x_min
    y_at_xmin = m * x_min + b;
    if y_at_xmin >= y_min && y_at_xmin <= y_max
        points = [points; x_min, y_at_xmin];
    end
    
    % 右边 x = x_max
    y_at_xmax = m * x_max + b;
    if y_at_xmax >= y_min && y_at_xmax <= y_max
        points = [points; x_max, y_at_xmax];
    end
    
    % 下边 y = y_min
    if abs(m) > 1e-10
        x_at_ymin = (y_min - b) / m;
        if x_at_ymin >= x_min && x_at_ymin <= x_max
            points = [points; x_at_ymin, y_min];
        end
    end
    
    % 上边 y = y_max
    if abs(m) > 1e-10
        x_at_ymax = (y_max - b) / m;
        if x_at_ymax >= x_min && x_at_ymax <= x_max
            points = [points; x_at_ymax, y_max];
        end
    end
    
    % 去重并排序
    if size(points, 1) >= 2
        points = unique(points, 'rows');
        if size(points, 1) >= 2
            xi = points(1:2, 1)';
            yi = points(1:2, 2)';
        end
    end
end

function add_hatch_pattern(x_left, x_right, y_bottom, y_top, pattern, spacing, line_width)
    % 在指定矩形区域内添加填充图案
    % pattern: '/' 正斜线, '\' 反斜线, '-' 水平线, '|' 垂直线, 'x' 交叉线
    % spacing: 线条间距
    % line_width: 线条宽度
    
    if nargin < 6, spacing = 2; end
    if nargin < 7, line_width = 0.5; end
    
    w = x_right - x_left;
    h = y_top - y_bottom;
    
    switch pattern
        case '/'  % 正斜线
            % 从左下到右上的斜线
            num_lines = ceil((w + h) / spacing);
            for k = 0:num_lines
                y_start = y_bottom + k * spacing - w;
                y_end = y_start + w;
                
                % 裁剪到矩形范围内
                x1 = x_left;
                x2 = x_right;
                y1 = y_start;
                y2 = y_end;
                
                % 调整起点
                if y1 < y_bottom
                    x1 = x_left + (y_bottom - y1);
                    y1 = y_bottom;
                end
                if y1 > y_top
                    continue;
                end
                
                % 调整终点
                if y2 > y_top
                    x2 = x_right - (y2 - y_top);
                    y2 = y_top;
                end
                if y2 < y_bottom
                    continue;
                end
                
                if x1 < x_right && x2 > x_left && y1 < y_top && y2 > y_bottom
                    plot([x1, x2], [y1, y2], 'k-', 'LineWidth', line_width);
                end
            end
            
        case '\'  % 反斜线
            num_lines = ceil((w + h) / spacing);
            for k = 0:num_lines
                y_start = y_top - k * spacing + w;
                y_end = y_start - w;
                
                x1 = x_left;
                x2 = x_right;
                y1 = y_start;
                y2 = y_end;
                
                if y1 > y_top
                    x1 = x_left + (y1 - y_top);
                    y1 = y_top;
                end
                if y1 < y_bottom
                    continue;
                end
                
                if y2 < y_bottom
                    x2 = x_right - (y_bottom - y2);
                    y2 = y_bottom;
                end
                if y2 > y_top
                    continue;
                end
                
                if x1 < x_right && x2 > x_left
                    plot([x1, x2], [y1, y2], 'k-', 'LineWidth', line_width);
                end
            end
            
        case '-'  % 水平线
            y_vals = y_bottom:spacing:y_top;
            for y = y_vals
                plot([x_left, x_right], [y, y], 'k-', 'LineWidth', line_width);
            end
            
        case '|'  % 垂直线
            x_vals = x_left:spacing:x_right;
            for x = x_vals
                plot([x, x], [y_bottom, y_top], 'k-', 'LineWidth', line_width);
            end
            
        case 'x'  % 交叉线 (正斜线 + 反斜线)
            add_hatch_pattern(x_left, x_right, y_bottom, y_top, '/', spacing, line_width);
            add_hatch_pattern(x_left, x_right, y_bottom, y_top, '\', spacing, line_width);
    end
end

