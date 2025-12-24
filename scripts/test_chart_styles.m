%% 测试不同图表样式 - 孔径扩展数据可视化
% 对比：点状图、哑铃图、棒棒糖图、分组散点图
clear; clc; close all;

addpath('asset');

% 加载数据
data_folder = 'E:\code\matlab\rader_mimo\big_paper\yongjin_paper\images\chapter4\comprehensive_motion_array_test_20251209_191404';
load(fullfile(data_folder, 'experiment_results.mat'));

% 输出文件夹
output_folder = fullfile('validation_results', 'chart_style_test');
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

% 提取数据
aperture_data = results.aperture;  % [阵列 × 运动模式]
array_names = results.array_names;
motion_names = results.motion_names;

num_arrays = length(array_names);
num_motions = length(motion_names);

% 设置中文字体
set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultAxesFontSize', 11);

% 颜色和标记配置
colors = [
    0.2, 0.2, 0.2;    % 静态 - 深灰
    0.85, 0.33, 0.1;  % x平移 - 橙
    0.0, 0.45, 0.74;  % y平移 - 蓝
    0.47, 0.67, 0.19; % 旋转 - 绿
    0.49, 0.18, 0.56; % 平移+旋转 - 紫
];
markers = {'o', 's', 'd', '^', 'v'};

fprintf('生成不同图表样式对比...\n\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案1: 点状图 (Cleveland Dot Plot)
%% ═══════════════════════════════════════════════════════════════════
fprintf('1. 点状图 (Cleveland Dot Plot)\n');

fig1 = figure('Position', [100, 100, 800, 500], 'Color', 'white');
hold on;

% Y轴是阵列名称 (从下到上: 1到num_arrays)
y_positions = 1:num_arrays;

for mot_idx = 1:num_motions
    x_vals = aperture_data(:, mot_idx);
    % 水平方向稍微错开，避免重叠
    y_offset = (mot_idx - 3) * 0.12;
    
    scatter(x_vals, y_positions + y_offset, 100, colors(mot_idx, :), ...
        markers{mot_idx}, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1, ...
        'DisplayName', motion_names{mot_idx});
end

% 添加水平参考线
for i = 1:num_arrays
    plot([0, 35], [y_positions(i), y_positions(i)], 'k:', 'LineWidth', 0.5, 'HandleVisibility', 'off');
end

hold off;

set(gca, 'YTick', y_positions, 'YTickLabel', array_names);
xlabel('合成孔径 (λ)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('阵列配置', 'FontWeight', 'bold', 'FontSize', 12);
title('方案1: 点状图 (Cleveland Dot Plot)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 10);
grid on;
box on;
xlim([0, 35]);
ylim([0.5, num_arrays + 0.5]);

saveas(fig1, fullfile(output_folder, '方案1_点状图.png'));
fprintf('   已保存: 方案1_点状图.png\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案2: 哑铃图 (Dumbbell Chart) - 静态 vs 最佳运动
%% ═══════════════════════════════════════════════════════════════════
fprintf('2. 哑铃图 (Dumbbell Chart)\n');

fig2 = figure('Position', [100, 100, 800, 500], 'Color', 'white');
hold on;

y_positions = 1:num_arrays;

for i = 1:num_arrays
    static_val = aperture_data(i, 1);  % 静态
    % 找最佳运动模式（排除静态和旋转）
    motion_vals = aperture_data(i, [2, 3, 5]);  % x平移, y平移, 平移+旋转
    [best_val, best_idx] = max(motion_vals);
    best_motion_idx = [2, 3, 5];
    actual_best_idx = best_motion_idx(best_idx);
    
    y = y_positions(i);
    
    % 画连接线
    plot([static_val, best_val], [y, y], 'k-', 'LineWidth', 2, 'HandleVisibility', 'off');
    
    % 画静态点（空心圆）
    scatter(static_val, y, 150, colors(1, :), 'o', 'LineWidth', 2, ...
        'MarkerEdgeColor', colors(1, :), 'HandleVisibility', 'off');
    
    % 画最佳运动点（实心）
    scatter(best_val, y, 150, colors(actual_best_idx, :), markers{actual_best_idx}, ...
        'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1, 'HandleVisibility', 'off');
    
    % 标注改善倍数
    improvement = best_val / max(static_val, 0.1);
    text(best_val + 1, y, sprintf('%.1f×', improvement), 'FontSize', 9, 'FontWeight', 'bold');
end

% 图例（手动创建）
scatter(NaN, NaN, 150, colors(1, :), 'o', 'LineWidth', 2, 'DisplayName', '静态');
scatter(NaN, NaN, 150, colors(3, :), 'd', 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', '最佳运动');

hold off;

set(gca, 'YTick', y_positions, 'YTickLabel', array_names);
xlabel('合成孔径 (λ)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('阵列配置', 'FontWeight', 'bold', 'FontSize', 12);
title('方案2: 哑铃图 - 静态 vs 最佳运动模式', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 10);
grid on;
box on;
xlim([0, 35]);
ylim([0.5, num_arrays + 0.5]);

saveas(fig2, fullfile(output_folder, '方案2_哑铃图.png'));
fprintf('   已保存: 方案2_哑铃图.png\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案3: 棒棒糖图 (Lollipop Chart)
%% ═══════════════════════════════════════════════════════════════════
fprintf('3. 棒棒糖图 (Lollipop Chart)\n');

fig3 = figure('Position', [100, 100, 900, 500], 'Color', 'white');
hold on;

% X轴是阵列，不同运动模式用不同标记
x_positions = 1:num_arrays;
group_width = 0.7;
offset_step = group_width / (num_motions - 1);

for mot_idx = 1:num_motions
    x_offset = (mot_idx - (num_motions + 1) / 2) * offset_step * 0.8;
    
    for arr_idx = 1:num_arrays
        x = x_positions(arr_idx) + x_offset;
        y = aperture_data(arr_idx, mot_idx);
        
        % 画细线（棒）
        plot([x, x], [0, y], '-', 'Color', colors(mot_idx, :), 'LineWidth', 1.5, 'HandleVisibility', 'off');
        
        % 画点（糖）
        if arr_idx == 1
            scatter(x, y, 100, colors(mot_idx, :), markers{mot_idx}, 'filled', ...
                'MarkerEdgeColor', 'k', 'LineWidth', 1, 'DisplayName', motion_names{mot_idx});
        else
            scatter(x, y, 100, colors(mot_idx, :), markers{mot_idx}, 'filled', ...
                'MarkerEdgeColor', 'k', 'LineWidth', 1, 'HandleVisibility', 'off');
        end
    end
end

hold off;

set(gca, 'XTick', x_positions, 'XTickLabel', array_names);
xtickangle(30);
xlabel('阵列配置', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('合成孔径 (λ)', 'FontWeight', 'bold', 'FontSize', 12);
title('方案3: 棒棒糖图 (Lollipop Chart)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 10);
grid on;
box on;
xlim([0.5, num_arrays + 0.5]);
ylim([0, 35]);

saveas(fig3, fullfile(output_folder, '方案3_棒棒糖图.png'));
fprintf('   已保存: 方案3_棒棒糖图.png\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案4: 分组散点图 (Grouped Scatter)
%% ═══════════════════════════════════════════════════════════════════
fprintf('4. 分组散点图 (Grouped Scatter)\n');

fig4 = figure('Position', [100, 100, 900, 500], 'Color', 'white');
hold on;

x_positions = 1:num_arrays;
jitter_width = 0.3;

for mot_idx = 1:num_motions
    % 添加一点随机抖动避免重叠
    x_jitter = (mot_idx - (num_motions + 1) / 2) * jitter_width / 2;
    x_vals = x_positions + x_jitter;
    y_vals = aperture_data(:, mot_idx);
    
    scatter(x_vals, y_vals, 120, colors(mot_idx, :), markers{mot_idx}, ...
        'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1, ...
        'DisplayName', motion_names{mot_idx});
end

hold off;

set(gca, 'XTick', x_positions, 'XTickLabel', array_names);
xtickangle(30);
xlabel('阵列配置', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('合成孔径 (λ)', 'FontWeight', 'bold', 'FontSize', 12);
title('方案4: 分组散点图', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 10);
grid on;
box on;
xlim([0.5, num_arrays + 0.5]);
ylim([0, 35]);

saveas(fig4, fullfile(output_folder, '方案4_分组散点图.png'));
fprintf('   已保存: 方案4_分组散点图.png\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案5: 热力图 (Heatmap) - 带数值标注
%% ═══════════════════════════════════════════════════════════════════
fprintf('5. 热力图 (Heatmap)\n');

fig5 = figure('Position', [100, 100, 700, 500], 'Color', 'white');

% 转置数据，让运动模式在X轴
data_for_heatmap = aperture_data';

imagesc(data_for_heatmap);
colormap(flipud(gray));  % 灰度色图，值越大越深
cb = colorbar;
cb.Label.String = '合成孔径 (λ)';

% 添加数值标注
for i = 1:num_motions
    for j = 1:num_arrays
        val = data_for_heatmap(i, j);
        if val > 15
            text_color = 'w';
        else
            text_color = 'k';
        end
        text(j, i, sprintf('%.1f', val), 'HorizontalAlignment', 'center', ...
            'FontSize', 10, 'FontWeight', 'bold', 'Color', text_color);
    end
end

set(gca, 'XTick', 1:num_arrays, 'XTickLabel', array_names);
set(gca, 'YTick', 1:num_motions, 'YTickLabel', motion_names);
xtickangle(30);
xlabel('阵列配置', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('运动模式', 'FontWeight', 'bold', 'FontSize', 12);
title('方案5: 热力图 (带数值)', 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig5, fullfile(output_folder, '方案5_热力图.png'));
fprintf('   已保存: 方案5_热力图.png\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案6: 水平条形图 (Horizontal Bar) - 分面
%% ═══════════════════════════════════════════════════════════════════
fprintf('6. 分面水平条形图\n');

fig6 = figure('Position', [100, 100, 1000, 600], 'Color', 'white');

% 选择4个代表性阵列
selected = [1, 2, 6, 8];
selected_names = {'ULA-8', 'URA-3×3', '圆阵-8', 'Y阵列'};

for sub_idx = 1:4
    subplot(2, 2, sub_idx);
    arr_idx = selected(sub_idx);
    
    y_pos = 1:num_motions;
    vals = aperture_data(arr_idx, :);
    
    for mot_idx = 1:num_motions
        barh(y_pos(mot_idx), vals(mot_idx), 0.6, 'FaceColor', colors(mot_idx, :), ...
            'EdgeColor', 'k', 'LineWidth', 1);
        hold on;
        % 数值标注
        text(vals(mot_idx) + 0.5, y_pos(mot_idx), sprintf('%.1fλ', vals(mot_idx)), ...
            'FontSize', 9, 'VerticalAlignment', 'middle');
    end
    hold off;
    
    set(gca, 'YTick', y_pos, 'YTickLabel', motion_names);
    xlabel('孔径 (λ)', 'FontWeight', 'bold');
    title(selected_names{sub_idx}, 'FontSize', 12, 'FontWeight', 'bold');
    xlim([0, 35]);
    grid on;
    box on;
end

sgtitle('方案6: 分面水平条形图', 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig6, fullfile(output_folder, '方案6_分面水平条形图.png'));
fprintf('   已保存: 方案6_分面水平条形图.png\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案7: 整合分组条形图 - 所有阵列在一张图，下方标注运动模式
%% ═══════════════════════════════════════════════════════════════════
fprintf('7. 整合分组条形图 (运动模式标注在下方)\n');

fig7 = figure('Position', [100, 100, 1200, 500], 'Color', 'white');
hold on;

% 计算位置
group_gap = 1.5;  % 组间距
bar_width = 0.7;
bar_gap = 0.1;

% 每个阵列一组，组内每个运动模式一个条
x_start = 1;
x_ticks = [];
x_labels = {};

for arr_idx = 1:num_arrays
    x_group_start = x_start;
    
    for mot_idx = 1:num_motions
        x = x_start + (mot_idx - 1) * (bar_width + bar_gap);
        y = aperture_data(arr_idx, mot_idx);
        
        % 画条形
        bar(x, y, bar_width, 'FaceColor', colors(mot_idx, :), ...
            'EdgeColor', 'k', 'LineWidth', 1);
        
        % 数值标注（条形顶部）
        text(x, y + 0.8, sprintf('%.1f', y), 'HorizontalAlignment', 'center', ...
            'FontSize', 8, 'FontWeight', 'bold');
        
        % 运动模式标注（条形底部）
        % 使用简写
        short_names = {'静', 'x移', 'y移', '转', '移转'};
        text(x, -1.5, short_names{mot_idx}, 'HorizontalAlignment', 'center', ...
            'FontSize', 8, 'Rotation', 0);
    end
    
    % 阵列名称标注（组的中心位置）
    x_center = x_group_start + (num_motions - 1) * (bar_width + bar_gap) / 2;
    x_ticks = [x_ticks, x_center];
    x_labels{end+1} = array_names{arr_idx};
    
    % 下一组起始位置
    x_start = x_start + num_motions * (bar_width + bar_gap) + group_gap;
end

hold off;

set(gca, 'XTick', x_ticks, 'XTickLabel', x_labels);
xtickangle(30);
ylabel('合成孔径 (λ)', 'FontWeight', 'bold', 'FontSize', 12);
title('方案7: 整合分组条形图', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;
ylim([-3, 35]);
xlim([0, x_start]);

% 添加图例 - 使用patch创建
legend_handles = gobjects(num_motions, 1);
for i = 1:num_motions
    legend_handles(i) = patch(NaN, NaN, colors(i, :), 'EdgeColor', 'k', 'LineWidth', 1);
end
legend(legend_handles, motion_names, 'Location', 'northwest', 'FontSize', 9);

saveas(fig7, fullfile(output_folder, '方案7_整合分组条形图.png'));
fprintf('   已保存: 方案7_整合分组条形图.png\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案8: 紧凑分组条形图 - 选择代表性阵列
%% ═══════════════════════════════════════════════════════════════════
fprintf('8. 紧凑分组条形图 (选择代表性阵列)\n');

fig8 = figure('Position', [100, 100, 900, 450], 'Color', 'white');
hold on;

% 选择4个代表性阵列
selected = [1, 2, 6, 8];
selected_names = {'ULA-8', 'URA-3×3', '圆阵-8', 'Y阵列'};
num_selected = length(selected);

group_gap = 2;
bar_width = 0.8;
bar_gap = 0.15;

x_start = 1;
x_ticks = [];

for sel_idx = 1:num_selected
    arr_idx = selected(sel_idx);
    x_group_start = x_start;
    
    for mot_idx = 1:num_motions
        x = x_start + (mot_idx - 1) * (bar_width + bar_gap);
        y = aperture_data(arr_idx, mot_idx);
        
        % 画条形
        bar(x, y, bar_width, 'FaceColor', colors(mot_idx, :), ...
            'EdgeColor', 'k', 'LineWidth', 1.2);
        
        % 数值标注
        text(x, y + 1, sprintf('%.1f', y), 'HorizontalAlignment', 'center', ...
            'FontSize', 9, 'FontWeight', 'bold');
    end
    
    % 阵列名称
    x_center = x_group_start + (num_motions - 1) * (bar_width + bar_gap) / 2;
    x_ticks = [x_ticks, x_center];
    
    x_start = x_start + num_motions * (bar_width + bar_gap) + group_gap;
end

hold off;

set(gca, 'XTick', x_ticks, 'XTickLabel', selected_names);
ylabel('合成孔径 (λ)', 'FontWeight', 'bold', 'FontSize', 12);
xlabel('阵列配置', 'FontWeight', 'bold', 'FontSize', 12);
title('方案8: 紧凑分组条形图', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;
ylim([0, 35]);
xlim([0, x_start - group_gap + 1]);

% 图例 - 使用patch创建
legend_handles = gobjects(num_motions, 1);
for i = 1:num_motions
    legend_handles(i) = patch(NaN, NaN, colors(i, :), 'EdgeColor', 'k', 'LineWidth', 1);
end
legend(legend_handles, motion_names, 'Location', 'northwest', 'FontSize', 10);

saveas(fig8, fullfile(output_folder, '方案8_紧凑分组条形图.png'));
fprintf('   已保存: 方案8_紧凑分组条形图.png\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案9: 堆叠+并列混合图 - 静态vs运动对比
%% ═══════════════════════════════════════════════════════════════════
fprintf('9. 静态vs运动对比条形图\n');

fig9 = figure('Position', [100, 100, 800, 450], 'Color', 'white');

% 选择代表性阵列
selected = [1, 2, 6, 8];
selected_names = {'ULA-8', 'URA-3×3', '圆阵-8', 'Y阵列'};
num_selected = length(selected);

% 准备数据: 静态 vs 最佳运动
static_vals = zeros(num_selected, 1);
best_motion_vals = zeros(num_selected, 1);
best_motion_names = cell(num_selected, 1);

for i = 1:num_selected
    arr_idx = selected(i);
    static_vals(i) = aperture_data(arr_idx, 1);
    [best_val, best_idx] = max(aperture_data(arr_idx, 2:end));
    best_motion_vals(i) = best_val;
    best_motion_names{i} = motion_names{best_idx + 1};
end

x = 1:num_selected;
bar_w = 0.35;

hold on;
b1 = bar(x - bar_w/2, static_vals, bar_w, 'FaceColor', colors(1, :), 'EdgeColor', 'k', 'LineWidth', 1.5);
b2 = bar(x + bar_w/2, best_motion_vals, bar_w, 'FaceColor', colors(3, :), 'EdgeColor', 'k', 'LineWidth', 1.5);

% 数值标注
for i = 1:num_selected
    text(x(i) - bar_w/2, static_vals(i) + 1, sprintf('%.1fλ', static_vals(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    text(x(i) + bar_w/2, best_motion_vals(i) + 1, sprintf('%.1fλ', best_motion_vals(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    
    % 改善倍数
    improvement = best_motion_vals(i) / max(static_vals(i), 0.1);
    text(x(i), 33, sprintf('↑%.0f×', improvement), 'HorizontalAlignment', 'center', ...
        'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.1, 0.5, 0.1]);
end

hold off;

set(gca, 'XTick', x, 'XTickLabel', selected_names);
ylabel('合成孔径 (λ)', 'FontWeight', 'bold', 'FontSize', 12);
xlabel('阵列配置', 'FontWeight', 'bold', 'FontSize', 12);
title('方案9: 静态 vs 最佳运动 孔径对比', 'FontSize', 14, 'FontWeight', 'bold');
legend([b1, b2], {'静态', '最佳运动'}, 'Location', 'northwest', 'FontSize', 11);
grid on;
box on;
ylim([0, 38]);

saveas(fig9, fullfile(output_folder, '方案9_静态vs运动对比.png'));
fprintf('   已保存: 方案9_静态vs运动对比.png\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案10: 竖向水平条形图 - 每个条形旁标注名称（不依赖图例）
%% ═══════════════════════════════════════════════════════════════════
fprintf('10. 竖向水平条形图 (直接标注，不依赖图例)\n');

% 选择代表性阵列
selected = [1, 2, 6, 8];
selected_names = {'ULA-8', 'URA-3×3', '圆阵-8', 'Y阵列'};
num_selected = length(selected);

fig10 = figure('Position', [100, 100, 900, 700], 'Color', 'white');

% 每个阵列占一行，运动模式竖向排列
bar_height = 0.7;
group_gap = 1.5;  % 阵列组之间的间距

y_current = 0;
y_group_centers = [];

for sel_idx = num_selected:-1:1  % 从下往上画
    arr_idx = selected(sel_idx);
    y_group_start = y_current;
    
    for mot_idx = 1:num_motions
        y = y_current;
        x_val = aperture_data(arr_idx, mot_idx);
        
        % 画水平条形
        barh(y, x_val, bar_height, 'FaceColor', colors(mot_idx, :), ...
            'EdgeColor', 'k', 'LineWidth', 1.2);
        hold on;
        
        % 左侧标注运动模式名称
        text(-0.5, y, motion_names{mot_idx}, 'HorizontalAlignment', 'right', ...
            'FontSize', 10, 'FontWeight', 'normal');
        
        % 右侧标注数值
        text(x_val + 0.5, y, sprintf('%.1fλ', x_val), 'HorizontalAlignment', 'left', ...
            'FontSize', 10, 'FontWeight', 'bold');
        
        y_current = y_current + bar_height + 0.1;
    end
    
    % 记录组中心位置（用于标注阵列名称）
    y_group_center = (y_group_start + y_current - bar_height - 0.1) / 2 + y_group_start / 2;
    y_group_centers = [y_group_center, y_group_centers];
    
    % 阵列分隔线
    if sel_idx > 1
        y_current = y_current + group_gap;
        plot([0, 35], [y_current - group_gap/2, y_current - group_gap/2], 'k--', ...
            'LineWidth', 0.5, 'HandleVisibility', 'off');
    end
end

hold off;

% 设置坐标轴
set(gca, 'YTick', []);  % 不显示Y轴刻度
xlabel('合成孔径 (λ)', 'FontWeight', 'bold', 'FontSize', 12);
title('方案10: 竖向排列 - 直接标注运动模式', 'FontSize', 14, 'FontWeight', 'bold');
xlim([-8, 35]);
ylim([-0.5, y_current]);
grid on;
box on;

% 在右侧标注阵列名称
for i = 1:num_selected
    % 计算每组的中心Y位置
    if i == 1
        y_center = (num_motions - 1) * (bar_height + 0.1) / 2;
    else
        y_center = y_group_centers(i);
    end
end

% 重新计算并标注阵列名称
y_pos = 0;
for sel_idx = num_selected:-1:1
    y_center = y_pos + (num_motions - 1) * (bar_height + 0.1) / 2;
    text(33, y_center, selected_names{num_selected - sel_idx + 1}, ...
        'HorizontalAlignment', 'left', 'FontSize', 12, 'FontWeight', 'bold', ...
        'BackgroundColor', 'w', 'EdgeColor', 'k');
    y_pos = y_pos + num_motions * (bar_height + 0.1) + group_gap;
end

saveas(fig10, fullfile(output_folder, '方案10_竖向水平条形图.png'));
fprintf('   已保存: 方案10_竖向水平条形图.png\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案11: 更紧凑的竖向布局 - 所有阵列
%% ═══════════════════════════════════════════════════════════════════
fprintf('11. 紧凑竖向布局 (所有阵列)\n');

fig11 = figure('Position', [100, 100, 1100, 800], 'Color', 'white');

bar_height = 0.5;
group_gap = 1.0;

y_current = 0;

for arr_idx = num_arrays:-1:1
    for mot_idx = 1:num_motions
        y = y_current;
        x_val = aperture_data(arr_idx, mot_idx);
        
        barh(y, x_val, bar_height, 'FaceColor', colors(mot_idx, :), ...
            'EdgeColor', 'k', 'LineWidth', 1);
        hold on;
        
        % 左侧标注（简写）
        short_motion = {'静态', 'x移', 'y移', '旋转', '移+转'};
        text(-0.3, y, short_motion{mot_idx}, 'HorizontalAlignment', 'right', ...
            'FontSize', 8);
        
        % 数值标注
        text(x_val + 0.3, y, sprintf('%.1f', x_val), 'HorizontalAlignment', 'left', ...
            'FontSize', 8, 'FontWeight', 'bold');
        
        y_current = y_current + bar_height + 0.05;
    end
    
    % 阵列名称标注
    y_center = y_current - num_motions * (bar_height + 0.05) / 2 - 0.1;
    text(32, y_center, array_names{arr_idx}, 'HorizontalAlignment', 'left', ...
        'FontSize', 9, 'FontWeight', 'bold');
    
    if arr_idx > 1
        y_current = y_current + group_gap;
        plot([0, 35], [y_current - group_gap/2, y_current - group_gap/2], ...
            'k:', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    end
end

hold off;

set(gca, 'YTick', []);
xlabel('合成孔径 (λ)', 'FontWeight', 'bold', 'FontSize', 12);
title('方案11: 所有阵列竖向排列', 'FontSize', 14, 'FontWeight', 'bold');
xlim([-5, 38]);
ylim([-0.5, y_current]);
grid on;
box on;

saveas(fig11, fullfile(output_folder, '方案11_紧凑竖向布局.png'));
fprintf('   已保存: 方案11_紧凑竖向布局.png\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案12: 宽松竖向布局 - 大字体版本
%% ═══════════════════════════════════════════════════════════════════
fprintf('12. 宽松竖向布局 (大字体)\n');

fig12 = figure('Position', [100, 50, 1200, 1000], 'Color', 'white');

bar_height = 0.7;      % 增大条形高度
bar_gap = 0.15;        % 条形间距
group_gap = 1.8;       % 组间距增大

y_current = 0;

for arr_idx = num_arrays:-1:1
    for mot_idx = 1:num_motions
        y = y_current;
        x_val = aperture_data(arr_idx, mot_idx);
        
        barh(y, x_val, bar_height, 'FaceColor', colors(mot_idx, :), ...
            'EdgeColor', 'k', 'LineWidth', 1.5);
        hold on;
        
        % 左侧标注运动模式（大字体）
        text(-0.5, y, motion_names{mot_idx}, 'HorizontalAlignment', 'right', ...
            'FontSize', 11, 'FontWeight', 'normal');
        
        % 右侧数值标注（大字体加粗）
        text(x_val + 0.5, y, sprintf('%.1fλ', x_val), 'HorizontalAlignment', 'left', ...
            'FontSize', 11, 'FontWeight', 'bold');
        
        y_current = y_current + bar_height + bar_gap;
    end
    
    % 阵列名称标注（右侧，大字体）
    y_center = y_current - num_motions * (bar_height + bar_gap) / 2 - bar_gap;
    text(34, y_center, array_names{arr_idx}, 'HorizontalAlignment', 'left', ...
        'FontSize', 13, 'FontWeight', 'bold', 'BackgroundColor', [0.95, 0.95, 0.95]);
    
    if arr_idx > 1
        y_current = y_current + group_gap;
        % 分隔线
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

saveas(fig12, fullfile(output_folder, '方案12_宽松竖向布局_大字体.png'));
fprintf('   已保存: 方案12_宽松竖向布局_大字体.png\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案13: 精选阵列版本 - 4个代表性阵列，更大字体
%% ═══════════════════════════════════════════════════════════════════
fprintf('13. 精选阵列版本 (4个代表性阵列)\n');

fig13 = figure('Position', [100, 100, 900, 700], 'Color', 'white');

selected = [1, 2, 6, 8];
selected_names = {'ULA-8', 'URA-3×3', '圆阵-8', 'Y阵列'};
num_selected = length(selected);

bar_height = 0.8;
bar_gap = 0.2;
group_gap = 2.5;

y_current = 0;

for sel_idx = num_selected:-1:1
    arr_idx = selected(sel_idx);
    
    for mot_idx = 1:num_motions
        y = y_current;
        x_val = aperture_data(arr_idx, mot_idx);
        
        barh(y, x_val, bar_height, 'FaceColor', colors(mot_idx, :), ...
            'EdgeColor', 'k', 'LineWidth', 1.5);
        hold on;
        
        % 左侧标注
        text(-0.8, y, motion_names{mot_idx}, 'HorizontalAlignment', 'right', ...
            'FontSize', 12);
        
        % 数值标注
        text(x_val + 0.5, y, sprintf('%.1fλ', x_val), 'HorizontalAlignment', 'left', ...
            'FontSize', 12, 'FontWeight', 'bold');
        
        y_current = y_current + bar_height + bar_gap;
    end
    
    % 阵列名称
    y_center = y_current - num_motions * (bar_height + bar_gap) / 2 - bar_gap/2;
    text(33, y_center, selected_names{num_selected - sel_idx + 1}, ...
        'HorizontalAlignment', 'left', 'FontSize', 14, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.9, 0.9, 0.9], 'EdgeColor', 'k', 'Margin', 3);
    
    if sel_idx > 1
        y_current = y_current + group_gap;
        plot([0, 32], [y_current - group_gap/2, y_current - group_gap/2], ...
            'k-', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    end
end

hold off;

set(gca, 'YTick', []);
xlabel('合成孔径 (λ)', 'FontWeight', 'bold', 'FontSize', 14);
title('代表性阵列的合成孔径对比', 'FontSize', 16, 'FontWeight', 'bold');
xlim([-10, 40]);
ylim([-1, y_current + 1]);
grid on;
box on;
set(gca, 'FontSize', 12);

saveas(fig13, fullfile(output_folder, '方案13_精选阵列_大字体.png'));
fprintf('   已保存: 方案13_精选阵列_大字体.png\n');

%% 完成
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('所有图表已生成！\n');
fprintf('输出目录: %s\n', output_folder);
fprintf('═══════════════════════════════════════════════════════════════════\n');

