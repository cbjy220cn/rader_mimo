%% 测试二值化数据的不同展示方式 - 分辨能力对比
clear; clc; close all;

% 加载数据
data_folder = 'E:\code\matlab\rader_mimo\big_paper\yongjin_paper\images\chapter4\experiment_dual_target_1d_20251210_014618';
load(fullfile(data_folder, 'experiment_results.mat'));

% 输出文件夹
output_folder = fullfile('validation_results', 'binary_chart_test');
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

% 设置中文字体
set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultAxesFontSize', 11);

angle_separations = results.separations;
static_resolved = results.static_resolved;
motion_resolved = results.motion_resolved;
num_seps = length(angle_separations);

fprintf('数据概览:\n');
fprintf('角度间隔: %s\n', mat2str(angle_separations));
fprintf('静态分辨: %s\n', mat2str(static_resolved));
fprintf('运动分辨: %s\n', mat2str(motion_resolved));
fprintf('\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案1: 原始柱状图
%% ═══════════════════════════════════════════════════════════════════
fprintf('方案1: 原始柱状图\n');

fig1 = figure('Position', [100, 100, 500, 350], 'Color', 'white');

bar_data = [static_resolved; motion_resolved]';
b = bar(bar_data, 'grouped');
b(1).FaceColor = [0.4, 0.4, 0.4];
b(1).EdgeColor = 'k';
b(1).LineWidth = 1.5;
b(2).FaceColor = [0.0, 0.45, 0.74];
b(2).EdgeColor = 'k';
b(2).LineWidth = 1.5;

set(gca, 'XTick', 1:num_seps, 'XTickLabel', arrayfun(@(x) sprintf('%d°', x), angle_separations, 'UniformOutput', false));
xlabel('双目标角度间隔', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('分辨成功 (1=是, 0=否)', 'FontWeight', 'bold', 'FontSize', 12);
title('方案1: 柱状图', 'FontSize', 14, 'FontWeight', 'bold');
legend({'静态阵列', '运动阵列'}, 'Location', 'southeast', 'FontSize', 10);
grid on;
box on;
ylim([0, 1.3]);

saveas(fig1, fullfile(output_folder, '方案1_柱状图.png'));
fprintf('  已保存\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案2: 标记对比图 (色块+符号)
%% ═══════════════════════════════════════════════════════════════════
fprintf('方案2: 标记对比图\n');

fig2 = figure('Position', [100, 100, 700, 280], 'Color', 'white');
hold on;

box_width = 0.8;
box_height = 0.35;
color_yes = [0.3, 0.7, 0.3];
color_no = [0.8, 0.3, 0.3];
y_static = 1.5;
y_motion = 0.5;

for i = 1:num_seps
    x = i;
    
    % 静态
    if static_resolved(i)
        rectangle('Position', [x-box_width/2, y_static-box_height/2, box_width, box_height], ...
            'FaceColor', color_yes, 'EdgeColor', 'k', 'LineWidth', 1.5);
        text(x, y_static, '✓', 'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold');
    else
        rectangle('Position', [x-box_width/2, y_static-box_height/2, box_width, box_height], ...
            'FaceColor', color_no, 'EdgeColor', 'k', 'LineWidth', 1.5);
        text(x, y_static, '✗', 'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'w');
    end
    
    % 运动
    if motion_resolved(i)
        rectangle('Position', [x-box_width/2, y_motion-box_height/2, box_width, box_height], ...
            'FaceColor', color_yes, 'EdgeColor', 'k', 'LineWidth', 1.5);
        text(x, y_motion, '✓', 'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold');
    else
        rectangle('Position', [x-box_width/2, y_motion-box_height/2, box_width, box_height], ...
            'FaceColor', color_no, 'EdgeColor', 'k', 'LineWidth', 1.5);
        text(x, y_motion, '✗', 'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'w');
    end
end

text(0.3, y_static, '静态阵列', 'HorizontalAlignment', 'right', 'FontSize', 12, 'FontWeight', 'bold');
text(0.3, y_motion, '运动阵列', 'HorizontalAlignment', 'right', 'FontSize', 12, 'FontWeight', 'bold');

for i = 1:num_seps
    text(i, 0, sprintf('%d°', angle_separations(i)), 'HorizontalAlignment', 'center', 'FontSize', 11);
end

hold off;
xlim([0, num_seps + 1]);
ylim([-0.2, 2.1]);
set(gca, 'XTick', [], 'YTick', []);
title('方案2: 标记对比图', 'FontSize', 14, 'FontWeight', 'bold');
axis off;

saveas(fig2, fullfile(output_folder, '方案2_标记对比图.png'));
fprintf('  已保存\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案3: 热力图风格
%% ═══════════════════════════════════════════════════════════════════
fprintf('方案3: 热力图风格\n');

fig3 = figure('Position', [100, 100, 600, 250], 'Color', 'white');

data_matrix = [static_resolved; motion_resolved];
imagesc(data_matrix);
colormap([0.85, 0.3, 0.3; 0.3, 0.75, 0.3]);  % 红-绿

set(gca, 'XTick', 1:num_seps, 'XTickLabel', arrayfun(@(x) sprintf('%d°', x), angle_separations, 'UniformOutput', false));
set(gca, 'YTick', 1:2, 'YTickLabel', {'静态阵列', '运动阵列'});
xlabel('角度间隔', 'FontWeight', 'bold', 'FontSize', 12);
title('方案3: 热力图', 'FontSize', 14, 'FontWeight', 'bold');

% 添加文字标注
for i = 1:2
    for j = 1:num_seps
        if data_matrix(i,j) == 1
            text(j, i, '✓', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
        else
            text(j, i, '✗', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'w');
        end
    end
end

saveas(fig3, fullfile(output_folder, '方案3_热力图.png'));
fprintf('  已保存\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案4: 阶梯线图
%% ═══════════════════════════════════════════════════════════════════
fprintf('方案4: 阶梯线图\n');

fig4 = figure('Position', [100, 100, 550, 350], 'Color', 'white');
hold on;

% 找到最小可分辨角度
static_min_idx = find(static_resolved == 1, 1);
motion_min_idx = find(motion_resolved == 1, 1);

if isempty(static_min_idx), static_min_idx = num_seps + 1; end
if isempty(motion_min_idx), motion_min_idx = num_seps + 1; end

% 画阶梯线
stairs(0.5:num_seps+0.5, [static_resolved, static_resolved(end)], '-', ...
    'Color', [0.4, 0.4, 0.4], 'LineWidth', 3, 'DisplayName', '静态阵列');
stairs(0.5:num_seps+0.5, [motion_resolved, motion_resolved(end)], '-', ...
    'Color', [0.0, 0.45, 0.74], 'LineWidth', 3, 'DisplayName', '运动阵列');

% 标记最小可分辨角度
if static_min_idx <= num_seps
    plot(static_min_idx, 1, 'o', 'MarkerSize', 12, 'MarkerFaceColor', [0.4,0.4,0.4], ...
        'MarkerEdgeColor', 'k', 'LineWidth', 2);
    text(static_min_idx, 1.15, sprintf('%d°', angle_separations(static_min_idx)), ...
        'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
end
if motion_min_idx <= num_seps
    plot(motion_min_idx, 1, 's', 'MarkerSize', 12, 'MarkerFaceColor', [0.0,0.45,0.74], ...
        'MarkerEdgeColor', 'k', 'LineWidth', 2);
    text(motion_min_idx, 0.85, sprintf('%d°', angle_separations(motion_min_idx)), ...
        'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
end

hold off;

set(gca, 'XTick', 1:num_seps, 'XTickLabel', arrayfun(@(x) sprintf('%d°', x), angle_separations, 'UniformOutput', false));
xlabel('角度间隔', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('可分辨', 'FontWeight', 'bold', 'FontSize', 12);
title('方案4: 阶梯线图', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 10);
ylim([-0.1, 1.3]);
xlim([0.5, num_seps + 0.5]);
set(gca, 'YTick', [0, 1], 'YTickLabel', {'否', '是'});
grid on;
box on;

saveas(fig4, fullfile(output_folder, '方案4_阶梯线图.png'));
fprintf('  已保存\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案5: 散点标记图
%% ═══════════════════════════════════════════════════════════════════
fprintf('方案5: 散点标记图\n');

fig5 = figure('Position', [100, 100, 550, 300], 'Color', 'white');
hold on;

y_static = 2;
y_motion = 1;
marker_size = 200;

for i = 1:num_seps
    % 静态
    if static_resolved(i)
        scatter(i, y_static, marker_size, 'o', 'filled', 'MarkerFaceColor', [0.3, 0.7, 0.3], ...
            'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    else
        scatter(i, y_static, marker_size, 'o', 'MarkerEdgeColor', [0.7, 0.3, 0.3], 'LineWidth', 2.5);
    end
    
    % 运动
    if motion_resolved(i)
        scatter(i, y_motion, marker_size, 's', 'filled', 'MarkerFaceColor', [0.3, 0.7, 0.3], ...
            'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    else
        scatter(i, y_motion, marker_size, 's', 'MarkerEdgeColor', [0.7, 0.3, 0.3], 'LineWidth', 2.5);
    end
end

hold off;

set(gca, 'XTick', 1:num_seps, 'XTickLabel', arrayfun(@(x) sprintf('%d°', x), angle_separations, 'UniformOutput', false));
set(gca, 'YTick', [y_motion, y_static], 'YTickLabel', {'运动阵列', '静态阵列'});
xlabel('角度间隔', 'FontWeight', 'bold', 'FontSize', 12);
title('方案5: 散点标记图 (实心=可分辨, 空心=不可分辨)', 'FontSize', 13, 'FontWeight', 'bold');
xlim([0.5, num_seps + 0.5]);
ylim([0.5, 2.5]);
grid on;
box on;

saveas(fig5, fullfile(output_folder, '方案5_散点标记图.png'));
fprintf('  已保存\n');

%% ═══════════════════════════════════════════════════════════════════
%  方案6: 简洁表格风格
%% ═══════════════════════════════════════════════════════════════════
fprintf('方案6: 简洁表格风格\n');

fig6 = figure('Position', [100, 100, 650, 200], 'Color', 'white');
hold on;

cell_width = 1;
cell_height = 0.8;

% 表头背景
for i = 1:num_seps
    rectangle('Position', [i-0.5, 2, cell_width, cell_height], 'FaceColor', [0.9, 0.9, 0.9], 'EdgeColor', 'k');
    text(i, 2.4, sprintf('%d°', angle_separations(i)), 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
end

% 数据行
for i = 1:num_seps
    % 静态行
    if static_resolved(i)
        rectangle('Position', [i-0.5, 1, cell_width, cell_height], 'FaceColor', [0.8, 1, 0.8], 'EdgeColor', 'k');
        text(i, 1.4, '○', 'HorizontalAlignment', 'center', 'FontSize', 16, 'Color', [0, 0.5, 0]);
    else
        rectangle('Position', [i-0.5, 1, cell_width, cell_height], 'FaceColor', [1, 0.85, 0.85], 'EdgeColor', 'k');
        text(i, 1.4, '×', 'HorizontalAlignment', 'center', 'FontSize', 14, 'Color', [0.7, 0, 0]);
    end
    
    % 运动行
    if motion_resolved(i)
        rectangle('Position', [i-0.5, 0, cell_width, cell_height], 'FaceColor', [0.8, 1, 0.8], 'EdgeColor', 'k');
        text(i, 0.4, '○', 'HorizontalAlignment', 'center', 'FontSize', 16, 'Color', [0, 0.5, 0]);
    else
        rectangle('Position', [i-0.5, 0, cell_width, cell_height], 'FaceColor', [1, 0.85, 0.85], 'EdgeColor', 'k');
        text(i, 0.4, '×', 'HorizontalAlignment', 'center', 'FontSize', 14, 'Color', [0.7, 0, 0]);
    end
end

% 行标签
text(0.3, 1.4, '静态', 'HorizontalAlignment', 'right', 'FontSize', 12, 'FontWeight', 'bold');
text(0.3, 0.4, '运动', 'HorizontalAlignment', 'right', 'FontSize', 12, 'FontWeight', 'bold');

hold off;

xlim([-0.5, num_seps + 0.5]);
ylim([-0.3, 3]);
axis off;
title('方案6: 表格风格', 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig6, fullfile(output_folder, '方案6_表格风格.png'));
fprintf('  已保存\n');

%% 完成
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('所有方案已生成！\n');
fprintf('输出目录: %s\n', output_folder);
fprintf('═══════════════════════════════════════════════════════════════════\n');




