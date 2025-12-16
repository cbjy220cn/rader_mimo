%% ═══════════════════════════════════════════════════════════════════════════
%  生成对数Y轴的论文图表
%  - 从实验结果mat文件加载数据
%  - 生成与原始图相同尺寸的对数Y轴版本 (10^0, 10^-1, 10^-2等)
%% ═══════════════════════════════════════════════════════════════════════════
clear; clc; close all;

% 加载数据
data_folder = 'validation_results/comprehensive_motion_array_test_20251209_191404';
load(fullfile(data_folder, 'experiment_results.mat'));

fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('生成对数Y轴论文图表\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% 设置论文风格
set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultLineLineWidth', 1.5);
set(0, 'DefaultTextInterpreter', 'none');

% 配色方案
color_static = [0.2, 0.2, 0.2];
color_x_trans = [0.85, 0.33, 0.1];
color_y_trans = [0.0, 0.45, 0.74];
color_rotation = [0.47, 0.67, 0.19];
color_combined = [0.49, 0.18, 0.56];
motion_colors = [color_static; color_x_trans; color_y_trans; color_rotation; color_combined];

snr_range = results.snr_range;

%% ═══════════════════════════════════════════════════════════════════════════
%  图1: 所有阵列静态vs运动RMSE综合对比 (2×4子图) - 对数Y轴
%% ═══════════════════════════════════════════════════════════════════════════
figure('Position', [50, 50, 1400, 700], 'Color', 'white');

array_names = {'ULA-8', 'URA-3×3', 'URA-4×2', 'L阵列', 'T阵列', '圆阵-8', '十字阵列', 'Y阵列'};

for arr_idx = 1:8
    subplot(2, 4, arr_idx);
    hold on;
    
    % 静态
    rmse_static = squeeze(results.rmse(arr_idx, 1, :));
    plot(snr_range, rmse_static, '-o', 'Color', color_static, ...
        'LineWidth', 2.5, 'MarkerSize', 7, 'MarkerFaceColor', color_static, ...
        'DisplayName', '静态');
    
    % x平移
    rmse_x = squeeze(results.rmse(arr_idx, 2, :));
    plot(snr_range, rmse_x, '-s', 'Color', color_x_trans, ...
        'LineWidth', 2.5, 'MarkerSize', 7, 'MarkerFaceColor', color_x_trans, ...
        'DisplayName', 'x平移');
    
    % y平移  
    rmse_y = squeeze(results.rmse(arr_idx, 3, :));
    plot(snr_range, rmse_y, '-d', 'Color', color_y_trans, ...
        'LineWidth', 2.5, 'MarkerSize', 7, 'MarkerFaceColor', color_y_trans, ...
        'DisplayName', 'y平移');
    
    % 旋转
    rmse_rot = squeeze(results.rmse(arr_idx, 4, :));
    plot(snr_range, rmse_rot, '-^', 'Color', color_rotation, ...
        'LineWidth', 2.5, 'MarkerSize', 7, 'MarkerFaceColor', color_rotation, ...
        'DisplayName', '旋转');
    
    % 对数坐标轴
    set(gca, 'YScale', 'log');
    xlabel('SNR (dB)', 'FontWeight', 'bold');
    ylabel('RMSE (°)', 'FontWeight', 'bold');
    title(array_names{arr_idx}, 'FontWeight', 'bold', 'FontSize', 12);
    if arr_idx == 1
        legend('Location', 'southwest', 'FontSize', 8);
    end
    grid on;
    
    % 对数Y轴范围: 10^-2 到 10^2
    ylim([0.01, 100]);
    xlim([snr_range(1)-1, snr_range(end)+1]);
    
    % 设置Y轴刻度
    set(gca, 'YTick', [0.01, 0.1, 1, 10, 100]);
    set(gca, 'YTickLabel', {'10^{-2}', '10^{-1}', '10^0', '10^1', '10^2'});
end

sgtitle('所有阵列: 静态 vs 运动模式 RMSE对比 (对数坐标)', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(data_folder, 'fig1_静态vs运动阵列_对数.png'));
saveas(gcf, fullfile(data_folder, 'fig1_静态vs运动阵列_对数.eps'), 'epsc');
fprintf('保存: fig1_静态vs运动阵列_对数.png\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图2: 多阵列运动模式对比 (2×2子图) - 对数Y轴
%% ═══════════════════════════════════════════════════════════════════════════
figure('Position', [50, 50, 1100, 900], 'Color', 'white');

% 选择4个代表性阵列
selected_arrays = [1, 2, 6, 8];  % ULA-8, URA-3x3, 圆阵-8, Y阵列
arr_names_fig2 = {'ULA-8 (线阵)', 'URA-3×3 (面阵)', '圆阵-8 (圆形)', 'Y阵列 (稀疏)'};
motion_markers = {'o', 's', 'd', '^', 'v'};
motion_names = {'静态', 'x平移', 'y平移', '旋转', '平移+旋转'};

low_snr_idx = find(snr_range == -10, 1);
if isempty(low_snr_idx), low_snr_idx = 2; end

for i = 1:4
    subplot(2, 2, i);
    arr_idx = selected_arrays(i);
    hold on;
    
    for mot_idx = 1:5
        rmse_curve = squeeze(results.rmse(arr_idx, mot_idx, :));
        plot(snr_range, rmse_curve, ['-' motion_markers{mot_idx}], ...
            'Color', motion_colors(mot_idx, :), ...
            'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerFaceColor', motion_colors(mot_idx, :), ...
            'DisplayName', motion_names{mot_idx});
    end
    
    % 对数坐标轴
    set(gca, 'YScale', 'log');
    xlabel('信噪比 (dB)', 'FontWeight', 'bold');
    ylabel('RMSE (°)', 'FontWeight', 'bold');
    title(arr_names_fig2{i}, 'FontSize', 12, 'FontWeight', 'bold');
    
    if i == 1
        legend('Location', 'southwest', 'FontSize', 9);
    end
    grid on;
    
    % 对数Y轴范围
    ylim([0.01, 100]);
    xlim([snr_range(1)-1, snr_range(end)+1]);
    
    % 设置Y轴刻度
    set(gca, 'YTick', [0.01, 0.1, 1, 10, 100]);
    set(gca, 'YTickLabel', {'10^{-2}', '10^{-1}', '10^0', '10^1', '10^2'});
    
    % 计算并显示改善倍数
    static_rmse = results.rmse(arr_idx, 1, low_snr_idx);
    best_motion_rmse = min([results.rmse(arr_idx, 2, low_snr_idx), ...
                           results.rmse(arr_idx, 3, low_snr_idx), ...
                           results.rmse(arr_idx, 5, low_snr_idx)]);
    improvement = static_rmse / max(best_motion_rmse, 0.01);
    
    % 显示改善倍数 - 放在左下角
    text(snr_range(end)-8, 0.02, ...
        sprintf('低SNR改善: %.1f×', improvement), ...
        'FontSize', 10, 'FontWeight', 'bold', 'Color', [0 0.5 0], ...
        'BackgroundColor', [1 1 1 0.8], 'EdgeColor', [0 0.5 0]);
end

sgtitle('不同阵列形状: 各运动模式RMSE对比 (对数坐标)', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(data_folder, 'fig2_运动模式对比_对数.png'));
saveas(gcf, fullfile(data_folder, 'fig2_运动模式对比_对数.eps'), 'epsc');
fprintf('保存: fig2_运动模式对比_对数.png\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图3: 核心对比 - ULA专注图（论文核心）- 对数坐标
%% ═══════════════════════════════════════════════════════════════════════════
figure('Position', [50, 50, 800, 600], 'Color', 'white');

arr_idx = 1;  % ULA-8

% 只显示静态、x平移、y平移（不含旋转）
hold on;
rmse_static = squeeze(results.rmse(arr_idx, 1, :));
rmse_x = squeeze(results.rmse(arr_idx, 2, :));
rmse_y = squeeze(results.rmse(arr_idx, 3, :));

plot(snr_range, rmse_static, '-o', 'Color', color_static, ...
    'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', color_static, ...
    'DisplayName', '静态阵列');
plot(snr_range, rmse_x, '-s', 'Color', color_x_trans, ...
    'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', color_x_trans, ...
    'DisplayName', 'x方向平移');
plot(snr_range, rmse_y, '-d', 'Color', color_y_trans, ...
    'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', color_y_trans, ...
    'DisplayName', 'y方向平移');

set(gca, 'YScale', 'log');
xlabel('信噪比 SNR (dB)', 'FontWeight', 'bold', 'FontSize', 14);
ylabel('DOA估计RMSE (°)', 'FontWeight', 'bold', 'FontSize', 14);
title('8元线阵: 运动模式对DOA精度的影响', 'FontSize', 16, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 12);

% 对数Y轴
ylim([0.01, 100]);
xlim([snr_range(1)-1, snr_range(end)+1]);
set(gca, 'YTick', [0.01, 0.1, 1, 10, 100]);
set(gca, 'YTickLabel', {'10^{-2}', '10^{-1}', '10^0', '10^1', '10^2'});

% 添加关键信息标注
annotation('textbox', [0.15, 0.65, 0.25, 0.15], ...
    'String', {sprintf('静态孔径: %.1fλ', results.aperture(arr_idx, 1)), ...
               sprintf('合成孔径: %.1fλ', results.aperture(arr_idx, 3)), ...
               sprintf('孔径扩展: %.1f倍', results.aperture(arr_idx, 3)/results.aperture(arr_idx, 1))}, ...
    'FontSize', 11, 'BackgroundColor', [1 1 0.9], 'EdgeColor', [0 0 0]);

saveas(gcf, fullfile(data_folder, 'fig3_ULA核心对比_对数.png'));
saveas(gcf, fullfile(data_folder, 'fig3_ULA核心对比_对数.eps'), 'epsc');
fprintf('保存: fig3_ULA核心对比_对数.png\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图4: 论文最终版 - 双列布局 (对数坐标)
%% ═══════════════════════════════════════════════════════════════════════════
figure('Position', [50, 50, 1200, 500], 'Color', 'white');

% 左图: 静态 vs 运动（代表性阵列）
subplot(1, 2, 1);
hold on;

markers = {'o', 's', 'd', '^'};
selected = [1, 2, 6, 8];
names = {'ULA-8', 'URA-3×3', '圆阵-8', 'Y阵列'};
colors_arr = lines(4);

% 静态（虚线）
for i = 1:4
    arr = selected(i);
    rmse = squeeze(results.rmse(arr, 1, :));
    plot(snr_range, rmse, ['--' markers{i}], 'Color', colors_arr(i,:), ...
        'LineWidth', 1.5, 'MarkerSize', 6);
end
% y平移（实线）
for i = 1:4
    arr = selected(i);
    rmse = squeeze(results.rmse(arr, 3, :));
    plot(snr_range, rmse, ['-' markers{i}], 'Color', colors_arr(i,:), ...
        'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', colors_arr(i,:));
end

set(gca, 'YScale', 'log');
xlabel('SNR (dB)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('RMSE (°)', 'FontWeight', 'bold', 'FontSize', 12);
title('(a) 静态(虚线) vs y平移(实线)', 'FontSize', 13, 'FontWeight', 'bold');
legend(names, 'Location', 'southwest', 'FontSize', 10);
grid on;
ylim([0.01, 100]);
xlim([snr_range(1)-1, snr_range(end)+1]);
set(gca, 'FontSize', 11);
set(gca, 'YTick', [0.01, 0.1, 1, 10, 100]);
set(gca, 'YTickLabel', {'10^{-2}', '10^{-1}', '10^0', '10^1', '10^2'});

% 右图: 改善倍数柱状图
subplot(1, 2, 2);

% 计算低SNR和高SNR下的改善倍数
low_snr_idx = find(snr_range == -10, 1);
high_snr_idx = find(snr_range == 20, 1);
if isempty(low_snr_idx), low_snr_idx = 2; end
if isempty(high_snr_idx), high_snr_idx = length(snr_range); end

improvement_low = zeros(1, 4);
improvement_high = zeros(1, 4);
for i = 1:4
    arr = selected(i);
    improvement_low(i) = results.rmse(arr, 1, low_snr_idx) / max(results.rmse(arr, 3, low_snr_idx), 0.01);
    improvement_high(i) = results.rmse(arr, 1, high_snr_idx) / max(results.rmse(arr, 3, high_snr_idx), 0.01);
end

bar_data = [improvement_low; improvement_high]';
b = bar(1:4, bar_data, 'grouped');
b(1).FaceColor = [0.2, 0.6, 0.9];
b(2).FaceColor = [0.9, 0.4, 0.2];

set(gca, 'XTick', 1:4, 'XTickLabel', names);
xlabel('阵列配置', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('RMSE改善倍数', 'FontWeight', 'bold', 'FontSize', 12);
title('(b) y平移相对静态的改善', 'FontSize', 13, 'FontWeight', 'bold');
legend({'SNR=-10dB', 'SNR=20dB'}, 'Location', 'northwest', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 11);

sgtitle('运动阵列DOA性能提升分析', 'FontSize', 15, 'FontWeight', 'bold');
saveas(gcf, fullfile(data_folder, 'fig4_论文综合图_对数.png'));
saveas(gcf, fullfile(data_folder, 'fig4_论文综合图_对数.eps'), 'epsc');
fprintf('保存: fig4_论文综合图_对数.png\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图5: 主瓣宽度对比柱状图
%% ═══════════════════════════════════════════════════════════════════════════
figure('Position', [50, 50, 900, 400], 'Color', 'white');

% 选择代表性阵列
selected = [1, 2, 6, 8];
names = {'ULA-8', 'URA-3×3', '圆阵-8', 'Y阵列'};
motion_names = {'静态', 'x平移', 'y平移', '旋转', '平移+旋转'};

beamwidth_selected = results.beamwidth(selected, :);  % [4阵列 × 5运动]

b = bar(beamwidth_selected, 'grouped');
for i = 1:5
    b(i).FaceColor = motion_colors(i, :);
end

set(gca, 'XTick', 1:4, 'XTickLabel', names);
ylabel('主瓣宽度 (°)', 'FontWeight', 'bold', 'FontSize', 12);
title('运动模式对主瓣宽度的影响', 'FontSize', 14, 'FontWeight', 'bold');
legend(motion_names, 'Location', 'northwest', 'FontSize', 10);
grid on;
ylim([0, 3]);

saveas(gcf, fullfile(data_folder, 'fig5_主瓣宽度对比.png'));
saveas(gcf, fullfile(data_folder, 'fig5_主瓣宽度对比.eps'), 'epsc');
fprintf('保存: fig5_主瓣宽度对比.png\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图6: 孔径扩展对比柱状图
%% ═══════════════════════════════════════════════════════════════════════════
figure('Position', [50, 50, 900, 400], 'Color', 'white');

aperture_selected = results.aperture(selected, :);  % [4阵列 × 5运动]

b = bar(aperture_selected, 'grouped');
for i = 1:5
    b(i).FaceColor = motion_colors(i, :);
end

set(gca, 'XTick', 1:4, 'XTickLabel', names);
ylabel('合成孔径 (λ)', 'FontWeight', 'bold', 'FontSize', 12);
title('运动模式对合成孔径的影响', 'FontSize', 14, 'FontWeight', 'bold');
legend(motion_names, 'Location', 'northwest', 'FontSize', 10);
grid on;
ylim([0, 30]);

saveas(gcf, fullfile(data_folder, 'fig6_孔径扩展对比.png'));
saveas(gcf, fullfile(data_folder, 'fig6_孔径扩展对比.eps'), 'epsc');
fprintf('保存: fig6_孔径扩展对比.png\n');

%% 完成
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('所有对数Y轴图表已生成！\n');
fprintf('保存位置: %s\n', data_folder);
fprintf('═══════════════════════════════════════════════════════════════════\n');
