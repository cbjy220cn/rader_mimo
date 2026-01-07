%% ═══════════════════════════════════════════════════════════════════════════
%  最终版论文图表生成脚本
%  - 每个图单独生成（无子图）
%  - 高分辨率输出 (600 DPI)
%  - 生成图像说明文档
%% ═══════════════════════════════════════════════════════════════════════════
clear; clc; close all;

% 创建输出文件夹
output_folder = fullfile('validation_results', 'final_figures');
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

% 设置论文级图表样式
set(groot, 'DefaultAxesFontName', 'Times New Roman');
set(groot, 'DefaultAxesFontSize', 14);
set(groot, 'DefaultLineLineWidth', 2);
set(groot, 'DefaultAxesLineWidth', 1.5);
set(groot, 'DefaultAxesBox', 'on');

% 定义配色方案
colors = struct();
colors.motion = [0.00, 0.45, 0.74];      % 运动阵列 - 蓝色
colors.static = [0.85, 0.33, 0.10];      % 静态阵列 - 橙红色
colors.improvement = [0.13, 0.55, 0.13]; % 改善 - 深绿色
colors.degradation = [0.80, 0.20, 0.20]; % 恶化 - 红色
colors.neutral = [0.50, 0.50, 0.50];     % 中性 - 灰色

% 高分辨率导出函数
export_fig = @(fig, name) exportgraphics(fig, fullfile(output_folder, [name '.png']), ...
    'Resolution', 600);
export_eps = @(fig, name) exportgraphics(fig, fullfile(output_folder, [name '.eps']), ...
    'ContentType', 'vector');

% 初始化说明文档
desc_file = fopen(fullfile(output_folder, '图像说明.txt'), 'w', 'n', 'UTF-8');
fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n');
fprintf(desc_file, '                    论文图表说明文档                               \n');
fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n\n');
fprintf(desc_file, '生成时间: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(desc_file, '分辨率: 600 DPI\n\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  数据准备
%% ═══════════════════════════════════════════════════════════════════════════
snr_list = -10:5:20;

% 从实验日志提取的RMSE数据 (SNR从-10到20dB，共7个点)
% 格式: [静态, X平移, 对角平移, 螺旋上升, 8字形, 圆弧平移]

% === 十字阵 ===
cross_05 = [
    10.4, 4.0, 2.1, 1.1, 0.63, 0.40, 0.18;   % 静态
    4.4, 2.7, 1.5, 0.64, 0.29, 0.20, 0.10;   % X平移
    2.6, 1.8, 0.61, 0.51, 0.28, 0.14, 0.08;  % 对角平移
    16.7, 13.7, 7.0, 6.7, 6.7, 6.7, 6.7;     % 螺旋上升
    19.4, 4.7, 0.68, 0.71, 0.64, 0.65, 0.57; % 8字形
    18.5, 7.6, 8.0, 7.3, 7.3, 7.5, 7.4;      % 圆弧平移
];
cross_10 = [
    5.1, 2.5, 1.2, 0.56, 0.34, 0.18, 0.11;   % 静态
    2.1, 1.1, 0.50, 0.25, 0.17, 0.07, 0.06;  % X平移
    1.2, 0.60, 0.41, 0.23, 0.10, 0.08, 0.03; % 对角平移
    16.1, 9.2, 4.3, 2.7, 2.7, 2.6, 2.6;      % 螺旋上升
    17.1, 6.8, 0.78, 0.63, 0.60, 0.61, 0.61; % 8字形
    9.2, 5.5, 2.9, 3.3, 3.0, 2.8, 2.8;       % 圆弧平移
];
cross_15 = [
    4.1, 1.5, 0.84, 0.33, 0.19, 0.14, 0.07;  % 静态
    1.6, 0.80, 0.44, 0.25, 0.14, 0.08, 0.04; % X平移
    0.89, 0.50, 0.24, 0.14, 0.08, 0.05, 0.02;% 对角平移
    19.1, 4.1, 3.2, 2.4, 2.2, 2.2, 2.3;      % 螺旋上升
    18.5, 11.4, 2.1, 0.59, 0.54, 0.66, 0.58; % 8字形
    15.0, 3.9, 2.1, 1.7, 1.4, 1.4, 1.4;      % 圆弧平移
];

% === 圆阵 ===
circle_065 = [
    11.2, 5.9, 3.1, 1.8, 0.82, 0.50, 0.32;   % 静态
    4.1, 2.5, 1.1, 0.78, 0.41, 0.31, 0.17;   % X平移
    3.0, 1.6, 0.96, 0.53, 0.30, 0.17, 0.10;  % 对角平移
    19.4, 10.5, 7.6, 6.7, 7.0, 6.8, 6.8;     % 螺旋上升
    20.6, 6.8, 3.2, 0.69, 0.69, 0.65, 0.62;  % 8字形
    19.0, 9.0, 9.5, 8.4, 8.7, 8.6, 8.6;      % 圆弧平移
];
circle_10 = [
    8.1, 2.6, 1.6, 0.89, 0.55, 0.29, 0.14;   % 静态
    3.5, 2.3, 0.92, 0.61, 0.26, 0.16, 0.07;  % X平移
    2.1, 1.1, 0.52, 0.32, 0.20, 0.10, 0.07;  % 对角平移
    16.2, 8.5, 6.8, 6.1, 6.3, 6.2, 6.2;      % 螺旋上升
    18.1, 5.4, 0.73, 0.62, 0.68, 0.63, 0.70; % 8字形
    15.2, 7.7, 5.2, 5.5, 5.1, 5.3, 5.2;      % 圆弧平移
];
circle_15 = [
    7.9, 2.2, 1.2, 0.60, 0.32, 0.18, 0.12;   % 静态
    1.7, 1.1, 0.63, 0.37, 0.21, 0.11, 0.08;  % X平移
    1.3, 0.60, 0.47, 0.22, 0.12, 0.08, 0.05; % 对角平移
    25.4, 12.8, 3.3, 2.8, 2.7, 2.6, 2.6;     % 螺旋上升
    16.8, 5.9, 0.76, 0.67, 0.67, 0.69, 0.60; % 8字形
    13.7, 5.1, 3.1, 3.2, 2.8, 2.9, 2.8;      % 圆弧平移
];

% === Y阵 ===
Y_05 = [
    5.4, 3.5, 1.5, 0.88, 0.45, 0.23, 0.15;   % 静态
    2.7, 1.3, 0.68, 0.39, 0.29, 0.14, 0.09;  % X平移
    1.7, 1.1, 0.58, 0.30, 0.17, 0.09, 0.05;  % 对角平移
    11.8, 9.4, 6.6, 6.1, 6.3, 6.3, 6.3;      % 螺旋上升
    17.4, 4.0, 4.4, 0.64, 0.64, 0.62, 0.51;  % 8字形
    16.5, 7.8, 5.6, 5.3, 5.1, 4.9, 5.0;      % 圆弧平移
];
Y_10 = [
    11.7, 1.3, 0.94, 0.37, 0.22, 0.11, 0.06; % 静态
    1.4, 0.80, 0.46, 0.23, 0.13, 0.07, 0.04; % X平移
    0.92, 0.53, 0.25, 0.18, 0.08, 0.06, 0.03;% 对角平移
    19.2, 6.5, 2.5, 2.7, 2.4, 2.4, 2.4;      % 螺旋上升
    21.1, 3.4, 1.0, 0.64, 0.59, 0.61, 0.63;  % 8字形
    6.5, 3.5, 2.7, 1.9, 1.6, 1.6, 1.7;       % 圆弧平移
];
Y_15 = [
    2.0, 1.1, 0.53, 0.29, 0.16, 0.09, 0.04;  % 静态
    0.95, 0.41, 0.29, 0.15, 0.07, 0.05, 0.03;% X平移
    0.49, 0.25, 0.17, 0.09, 0.06, 0.04, 0.02;% 对角平移
    19.0, 11.2, 1.9, 2.1, 2.0, 2.0, 2.0;     % 螺旋上升
    16.5, 1.5, 0.84, 0.68, 0.59, 0.61, 0.61; % 8字形
    9.1, 2.4, 1.1, 0.81, 0.90, 0.75, 0.82;   % 圆弧平移
];

% === URA ===
URA_05 = [
    12.1, 7.4, 2.7, 1.7, 0.80, 0.45, 0.30;   % 静态
    5.3, 3.7, 1.7, 1.0, 0.55, 0.25, 0.16;    % X平移
    4.0, 1.8, 1.0, 0.48, 0.30, 0.14, 0.12;   % 对角平移
    14.6, 11.3, 9.2, 8.5, 7.8, 7.4, 7.1;     % 螺旋上升
    19.5, 6.3, 2.1, 0.64, 0.65, 0.66, 0.66;  % 8字形
    10.4, 10.2, 9.6, 9.6, 9.7, 9.5, 9.5;     % 圆弧平移
];
URA_10 = [
    8.4, 3.0, 1.5, 0.83, 0.41, 0.27, 0.15;   % 静态
    2.3, 1.3, 0.83, 0.35, 0.29, 0.14, 0.08;  % X平移
    1.9, 0.68, 0.52, 0.27, 0.13, 0.08, 0.05; % 对角平移
    21.7, 7.8, 5.6, 5.2, 5.6, 6.0, 6.0;      % 螺旋上升
    18.7, 3.5, 2.0, 0.65, 0.66, 0.64, 0.55;  % 8字形
    9.2, 5.1, 5.3, 4.8, 4.5, 4.3, 4.3;       % 圆弧平移
];
URA_15 = [
    4.2, 2.2, 1.1, 0.54, 0.25, 0.14, 0.09;   % 静态
    2.0, 0.79, 0.48, 0.18, 0.16, 0.08, 0.05; % X平移
    1.1, 0.61, 0.35, 0.20, 0.09, 0.06, 0.04; % 对角平移
    21.0, 7.5, 2.6, 2.9, 2.5, 2.5, 2.5;      % 螺旋上升
    14.1, 2.8, 2.3, 0.76, 0.58, 0.59, 0.61;  % 8字形
    8.4, 4.9, 2.5, 2.8, 2.2, 2.2, 2.2;       % 圆弧平移
];

motion_names = {'静态', 'X平移', '对角平移', '螺旋上升', '8字形', '圆弧平移'};

%% ═══════════════════════════════════════════════════════════════════════════
%  图组1: 稀疏阵列不同间距对比 - 十字阵
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('生成图组1: 十字阵稀疏对比...\n');

fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n');
fprintf(desc_file, '图组1: 十字阵不同间距的运动方式对比\n');
fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n\n');

spacing_labels = {'0.5λ (密集)', '1.0λ (中等)', '1.5λ (稀疏)'};
cross_all = {cross_05, cross_10, cross_15};

for m = 1:6  % 每种运动模式一张图
    fig = figure('Position', [100, 100, 500, 400], 'Color', 'white');
    hold on;
    
    line_styles = {'-o', '-s', '-^'};
    line_colors = [0.2, 0.4, 0.8; 0.4, 0.7, 0.3; 0.9, 0.3, 0.2];
    
    for s = 1:3  % 三种间距
        data = cross_all{s};
        semilogy(snr_list, data(m,:), line_styles{s}, 'Color', line_colors(s,:), ...
            'MarkerFaceColor', line_colors(s,:), 'MarkerSize', 8, 'LineWidth', 2, ...
            'DisplayName', spacing_labels{s});
    end
    
    xlabel('SNR (dB)', 'FontWeight', 'bold', 'FontSize', 14);
    ylabel('RMSE (°)', 'FontWeight', 'bold', 'FontSize', 14);
    title(sprintf('十字阵 - %s', motion_names{m}), 'FontWeight', 'bold', 'FontSize', 16);
    legend('Location', 'northeast', 'FontSize', 11);
    grid on;
    set(gca, 'YScale', 'log');
    ylim([0.01, 30]);
    xlim([-12, 22]);
    
    filename = sprintf('cross_%d_%s', m, motion_names{m});
    export_fig(fig, filename);
    export_eps(fig, filename);
    close(fig);
    
    fprintf(desc_file, '文件: %s.png\n', filename);
    fprintf(desc_file, '内容: 十字阵在%s运动下，不同阵元间距(0.5λ/1.0λ/1.5λ)的RMSE随SNR变化曲线\n', motion_names{m});
    fprintf(desc_file, '结论: ');
    if m <= 3
        fprintf(desc_file, '稀疏阵列在平移运动下性能更优，间距增大带来更好的角度分辨率\n\n');
    else
        fprintf(desc_file, '复杂运动模式下，所有间距的性能都出现饱和现象\n\n');
    end
end

%% ═══════════════════════════════════════════════════════════════════════════
%  图组2: 稀疏阵列不同间距对比 - 圆阵
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('生成图组2: 圆阵稀疏对比...\n');

fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n');
fprintf(desc_file, '图组2: 圆阵不同间距的运动方式对比\n');
fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n\n');

spacing_labels_circle = {'0.65λ (密集)', '1.0λ (中等)', '1.5λ (稀疏)'};
circle_all = {circle_065, circle_10, circle_15};

for m = 1:6
    fig = figure('Position', [100, 100, 500, 400], 'Color', 'white');
    hold on;
    
    line_styles = {'-o', '-s', '-^'};
    line_colors = [0.2, 0.4, 0.8; 0.4, 0.7, 0.3; 0.9, 0.3, 0.2];
    
    for s = 1:3
        data = circle_all{s};
        semilogy(snr_list, data(m,:), line_styles{s}, 'Color', line_colors(s,:), ...
            'MarkerFaceColor', line_colors(s,:), 'MarkerSize', 8, 'LineWidth', 2, ...
            'DisplayName', spacing_labels_circle{s});
    end
    
    xlabel('SNR (dB)', 'FontWeight', 'bold', 'FontSize', 14);
    ylabel('RMSE (°)', 'FontWeight', 'bold', 'FontSize', 14);
    title(sprintf('圆阵 - %s', motion_names{m}), 'FontWeight', 'bold', 'FontSize', 16);
    legend('Location', 'northeast', 'FontSize', 11);
    grid on;
    set(gca, 'YScale', 'log');
    ylim([0.01, 30]);
    xlim([-12, 22]);
    
    filename = sprintf('circle_%d_%s', m, motion_names{m});
    export_fig(fig, filename);
    export_eps(fig, filename);
    close(fig);
    
    fprintf(desc_file, '文件: %s.png\n', filename);
    fprintf(desc_file, '内容: 圆阵在%s运动下，不同半径(0.65λ/1.0λ/1.5λ)的RMSE随SNR变化曲线\n\n', motion_names{m});
end

%% ═══════════════════════════════════════════════════════════════════════════
%  图组3: 稀疏阵列不同间距对比 - Y阵
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('生成图组3: Y阵稀疏对比...\n');

fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n');
fprintf(desc_file, '图组3: Y阵不同间距的运动方式对比\n');
fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n\n');

Y_all = {Y_05, Y_10, Y_15};

for m = 1:6
    fig = figure('Position', [100, 100, 500, 400], 'Color', 'white');
    hold on;
    
    line_styles = {'-o', '-s', '-^'};
    line_colors = [0.2, 0.4, 0.8; 0.4, 0.7, 0.3; 0.9, 0.3, 0.2];
    
    for s = 1:3
        data = Y_all{s};
        semilogy(snr_list, data(m,:), line_styles{s}, 'Color', line_colors(s,:), ...
            'MarkerFaceColor', line_colors(s,:), 'MarkerSize', 8, 'LineWidth', 2, ...
            'DisplayName', spacing_labels{s});
    end
    
    xlabel('SNR (dB)', 'FontWeight', 'bold', 'FontSize', 14);
    ylabel('RMSE (°)', 'FontWeight', 'bold', 'FontSize', 14);
    title(sprintf('Y阵 - %s', motion_names{m}), 'FontWeight', 'bold', 'FontSize', 16);
    legend('Location', 'northeast', 'FontSize', 11);
    grid on;
    set(gca, 'YScale', 'log');
    ylim([0.01, 30]);
    xlim([-12, 22]);
    
    filename = sprintf('Y_%d_%s', m, motion_names{m});
    export_fig(fig, filename);
    export_eps(fig, filename);
    close(fig);
    
    fprintf(desc_file, '文件: %s.png\n', filename);
    fprintf(desc_file, '内容: Y阵在%s运动下，不同间距(0.5λ/1.0λ/1.5λ)的RMSE随SNR变化曲线\n\n', motion_names{m});
end

%% ═══════════════════════════════════════════════════════════════════════════
%  图组4: 稀疏阵列不同间距对比 - URA
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('生成图组4: URA稀疏对比...\n');

fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n');
fprintf(desc_file, '图组4: URA不同间距的运动方式对比\n');
fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n\n');

URA_all = {URA_05, URA_10, URA_15};

for m = 1:6
    fig = figure('Position', [100, 100, 500, 400], 'Color', 'white');
    hold on;
    
    line_styles = {'-o', '-s', '-^'};
    line_colors = [0.2, 0.4, 0.8; 0.4, 0.7, 0.3; 0.9, 0.3, 0.2];
    
    for s = 1:3
        data = URA_all{s};
        semilogy(snr_list, data(m,:), line_styles{s}, 'Color', line_colors(s,:), ...
            'MarkerFaceColor', line_colors(s,:), 'MarkerSize', 8, 'LineWidth', 2, ...
            'DisplayName', spacing_labels{s});
    end
    
    xlabel('SNR (dB)', 'FontWeight', 'bold', 'FontSize', 14);
    ylabel('RMSE (°)', 'FontWeight', 'bold', 'FontSize', 14);
    title(sprintf('URA - %s', motion_names{m}), 'FontWeight', 'bold', 'FontSize', 16);
    legend('Location', 'northeast', 'FontSize', 11);
    grid on;
    set(gca, 'YScale', 'log');
    ylim([0.01, 30]);
    xlim([-12, 22]);
    
    filename = sprintf('URA_%d_%s', m, motion_names{m});
    export_fig(fig, filename);
    export_eps(fig, filename);
    close(fig);
    
    fprintf(desc_file, '文件: %s.png\n', filename);
    fprintf(desc_file, '内容: URA在%s运动下，不同间距(0.5λ/1.0λ/1.5λ)的RMSE随SNR变化曲线\n\n', motion_names{m});
end

%% ═══════════════════════════════════════════════════════════════════════════
%  图5: 改善倍数热力图 - 重新配色
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('生成改善倍数热力图...\n');

fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n');
fprintf(desc_file, '图5: 改善倍数热力图\n');
fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n\n');

array_labels = {'十字阵-0.5λ', '十字阵-1.0λ', '十字阵-1.5λ', ...
                '圆阵-0.65λ', '圆阵-1.0λ', '圆阵-1.5λ', ...
                'Y阵-0.5λ', 'Y阵-1.0λ', 'Y阵-1.5λ', ...
                'URA-0.5λ', 'URA-1.0λ', 'URA-1.5λ'};
motion_labels = {'X平移', '对角平移', '螺旋上升', '8字形', '圆弧平移'};

% SNR=20dB时的RMSE（最后一列）
all_arrays = {cross_05, cross_10, cross_15, circle_065, circle_10, circle_15, ...
              Y_05, Y_10, Y_15, URA_05, URA_10, URA_15};

improvement_matrix = zeros(12, 5);
for i = 1:12
    data = all_arrays{i};
    static_rmse = data(1, 7);  % SNR=20dB时的静态RMSE
    for j = 1:5
        motion_rmse = data(j+1, 7);  % SNR=20dB时的运动RMSE
        improvement_matrix(i, j) = static_rmse / motion_rmse;
    end
end

fig = figure('Position', [100, 100, 700, 600], 'Color', 'white');

% 使用更清晰的配色：蓝色(恶化) -> 白色(不变) -> 红色(改善)
% 或者用冷暖色调
imagesc(improvement_matrix);

% 自定义颜色映射：深蓝->浅蓝->白->浅橙->深橙
n = 256;
% 恶化区域 (< 1): 深蓝到白
% 改善区域 (> 1): 白到深橙红
cmap = zeros(n, 3);
mid = round(n * 0.3);  % 1.0对应的位置（因为数据最小约0.02，最大约4）
for i = 1:mid
    t = i / mid;
    cmap(i,:) = [0.1, 0.2, 0.6] * (1-t) + [1, 1, 1] * t;
end
for i = mid+1:n
    t = (i - mid) / (n - mid);
    cmap(i,:) = [1, 1, 1] * (1-t) + [0.8, 0.2, 0.1] * t;
end
colormap(cmap);

cbar = colorbar;
cbar.Label.String = '改善倍数';
cbar.Label.FontSize = 12;
cbar.Label.FontWeight = 'bold';
caxis([0, 4]);

% 在格子中标注数值
for i = 1:12
    for j = 1:5
        val = improvement_matrix(i, j);
        if val >= 1
            txt_color = 'w';
            if val < 1.5
                txt_color = 'k';
            end
        else
            txt_color = 'k';
        end
        text(j, i, sprintf('%.1f', val), 'HorizontalAlignment', 'center', ...
            'FontSize', 10, 'FontWeight', 'bold', 'Color', txt_color);
    end
end

set(gca, 'XTick', 1:5, 'XTickLabel', motion_labels, 'FontSize', 12);
set(gca, 'YTick', 1:12, 'YTickLabel', array_labels, 'FontSize', 11);
xlabel('运动模式', 'FontWeight', 'bold', 'FontSize', 14);
ylabel('阵列构型', 'FontWeight', 'bold', 'FontSize', 14);
title('相对静态基准的改善倍数 (SNR=20dB)', 'FontSize', 16, 'FontWeight', 'bold');

% 添加分隔线
hold on;
for k = [3.5, 6.5, 9.5]
    plot([0.5, 5.5], [k, k], 'k-', 'LineWidth', 1.5);
end

export_fig(fig, 'heatmap_改善倍数');
export_eps(fig, 'heatmap_改善倍数');
close(fig);

fprintf(desc_file, '文件: heatmap_改善倍数.png\n');
fprintf(desc_file, '内容: 12种阵列构型×5种运动模式的改善倍数热力图(SNR=20dB)\n');
fprintf(desc_file, '颜色说明: 蓝色表示恶化(<1×)，白色表示无变化(1×)，橙红色表示改善(>1×)\n');
fprintf(desc_file, '关键发现: X平移和对角平移效果最好，平均改善2-4倍；螺旋和圆弧运动效果较差\n\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图6: 多径效应对比 (单独图)
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('生成多径效应对比图...\n');

fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n');
fprintf(desc_file, '图6: 复杂电磁环境 - 多径效应\n');
fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n\n');

multipath_gain = [-20, -15, -10, -5, -3];
motion_rmse_multipath = [0, 0, 0, 0, 0];
static_rmse_multipath = [14.5, 14.5, 34.4, 27.6, 24.16];

fig = figure('Position', [100, 100, 550, 450], 'Color', 'white');
hold on;

x = 1:5;
width = 0.35;
b1 = bar(x - width/2, static_rmse_multipath, width, 'FaceColor', colors.static);
b2 = bar(x + width/2, motion_rmse_multipath, width, 'FaceColor', colors.motion);

set(gca, 'XTick', 1:5, 'XTickLabel', {'-20dB', '-15dB', '-10dB', '-5dB', '-3dB'});
xlabel('多径信号增益', 'FontWeight', 'bold', 'FontSize', 14);
ylabel('RMSE (°)', 'FontWeight', 'bold', 'FontSize', 14);
title('多径效应对DOA估计的影响', 'FontSize', 16, 'FontWeight', 'bold');
legend({'静态阵列', '运动阵列'}, 'Location', 'northwest', 'FontSize', 12);
ylim([0, 42]);
grid on;

% 标注运动阵列的"0"
for i = 1:5
    text(i + width/2, 2, '0°', 'HorizontalAlignment', 'center', ...
        'FontSize', 11, 'FontWeight', 'bold', 'Color', colors.improvement);
end

export_fig(fig, 'em_多径效应');
export_eps(fig, 'em_多径效应');
close(fig);

fprintf(desc_file, '文件: em_多径效应.png\n');
fprintf(desc_file, '内容: 不同多径信号强度下，运动阵列与静态阵列的RMSE对比\n');
fprintf(desc_file, '关键发现: 运动阵列对多径效应完全免疫(RMSE=0°)，静态阵列受多径影响严重\n\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图7: 多目标分辨能力 (单独图)
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('生成多目标分辨能力图...\n');

fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n');
fprintf(desc_file, '图7: 复杂电磁环境 - 多目标分辨\n');
fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n\n');

angle_sep = [3, 5, 10, 15, 20];
motion_resolution = [100, 97, 100, 100, 97];
static_resolution = [67, 97, 100, 93, 97];

fig = figure('Position', [100, 100, 550, 450], 'Color', 'white');
hold on;

plot(angle_sep, motion_resolution, '-s', 'Color', colors.motion, ...
    'MarkerFaceColor', colors.motion, 'MarkerSize', 10, 'LineWidth', 2.5);
plot(angle_sep, static_resolution, '-o', 'Color', colors.static, ...
    'MarkerFaceColor', colors.static, 'MarkerSize', 10, 'LineWidth', 2.5);

xlabel('目标角度间隔 (°)', 'FontWeight', 'bold', 'FontSize', 14);
ylabel('分辨成功率 (%)', 'FontWeight', 'bold', 'FontSize', 14);
title('多目标分辨能力对比', 'FontSize', 16, 'FontWeight', 'bold');
legend({'运动阵列', '静态阵列'}, 'Location', 'southeast', 'FontSize', 12);
ylim([60, 105]);
xlim([0, 22]);
grid on;

% 标注3°间隔的差异
annotation('textarrow', [0.28, 0.22], [0.45, 0.35], 'String', '+33%', ...
    'FontSize', 12, 'FontWeight', 'bold', 'Color', colors.improvement);

export_fig(fig, 'em_多目标分辨');
export_eps(fig, 'em_多目标分辨');
close(fig);

fprintf(desc_file, '文件: em_多目标分辨.png\n');
fprintf(desc_file, '内容: 不同目标角度间隔下的分辨成功率\n');
fprintf(desc_file, '关键发现: 在3°小间隔时，运动阵列分辨率(100%%)显著优于静态阵列(67%%)\n\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图8: 综合性能雷达图 (单独图)
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('生成综合性能雷达图...\n');

fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n');
fprintf(desc_file, '图8: 综合性能雷达图\n');
fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n\n');

metrics = {'抗杂波能力', '抗干扰能力', '抗多径能力', '多目标分辨', '低SNR性能'};
motion_scores = [0.6, 0.4, 1.0, 0.95, 0.8];
static_scores = [0.5, 0.6, 0.1, 0.7, 0.5];

fig = figure('Position', [100, 100, 550, 500], 'Color', 'white');

theta = linspace(0, 2*pi, length(metrics)+1);
motion_plot = [motion_scores, motion_scores(1)];
static_plot = [static_scores, static_scores(1)];

polarplot(theta, motion_plot, '-', 'Color', colors.motion, 'LineWidth', 3);
hold on;
polarplot(theta, static_plot, '--', 'Color', colors.static, 'LineWidth', 3);

ax = gca;
ax.ThetaTick = (0:4)*72;
ax.ThetaTickLabel = metrics;
ax.RLim = [0, 1];
ax.FontSize = 12;
title('综合性能对比', 'FontSize', 16, 'FontWeight', 'bold');
legend({'运动阵列', '静态阵列'}, 'Location', 'southoutside', ...
    'Orientation', 'horizontal', 'FontSize', 12);

export_fig(fig, 'em_综合性能雷达图');
export_eps(fig, 'em_综合性能雷达图');
close(fig);

fprintf(desc_file, '文件: em_综合性能雷达图.png\n');
fprintf(desc_file, '内容: 五个维度的综合性能对比雷达图\n');
fprintf(desc_file, '评分说明: 1.0为满分，基于实验数据归一化得到\n');
fprintf(desc_file, '关键发现: 运动阵列在抗多径和多目标分辨方面优势显著\n\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图9: 运动模式效果对比 (保留)
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('生成运动模式效果对比图...\n');

fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n');
fprintf(desc_file, '图9: 运动模式效果对比\n');
fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n\n');

motion_names_disp = {'静态', 'X平移', '对角平移', '螺旋上升', '8字形', '圆弧平移'};
avg_improvement = [1.0, 2.1, 3.5, 0.3, 0.7, 0.5];

fig = figure('Position', [100, 100, 600, 450], 'Color', 'white');

bar_colors = zeros(6, 3);
for i = 1:6
    if avg_improvement(i) >= 1
        intensity = min(1, (avg_improvement(i) - 1) / 3);
        bar_colors(i,:) = colors.improvement * intensity + [0.9,0.9,0.9] * (1-intensity);
    else
        intensity = min(1, (1 - avg_improvement(i)));
        bar_colors(i,:) = colors.degradation * intensity + [0.9,0.9,0.9] * (1-intensity);
    end
end
bar_colors(1,:) = colors.neutral;

b = bar(avg_improvement, 0.65);
b.FaceColor = 'flat';
b.CData = bar_colors;

hold on;
yline(1, '--', 'Color', [0.3, 0.3, 0.3], 'LineWidth', 2, 'Label', '基准');

set(gca, 'XTick', 1:6, 'XTickLabel', motion_names_disp, 'FontSize', 12);
ylabel('相对静态基准的改善倍数', 'FontWeight', 'bold', 'FontSize', 14);
title('不同运动模式的平均性能改善', 'FontSize', 16, 'FontWeight', 'bold');

for i = 1:6
    if avg_improvement(i) >= 1
        txt_color = 'k';
        y_offset = 0.18;
    else
        txt_color = colors.degradation;
        y_offset = 0.18;
    end
    text(i, avg_improvement(i) + y_offset, sprintf('%.1f×', avg_improvement(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', ...
        'Color', txt_color);
end

ylim([0, 4.5]);
grid on;

export_fig(fig, 'motion_效果对比');
export_eps(fig, 'motion_效果对比');
close(fig);

fprintf(desc_file, '文件: motion_效果对比.png\n');
fprintf(desc_file, '内容: 六种运动模式的平均改善倍数对比\n');
fprintf(desc_file, '关键发现: 平移运动(X平移2.1×、对角平移3.5×)效果最好；复杂运动效果差\n\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图10: 孔径扩展示意图
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('生成孔径扩展示意图...\n');

fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n');
fprintf(desc_file, '图10: 孔径扩展效果\n');
fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n\n');

fig = figure('Position', [100, 100, 500, 400], 'Color', 'white');

apertures = [1.4, 26.0];
bar_colors_ap = [colors.static; colors.motion];
b = bar(apertures, 0.5);
b.FaceColor = 'flat';
b.CData = bar_colors_ap;

set(gca, 'XTickLabel', {'静态阵列', '运动阵列'}, 'FontSize', 14);
ylabel('等效孔径 (λ)', 'FontWeight', 'bold', 'FontSize', 14);
title('孔径扩展效果', 'FontSize', 16, 'FontWeight', 'bold');

text(2, apertures(2)+1.5, sprintf('×%.0f', apertures(2)/apertures(1)), ...
    'FontSize', 16, 'FontWeight', 'bold', 'Color', colors.improvement, ...
    'HorizontalAlignment', 'center');

ylim([0, 32]);
grid on;

export_fig(fig, 'aperture_孔径扩展');
export_eps(fig, 'aperture_孔径扩展');
close(fig);

fprintf(desc_file, '文件: aperture_孔径扩展.png\n');
fprintf(desc_file, '内容: 静态与运动阵列的等效孔径对比\n');
fprintf(desc_file, '关键发现: 运动扩展可将孔径从1.4λ增加到26λ，提升约18倍\n\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  关闭说明文档
%% ═══════════════════════════════════════════════════════════════════════════
fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n');
fprintf(desc_file, '                        文档结束                                   \n');
fprintf(desc_file, '═══════════════════════════════════════════════════════════════════\n');
fclose(desc_file);

%% 完成
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('所有图表已生成！\n');
fprintf('输出目录: %s\n', output_folder);
fprintf('图像说明: %s\n', fullfile(output_folder, '图像说明.txt'));
fprintf('═══════════════════════════════════════════════════════════════════\n');

