%% ═══════════════════════════════════════════════════════════════════════════
%  老数据图像重绘 - 高清版
%  基于 aa.m 原代码，仅放大字体和提高分辨率
%% ═══════════════════════════════════════════════════════════════════════════
clear; clc; close all;

%% ═══════════════════════════════════════════════════════════════════════════
%  配置
%% ═══════════════════════════════════════════════════════════════════════════

% 输出文件夹
output_folder = 'res_hd';
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

% 图像分辨率
dpi_resolution = 300;

% 辅助函数：高清保存图片
save_figure_hd = @(fig, filepath) print(fig, filepath, '-dpng', sprintf('-r%d', dpi_resolution));

fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('  老数据图像重绘 - 高清版 (300 DPI)\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  加载 abc._OLD.mat 数据 (与 aa.m 一致)
%% ═══════════════════════════════════════════════════════════════════════════

fprintf('加载数据: old_need_to_del/abc._OLD.mat\n');
load('old_need_to_del/abc._OLD.mat');

fprintf('数据维度: ang_err = %s\n', mat2str(size(ang_err)));
fprintf('SNR范围: %d ~ %d dB\n', min(snr), max(snr));
fprintf('N_L: %s\n', mat2str(N_L));
fprintf('numRX: %s\n', mat2str(numRX));

% ========== 计算统计值 ==========
% 原数据维度: [2, snr, N_L, pass_Cf, numRX, 50次试验]
% 对第6维（50次蒙特卡洛试验）计算RMSE
ang_err_rmse = sqrt(mean(ang_err.^2, 6));  % RMSE
ang_err_mean = mean(abs(ang_err), 6);      % 平均绝对误差

fprintf('已计算RMSE统计值 (基于50次蒙特卡洛试验)\n');
fprintf('\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图1: 仰角误差图 (大图 - 英文版)
%  还原 aa.m 第 111-151 行
%% ═══════════════════════════════════════════════════════════════════════════

fprintf('生成图1: 仰角误差图 (大图-英文)...\n');

% 原代码: figure('Position', [100 100 800 400])
% 放大尺寸以提高清晰度
fig = figure('Position', [100 100 1000 550], 'Color', 'white');
hold on; grid on; box on;

% 使用RMSE统计值绘图 (对50次试验取RMSE)
plot(snr(9:end), squeeze(ang_err_rmse(1,9:end,1,1,1)),  'o-','MarkerSize', 10, 'LineWidth', 2);
plot(snr(9:end), squeeze(ang_err_rmse(1,9:end,3,1,1)),  's--','MarkerSize', 10, 'LineWidth', 2);
plot(snr(9:end), squeeze(ang_err_rmse(1,9:end,5,1,1)),  'd-.','MarkerSize', 10, 'LineWidth', 2);
plot(snr(9:end), squeeze(ang_err_rmse(1,9:end,10,1,1)), '^:','MarkerSize', 10, 'LineWidth', 2);

% 坐标轴设置 - 字体放大 (原: FontSize 25)
xlabel('SNR (dB)', 'FontSize', 28, 'FontWeight', 'bold');
ylabel('RMSE (°)', 'FontSize', 28, 'FontWeight', 'bold');
set(gca, 'FontSize', 26);
box on;

% 加粗坐标轴
ax = gca;
ax.LineWidth = 2;

% 图例 - 适中大小
biaoqian = {'N_L=1 (Traditional)', 'N_L=3', 'N_L=5', 'N_L=10'};
lgd = legend(biaoqian, 'Location', 'northeast', 'FontSize', 20);
lgd.ItemTokenSize = [35, 16];  % 图例标记尺寸

grid minor;
title('Elevation DOA Estimation RMSE (N=50)', 'FontSize', 28, 'FontWeight', 'bold');
hold off;

% 保存
save_figure_hd(fig, fullfile(output_folder, '仰角误差图_HD.png'));
saveas(fig, fullfile(output_folder, '仰角误差图_HD.eps'), 'epsc');
close(fig);
fprintf('  已保存: 仰角误差图_HD.png (300 DPI)\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图2: 仰角误差图 (小图 - 中文版，所有N_L)
%  还原 aa.m 第 160-193 行
%% ═══════════════════════════════════════════════════════════════════════════

fprintf('生成图2: 仰角误差图 (小图-中文)...\n');

fig = figure('Position', [100 100 1000 550], 'Color', 'white');
hold on; grid on; box on;

% 使用RMSE统计值绘图
for i = N_L
    plot(snr(9:end), squeeze(ang_err_rmse(1,9:end,i,1,1)), 'MarkerSize', 10, 'LineWidth', 1.8);
end

% 坐标轴设置 - 字体放大 (原: FontSize 12)
xlabel('SNR (dB)', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('RMSE (°)', 'FontSize', 18, 'FontWeight', 'bold');
set(gca, 'FontSize', 16);

% 图例
biaoqian = arrayfun(@(x) ['N_L=', num2str(x)], N_L, 'UniformOutput', false);
legend(biaoqian, 'Location', 'northeast', 'FontSize', 14);

grid minor;
title('不同扩展倍数下的仰角估计RMSE (N=50)', 'FontSize', 18, 'FontWeight', 'bold');
hold off;

save_figure_hd(fig, fullfile(output_folder, '仰角误差图小图_HD.png'));
saveas(fig, fullfile(output_folder, '仰角误差图小图_HD.eps'), 'epsc');
close(fig);
fprintf('  已保存: 仰角误差图小图_HD.png (300 DPI)\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图3: 仰角误差图 (高SNR区间)
%  还原 aa.m 第 203-234 行
%% ═══════════════════════════════════════════════════════════════════════════

fprintf('生成图3: 仰角误差图 (高SNR区间)...\n');

fig = figure('Position', [100 100 1000 550], 'Color', 'white');
hold on; grid on; box on;

% 原代码使用 snr(24:end)，但我们的数据snr可能范围不同
% 找到SNR=0的索引
snr_high_idx = find(snr >= 0, 1);
if isempty(snr_high_idx), snr_high_idx = 24; end

plot(snr(snr_high_idx:end), squeeze(ang_err_rmse(1,snr_high_idx:end,1,1,1)),  'o-','MarkerSize', 10, 'LineWidth', 2);
plot(snr(snr_high_idx:end), squeeze(ang_err_rmse(1,snr_high_idx:end,3,1,1)),  's--','MarkerSize', 10, 'LineWidth', 2);
plot(snr(snr_high_idx:end), squeeze(ang_err_rmse(1,snr_high_idx:end,5,1,1)),  'd-.','MarkerSize', 10, 'LineWidth', 2);
plot(snr(snr_high_idx:end), squeeze(ang_err_rmse(1,snr_high_idx:end,10,1,1)), '^:','MarkerSize', 10, 'LineWidth', 2);

set(gca, 'FontSize', 16);
ax = gca;
ax.LineWidth = 2;

% 高SNR小图不需要图例（用于插入大图）
% biaoqian = {'N_L=1 (Traditional)', 'N_L=3', 'N_L=5', 'N_L=10'};
% legend(biaoqian, 'Location', 'northeast', 'FontSize', 14);

grid minor;
hold off;

save_figure_hd(fig, fullfile(output_folder, '仰角误差图_高SNR_HD.png'));
saveas(fig, fullfile(output_folder, '仰角误差图_高SNR_HD.eps'), 'epsc');
close(fig);
fprintf('  已保存: 仰角误差图_高SNR_HD.png (300 DPI)\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图4: DOA误差箱线图
%  还原 aa.m 第 237-283 行 (使用 aaaa.mat)
%% ═══════════════════════════════════════════════════════════════════════════

fprintf('生成图4: DOA误差箱线图...\n');

% 加载 aaaa.mat
load('old_need_to_del/aaaa.mat');

% 原代码: data=abs(squeeze(ang_err(2,1,:,1,1,:))')
data = abs(squeeze(ang_err(2,1,:,1,1,:))');

fig = figure('Position', [100 100 1000 650], 'Color', 'white');

% 绘制箱线图
boxplot(data, 'Colors', 'r', 'Symbol', 'r+', 'Widths', 0.8);
h = findobj(gca, 'Tag', 'Box');

% 设置箱体样式
for j = 1:length(h)
    patch(get(h(j), 'XData'), get(h(j), 'YData'), 'b', ...
        'FaceAlpha', 0, 'EdgeColor', 'b');
end

% 坐标轴设置 - 字体放大 (原: FontSize 16/25)
set(gca, 'FontSize', 20);
xlabel('Expand Number N_L', 'FontSize', 28, 'FontWeight', 'bold');
ylabel('Error (°)', 'FontSize', 28, 'FontWeight', 'bold');
title('Box-plot for DOA Estimation Error', 'FontSize', 28, 'FontWeight', 'bold');

box on;
ax = gca;
ax.LineWidth = 2;

% X轴刻度
xticks(1:3:length(N_L));
xticklabels(N_L(1:3:end));
xlim([0 length(N_L)+1]);

grid on;
set(findobj(gca, 'type', 'line'), 'LineWidth', 4);

save_figure_hd(fig, fullfile(output_folder, 'DOA误差箱线图_HD.png'));
saveas(fig, fullfile(output_folder, 'DOA误差箱线图_HD.eps'), 'epsc');
close(fig);
fprintf('  已保存: DOA误差箱线图_HD.png (300 DPI)\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  完成
%% ═══════════════════════════════════════════════════════════════════════════

fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('所有图像生成完成！\n');
fprintf('输出目录: %s\n', fullfile(pwd, output_folder));
fprintf('═══════════════════════════════════════════════════════════════════\n');

% 列出生成的文件
files = dir(fullfile(output_folder, '*.png'));
fprintf('\n生成的PNG文件 (%d个):\n', length(files));
for i = 1:length(files)
    fprintf('  %s\n', files(i).name);
end
