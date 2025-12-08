%% ═══════════════════════════════════════════════════════════════════════════
%  论文图表生成脚本
%  生成适合学术论文的高质量可视化图表
%% ═══════════════════════════════════════════════════════════════════════════
clear; clc; close all;

% 添加类文件路径
addpath('asset');

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║              论文图表生成器                                    ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

% 加载实验结果
load('validation_results/comprehensive_test_results.mat');

output_dir = 'validation_results/paper_figures';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% 设置论文风格
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultLineLineWidth', 1.5);

% 颜色方案（专业配色）
colors = struct();
colors.static = [0.2, 0.2, 0.2];       % 深灰
colors.x_trans = [0.85, 0.33, 0.1];    % 橙红
colors.y_trans = [0.0, 0.45, 0.74];    % 蓝色
colors.rotation = [0.47, 0.67, 0.19];  % 绿色
colors.combined = [0.49, 0.18, 0.56];  % 紫色
color_array = [colors.static; colors.x_trans; colors.y_trans; colors.rotation; colors.combined];

snr_range = results.snr_range;
motion_names = {'静态', 'x平移', 'y平移', '旋转', '平移+旋转'};
motion_names_en = {'Static', 'X-Translation', 'Y-Translation', 'Rotation', 'Trans.+Rot.'};

%% ═══════════════════════════════════════════════════════════════════════════
%  图1: RMSE vs SNR 对比曲线（核心结果图）
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('生成图1: RMSE vs SNR 对比曲线...\n');

fig1 = figure('Position', [100, 100, 800, 600], 'Color', 'white');

% 选择ULA-8作为代表
arr_idx = 1;

hold on;
markers = {'o', 's', 'd', '^', 'v'};
for mot_idx = 1:5
    rmse_curve = squeeze(results.rmse(arr_idx, mot_idx, :));
    plot(snr_range, rmse_curve, ['-' markers{mot_idx}], ...
        'Color', color_array(mot_idx, :), ...
        'MarkerFaceColor', color_array(mot_idx, :), ...
        'MarkerSize', 8, ...
        'DisplayName', motion_names_en{mot_idx});
end
hold off;

set(gca, 'YScale', 'log');
xlabel('SNR (dB)', 'FontWeight', 'bold');
ylabel('RMSE (°)', 'FontWeight', 'bold');
title('8-Element ULA: DOA Estimation RMSE vs SNR', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 10);
grid on;
box on;
ylim([0.01, 100]);
xlim([snr_range(1)-1, snr_range(end)+1]);

% 添加注释框
annotation('textbox', [0.15, 0.15, 0.25, 0.12], ...
    'String', {'\bf Key Finding:', 'Motion arrays show', '50-100x improvement', 'at low SNR'}, ...
    'FontSize', 9, 'BackgroundColor', [1 1 0.9], 'EdgeColor', [0.5 0.5 0.5]);

saveas(fig1, fullfile(output_dir, 'fig1_rmse_vs_snr.png'));
saveas(fig1, fullfile(output_dir, 'fig1_rmse_vs_snr.eps'), 'epsc');
fprintf('  → 保存: fig1_rmse_vs_snr.png/eps\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图2: 孔径扩展效果对比（柱状图）
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('生成图2: 孔径扩展效果对比...\n');

fig2 = figure('Position', [100, 100, 900, 500], 'Color', 'white');

% 提取孔径数据
aperture_data = results.aperture';  % [运动模式 x 阵列]

% 选择几个代表性阵列
selected_arrays = [1, 2, 4, 6, 8];  % ULA, URA, L, 圆阵, Y阵
array_names_short = {'ULA-8', 'URA-3×3', 'L-Array', 'Circular', 'Y-Array'};

aperture_selected = aperture_data(:, selected_arrays);

b = bar(aperture_selected', 'grouped');
for i = 1:5
    b(i).FaceColor = color_array(i, :);
end

set(gca, 'XTickLabel', array_names_short);
xlabel('Array Configuration', 'FontWeight', 'bold');
ylabel('Synthetic Aperture (λ)', 'FontWeight', 'bold');
title('Aperture Extension by Motion Mode', 'FontSize', 14, 'FontWeight', 'bold');
legend(motion_names_en, 'Location', 'northwest', 'FontSize', 9);
grid on;
box on;

% 添加数值标签
ylim([0, 35]);

saveas(fig2, fullfile(output_dir, 'fig2_aperture_comparison.png'));
saveas(fig2, fullfile(output_dir, 'fig2_aperture_comparison.eps'), 'epsc');
fprintf('  → 保存: fig2_aperture_comparison.png/eps\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图3: 低SNR改善倍数（雷达图/蛛网图）
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('生成图3: 低SNR改善倍数...\n');

fig3 = figure('Position', [100, 100, 700, 600], 'Color', 'white');

% 计算改善倍数 (静态 / 运动)
snr_idx_low = 1;  % -15dB
improvement_ratio = zeros(length(selected_arrays), 4);

for i = 1:length(selected_arrays)
    arr = selected_arrays(i);
    static_rmse = results.rmse(arr, 1, snr_idx_low);
    for mot = 2:5
        motion_rmse = results.rmse(arr, mot, snr_idx_low);
        improvement_ratio(i, mot-1) = static_rmse / max(motion_rmse, 0.01);
    end
end

% 极坐标雷达图
theta_radar = linspace(0, 2*pi, length(selected_arrays)+1);
theta_radar = theta_radar(1:end-1);

polaraxes;
hold on;
for mot = 1:4
    values = [improvement_ratio(:, mot); improvement_ratio(1, mot)];
    angles = [theta_radar, theta_radar(1)];
    polarplot(angles, values, '-o', 'Color', color_array(mot+1, :), ...
        'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', color_array(mot+1, :));
end
hold off;

thetaticks(rad2deg(theta_radar));
thetaticklabels(array_names_short);
title(sprintf('RMSE Improvement Ratio at SNR = %d dB', snr_range(snr_idx_low)), ...
    'FontSize', 14, 'FontWeight', 'bold');
legend(motion_names_en(2:5), 'Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 9);

saveas(fig3, fullfile(output_dir, 'fig3_improvement_radar.png'));
fprintf('  → 保存: fig3_improvement_radar.png\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图4: 静态vs运动阵列直接对比（双Y轴）
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('生成图4: 静态vs运动阵列直接对比...\n');

fig4 = figure('Position', [100, 100, 800, 500], 'Color', 'white');

% 选择y平移作为代表
mot_static = 1;
mot_motion = 3;  % y平移

% 选择多个阵列
arr_indices = [1, 2, 6];  % ULA, URA, 圆阵
arr_labels = {'ULA-8', 'URA-3×3', 'Circular-8'};
line_styles = {'-', '--', ':'};

subplot(1, 2, 1);
hold on;
for i = 1:length(arr_indices)
    arr = arr_indices(i);
    rmse_static = squeeze(results.rmse(arr, mot_static, :));
    plot(snr_range, rmse_static, line_styles{i}, 'Color', colors.static, ...
        'LineWidth', 2, 'DisplayName', [arr_labels{i} ' (Static)']);
end
hold off;
set(gca, 'YScale', 'log');
xlabel('SNR (dB)');
ylabel('RMSE (°)');
title('(a) Static Arrays', 'FontWeight', 'bold');
legend('Location', 'northeast');
grid on;
ylim([0.01, 100]);

subplot(1, 2, 2);
hold on;
for i = 1:length(arr_indices)
    arr = arr_indices(i);
    rmse_motion = squeeze(results.rmse(arr, mot_motion, :));
    plot(snr_range, rmse_motion, line_styles{i}, 'Color', colors.y_trans, ...
        'LineWidth', 2, 'DisplayName', [arr_labels{i} ' (Motion)']);
end
hold off;
set(gca, 'YScale', 'log');
xlabel('SNR (dB)');
ylabel('RMSE (°)');
title('(b) Motion Arrays (Y-Translation)', 'FontWeight', 'bold');
legend('Location', 'northeast');
grid on;
ylim([0.01, 100]);

sgtitle('Static vs Motion Array Performance', 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig4, fullfile(output_dir, 'fig4_static_vs_motion.png'));
saveas(fig4, fullfile(output_dir, 'fig4_static_vs_motion.eps'), 'epsc');
fprintf('  → 保存: fig4_static_vs_motion.png/eps\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图5: 运动轨迹与合成孔径可视化
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('生成图5: 运动轨迹与合成孔径可视化...\n');

fig5 = figure('Position', [100, 100, 1200, 400], 'Color', 'white');

% 参数
c = physconst('LightSpeed');
fc = 3e9;
lambda = c / fc;
d = lambda / 2;
v = 5;
T_obs = 0.5;
num_snapshots = 64;
t_axis = linspace(0, T_obs, num_snapshots);

% 创建ULA
x_pos = ((0:7) - 3.5) * d;
elements = [x_pos', zeros(8,1), zeros(8,1)];
array = ArrayPlatform(elements, 1, 1:8);

% 子图1: 静态
subplot(1, 3, 1);
pos_static = array.get_mimo_virtual_positions(0);
scatter(pos_static(:,1)/lambda, pos_static(:,2)/lambda, 100, 'filled', ...
    'MarkerFaceColor', colors.static);
xlabel('x (λ)');
ylabel('y (λ)');
title('(a) Static Array', 'FontWeight', 'bold');
axis equal;
xlim([-5, 5]);
ylim([-2, 30]);
grid on;
text(0, -1.5, sprintf('Aperture: %.1fλ', max(pos_static(:,1))-min(pos_static(:,1))/lambda), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% 子图2: y平移
subplot(1, 3, 2);
array.set_trajectory(@(t) struct('position', [0, v*t, 0], 'orientation', [0,0,0]));
all_pos = [];
for k = 1:num_snapshots
    pos_k = array.get_mimo_virtual_positions(t_axis(k));
    all_pos = [all_pos; pos_k];
end
scatter(all_pos(:,1)/lambda, all_pos(:,2)/lambda, 10, 'filled', ...
    'MarkerFaceColor', colors.y_trans, 'MarkerFaceAlpha', 0.5);
xlabel('x (λ)');
ylabel('y (λ)');
title('(b) Y-Translation (v=5m/s)', 'FontWeight', 'bold');
axis equal;
xlim([-5, 5]);
ylim([-2, 30]);
grid on;

% 绘制轨迹
hold on;
quiver(0, 0, 0, v*T_obs/lambda, 0, 'Color', 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);
hold off;

aperture_y = (max(all_pos(:,2)) - min(all_pos(:,2))) / lambda;
text(0, -1.5, sprintf('Aperture: %.1fλ', aperture_y), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% 子图3: 旋转
subplot(1, 3, 3);
omega_deg = 90;
array.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_deg*t/T_obs]));
all_pos = [];
for k = 1:num_snapshots
    pos_k = array.get_mimo_virtual_positions(t_axis(k));
    all_pos = [all_pos; pos_k];
end
scatter(all_pos(:,1)/lambda, all_pos(:,2)/lambda, 10, 'filled', ...
    'MarkerFaceColor', colors.rotation, 'MarkerFaceAlpha', 0.5);
xlabel('x (λ)');
ylabel('y (λ)');
title('(c) Rotation (90°)', 'FontWeight', 'bold');
axis equal;
xlim([-5, 5]);
ylim([-5, 5]);
grid on;

aperture_rot = sqrt((max(all_pos(:,1))-min(all_pos(:,1)))^2 + ...
                    (max(all_pos(:,2))-min(all_pos(:,2)))^2) / lambda;
text(0, -4, sprintf('Aperture: %.1fλ', aperture_rot), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold');

sgtitle('Virtual Array Positions Under Different Motion Modes', 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig5, fullfile(output_dir, 'fig5_motion_trajectory.png'));
saveas(fig5, fullfile(output_dir, 'fig5_motion_trajectory.eps'), 'epsc');
fprintf('  → 保存: fig5_motion_trajectory.png/eps\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图6: 阵列形状图（更美观版本）
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('生成图6: 阵列形状图...\n');

fig6 = figure('Position', [100, 100, 1200, 600], 'Color', 'white');

array_creators = {
    @() create_ula(8, d), 'ULA-8';
    @() create_ura(3, 3, d), 'URA-3×3';
    @() create_L_array(4, 4, d), 'L-Array';
    @() create_T_array(5, 5, d), 'T-Array';
    @() create_circular_array(8, lambda), 'Circular-8';
    @() create_Y_array(4, d), 'Y-Array'
};

for i = 1:6
    subplot(2, 3, i);
    arr = array_creators{i, 1}();
    pos = arr.get_mimo_virtual_positions(0);
    
    scatter(pos(:,1)*1000, pos(:,2)*1000, 150, 'filled', ...
        'MarkerFaceColor', [0.2, 0.4, 0.8], ...
        'MarkerEdgeColor', [0.1, 0.2, 0.5], ...
        'LineWidth', 1.5);
    
    xlabel('x (mm)');
    ylabel('y (mm)');
    title(array_creators{i, 2}, 'FontWeight', 'bold', 'FontSize', 12);
    axis equal;
    grid on;
    
    % 自适应坐标范围
    range_max = max(abs([pos(:,1); pos(:,2)])) * 1000 * 1.3;
    if range_max < 1
        range_max = 100;
    end
    xlim([-range_max, range_max]);
    ylim([-range_max, range_max]);
end

sgtitle('Array Configurations', 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig6, fullfile(output_dir, 'fig6_array_shapes.png'));
saveas(fig6, fullfile(output_dir, 'fig6_array_shapes.eps'), 'epsc');
fprintf('  → 保存: fig6_array_shapes.png/eps\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  图7: MUSIC谱对比（3D展示）
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('生成图7: MUSIC谱对比...\n');

% 重新计算MUSIC谱用于展示
target_phi = 30;
target_range = 500;
radar_params = struct('fc', fc, 'lambda', lambda);

x_pos = ((0:7) - 3.5) * d;
elements = [x_pos', zeros(8,1), zeros(8,1)];

% 静态阵列谱
array_static = ArrayPlatform(elements, 1, 1:8);
array_static.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

target_pos = target_range * [cosd(target_phi), sind(target_phi), 0];
target = Target(target_pos, [0,0,0], 1);
sig_gen = SignalGeneratorSimple(radar_params, array_static, {target});
snapshots_static = sig_gen.generate_snapshots(t_axis, 10);  % 10dB SNR

phi_search = 0:0.2:90;
search_grid = struct('phi', phi_search);

estimator = DoaEstimatorSynthetic(array_static, radar_params);
[spectrum_static, ~, ~] = estimator.estimate(snapshots_static, t_axis, search_grid, 1);

% 运动阵列谱
array_motion = ArrayPlatform(elements, 1, 1:8);
array_motion.set_trajectory(@(t) struct('position', [0, v*t, 0], 'orientation', [0,0,0]));

sig_gen_motion = SignalGeneratorSimple(radar_params, array_motion, {target});
snapshots_motion = sig_gen_motion.generate_snapshots(t_axis, 10);

estimator_motion = DoaEstimatorSynthetic(array_motion, radar_params);
[spectrum_motion, ~, ~] = estimator_motion.estimate(snapshots_motion, t_axis, search_grid, 1);

% 绘图
fig7 = figure('Position', [100, 100, 1000, 400], 'Color', 'white');

subplot(1, 2, 1);
spectrum_static_db = 10*log10(spectrum_static / max(spectrum_static));
plot(phi_search, spectrum_static_db, 'Color', colors.static, 'LineWidth', 2);
hold on;
xline(target_phi, 'r--', 'LineWidth', 1.5);
hold off;
xlabel('Azimuth φ (°)');
ylabel('Normalized Power (dB)');
title('(a) Static Array MUSIC Spectrum', 'FontWeight', 'bold');
xlim([0, 90]);
ylim([-40, 0]);
grid on;
legend({'Spectrum', 'True Target'}, 'Location', 'southwest');

subplot(1, 2, 2);
spectrum_motion_db = 10*log10(spectrum_motion / max(spectrum_motion));
plot(phi_search, spectrum_motion_db, 'Color', colors.y_trans, 'LineWidth', 2);
hold on;
xline(target_phi, 'r--', 'LineWidth', 1.5);
hold off;
xlabel('Azimuth φ (°)');
ylabel('Normalized Power (dB)');
title('(b) Motion Array MUSIC Spectrum', 'FontWeight', 'bold');
xlim([0, 90]);
ylim([-40, 0]);
grid on;
legend({'Spectrum', 'True Target'}, 'Location', 'southwest');

sgtitle('MUSIC Spectrum Comparison (SNR = 10 dB)', 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig7, fullfile(output_dir, 'fig7_music_spectrum.png'));
saveas(fig7, fullfile(output_dir, 'fig7_music_spectrum.eps'), 'epsc');
fprintf('  → 保存: fig7_music_spectrum.png/eps\n');

%% 完成
fprintf('\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('所有论文图表已生成完毕!\n');
fprintf('输出目录: %s/\n', output_dir);
fprintf('═══════════════════════════════════════════════════════════════════\n');

%% 辅助函数
function array = create_ula(num_elements, spacing)
    x_pos = ((0:num_elements-1) - (num_elements-1)/2) * spacing;
    elements = [x_pos', zeros(num_elements,1), zeros(num_elements,1)];
    array = ArrayPlatform(elements, 1, 1:num_elements);
end

function array = create_ura(num_x, num_y, spacing)
    elements = [];
    for iy = 1:num_y
        for ix = 1:num_x
            x = (ix - 1 - (num_x-1)/2) * spacing;
            y = (iy - 1 - (num_y-1)/2) * spacing;
            elements = [elements; x, y, 0];
        end
    end
    array = ArrayPlatform(elements, 1, 1:size(elements,1));
end

function array = create_L_array(num_x, num_y, spacing)
    elements = [];
    for i = 1:num_x
        elements = [elements; (i-1)*spacing, 0, 0];
    end
    for i = 2:num_y
        elements = [elements; 0, (i-1)*spacing, 0];
    end
    array = ArrayPlatform(elements, 1, 1:size(elements,1));
end

function array = create_T_array(num_h, num_v, spacing)
    elements = [];
    for i = 1:num_h
        x = (i-1-(num_h-1)/2) * spacing;
        elements = [elements; x, 0, 0];
    end
    for i = 2:num_v
        elements = [elements; 0, -(i-1)*spacing, 0];
    end
    array = ArrayPlatform(elements, 1, 1:size(elements,1));
end

function array = create_circular_array(num_elements, radius)
    elements = [];
    for i = 1:num_elements
        angle = 2*pi*(i-1)/num_elements;
        elements = [elements; radius*cos(angle), radius*sin(angle), 0];
    end
    array = ArrayPlatform(elements, 1, 1:num_elements);
end

function array = create_Y_array(arm_length, spacing)
    elements = [0, 0, 0];
    angles = [90, 210, 330];
    for ang = angles
        for i = 1:arm_length-1
            x = i * spacing * cosd(ang);
            y = i * spacing * sind(ang);
            elements = [elements; x, y, 0];
        end
    end
    array = ArrayPlatform(elements, 1, 1:size(elements,1));
end


