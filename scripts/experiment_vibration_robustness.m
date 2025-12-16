%% ═══════════════════════════════════════════════════════════════════════════
%  实验：飞行震动鲁棒性测试 v6.0 (多频段对比)
%  验证：不同工作频率下，阵元相对位置不变，但存在轨迹偏差
%  对比：S波段(3GHz, λ=10cm) vs VHF米波(300MHz, λ=1m)
%  算法：时间平滑MUSIC
%% ═══════════════════════════════════════════════════════════════════════════
clear; clc; close all;

addpath('asset');

% 创建输出文件夹
script_name = 'experiment_vibration_multiband';
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
output_folder = fullfile('validation_results', [script_name '_' timestamp]);
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

log_file = fullfile(output_folder, 'experiment_log.txt');
diary(log_file);

fprintf('============================================================\n');
fprintf('  飞行震动鲁棒性实验 v6.0 (多频段对比: S波段 vs VHF米波)\n');
fprintf('  验证：米波雷达对飞行震动的更强鲁棒性\n');
fprintf('============================================================\n\n');
fprintf('输出目录: %s\n\n', output_folder);

%% 多频段参数
c = physconst('LightSpeed');

% 频段定义
band_names = {'S波段', 'VHF米波'};
band_fc = [3e9, 300e6];  % Hz
band_lambda = c ./ band_fc;  % m
band_colors = [0.2, 0.4, 0.8; 0.8, 0.3, 0.2];

num_bands = length(band_names);

fprintf('【频段配置】\n');
for b = 1:num_bands
    fprintf('  %s: fc=%.0f MHz, lambda=%.1f cm\n', ...
        band_names{b}, band_fc(b)/1e6, band_lambda(b)*100);
end
fprintf('\n');

%% 公共参数
target_phi = 60;
target_range = 500;
snr_db = 10;
num_elements = 8;

% 运动参数
v = 5;  % m/s
T_chirp = 50e-3;
num_snapshots = 16;
T_obs = num_snapshots * T_chirp;
t_axis = (0:num_snapshots-1) * T_chirp;

% 震动参数 - 使用物理单位(cm)
vibration_cm_range = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10];  % cm
num_trials = 30;

fprintf('【实验设置】\n');
fprintf('  阵列: %d元ULA\n', num_elements);
fprintf('  目标: phi=%.0f deg\n', target_phi);
fprintf('  Chirp周期: %.0f ms\n', T_chirp*1000);
fprintf('  快拍数: %d, 观测时间: %.1f s\n', num_snapshots, T_obs);
fprintf('  震动范围: 0-%.0f cm\n', max(vibration_cm_range));
fprintf('  蒙特卡洛试验: %d次\n', num_trials);
fprintf('  SNR: %d dB\n\n', snr_db);

%% 主实验
fprintf('============================================================\n');
fprintf('开始多频段对比实验\n');
fprintf('============================================================\n\n');

% 结果存储
results_rmse = zeros(num_bands, length(vibration_cm_range));
results_bias = zeros(num_bands, length(vibration_cm_range));
results_vib_lambda = zeros(num_bands, length(vibration_cm_range));

for b = 1:num_bands
    lambda = band_lambda(b);
    fc = band_fc(b);
    d = lambda / 2;
    
    fprintf('【%s (lambda=%.1f cm)】\n', band_names{b}, lambda*100);
    
    % 创建阵列 (y方向ULA)
    y_pos = ((0:num_elements-1) - (num_elements-1)/2) * d;
    elements = [zeros(num_elements, 1), y_pos', zeros(num_elements, 1)];
    
    radar_params = struct('fc', fc, 'lambda', lambda);
    
    % 目标
    target_pos = target_range * [cosd(target_phi), sind(target_phi), 0];
    
    % 搜索网格
    phi_search = 30:0.5:90;
    search_grid.phi = phi_search;
    est_options.search_mode = '1d';
    
    fprintf('震动(cm) | 震动(lambda) | RMSE_phi\n');
    fprintf('---------|--------------|----------\n');
    
    for vib_idx = 1:length(vibration_cm_range)
        vib_cm = vibration_cm_range(vib_idx);
        vib_std = vib_cm / 100;  % 转换为米
        
        results_vib_lambda(b, vib_idx) = vib_std / lambda;
        
        errors_phi = zeros(1, num_trials);
        
        for trial = 1:num_trials
            rng(trial * 1000 + vib_idx + b * 10000);
            
            % 震动偏差
            vibration_offset = randn(num_snapshots, 1) * vib_std;
            
            % 实际阵列（带震动）
            array_actual = ArrayPlatform(elements, 1, 1:num_elements);
            vib_offset_copy = vibration_offset;
            t_axis_copy = t_axis;
            v_copy = v;
            array_actual.set_trajectory(@(t) get_vibrated_trajectory(t, v_copy, t_axis_copy, vib_offset_copy));
            
            % 生成信号
            target_obj = Target(target_pos, [0,0,0], 1);
            sig_gen = SignalGeneratorSimple(radar_params, array_actual, {target_obj});
            snapshots = sig_gen.generate_snapshots(t_axis, snr_db);
            
            % 理论阵列（用于估计）
            array_theoretical = ArrayPlatform(elements, 1, 1:num_elements);
            array_theoretical.set_trajectory(@(t) struct('position', [0, v*t, 0], 'orientation', [0,0,0]));
            
            % DOA估计
            estimator = DoaEstimatorSynthetic(array_theoretical, radar_params);
            try
                [~, peaks, ~] = estimator.estimate(snapshots, t_axis, search_grid, 1, est_options);
                errors_phi(trial) = peaks.phi(1) - target_phi;
            catch
                errors_phi(trial) = NaN;
            end
        end
        
        valid = errors_phi(~isnan(errors_phi));
        if ~isempty(valid)
            results_rmse(b, vib_idx) = sqrt(mean(valid.^2));
            results_bias(b, vib_idx) = mean(valid);
        else
            results_rmse(b, vib_idx) = NaN;
            results_bias(b, vib_idx) = NaN;
        end
        
        fprintf('  %5.1f   |    %.4f    | %5.2f deg\n', ...
            vib_cm, vib_std/lambda, results_rmse(b, vib_idx));
    end
    fprintf('\n');
end

%% 绘图
fprintf('============================================================\n');
fprintf('生成结果图表\n');
fprintf('============================================================\n\n');

fig = figure('Position', [100, 100, 1400, 500], 'Color', 'white');
set(gcf, 'DefaultAxesFontName', 'SimHei');

% 子图1: RMSE vs 震动幅度(cm) - 物理单位对比
subplot(1, 3, 1);
hold on;
for b = 1:num_bands
    plot(vibration_cm_range, results_rmse(b,:), '-o', 'LineWidth', 2, ...
        'MarkerSize', 8, 'MarkerFaceColor', band_colors(b,:), ...
        'Color', band_colors(b,:), 'DisplayName', band_names{b});
end
yline(1, 'g--', 'LineWidth', 1.5, 'DisplayName', '1 deg 线');
yline(2, 'Color', [0.8,0.8,0], 'LineStyle', '--', 'LineWidth', 1.5, 'DisplayName', '2 deg 线');
hold off;
xlabel('震动标准差 (cm)', 'FontWeight', 'bold');
ylabel('RMSE (deg)', 'FontWeight', 'bold');
title('(a) RMSE vs 物理震动幅度', 'FontWeight', 'bold');
legend('Location', 'northwest');
grid on;
xlim([0, max(vibration_cm_range)*1.05]);
ylim([0, 15]);

% 子图2: RMSE vs 震动幅度(lambda) - 电学单位对比
subplot(1, 3, 2);
hold on;
for b = 1:num_bands
    plot(results_vib_lambda(b,:), results_rmse(b,:), '-o', 'LineWidth', 2, ...
        'MarkerSize', 8, 'MarkerFaceColor', band_colors(b,:), ...
        'Color', band_colors(b,:), 'DisplayName', band_names{b});
end
yline(1, 'g--', 'LineWidth', 1.5, 'DisplayName', '1 deg 线');
yline(2, 'Color', [0.8,0.8,0], 'LineStyle', '--', 'LineWidth', 1.5, 'DisplayName', '2 deg 线');
hold off;
xlabel('震动标准差 (lambda)', 'FontWeight', 'bold');
ylabel('RMSE (deg)', 'FontWeight', 'bold');
title('(b) RMSE vs 电相位震动', 'FontWeight', 'bold');
legend('Location', 'northwest');
grid on;
xlim([0, 0.15]);
ylim([0, 15]);

% 子图3: 容许震动对比（条形图）
subplot(1, 3, 3);
tolerance_1deg = zeros(1, num_bands);
tolerance_2deg = zeros(1, num_bands);

for b = 1:num_bands
    idx_1deg = find(results_rmse(b,:) < 1, 1, 'last');
    idx_2deg = find(results_rmse(b,:) < 2, 1, 'last');
    
    if ~isempty(idx_1deg)
        tolerance_1deg(b) = vibration_cm_range(idx_1deg);
    end
    if ~isempty(idx_2deg)
        tolerance_2deg(b) = vibration_cm_range(idx_2deg);
    end
end

bar_data = [tolerance_1deg; tolerance_2deg]';
bar_handle = bar(1:num_bands, bar_data, 'grouped');
bar_handle(1).FaceColor = [0.3, 0.7, 0.3];
bar_handle(2).FaceColor = [0.9, 0.7, 0.2];

set(gca, 'XTick', 1:num_bands, 'XTickLabel', band_names);
ylabel('容许震动 (cm)', 'FontWeight', 'bold');
title('(c) 最大容许震动对比', 'FontWeight', 'bold');
legend({'RMSE<1 deg', 'RMSE<2 deg'}, 'Location', 'northwest');
grid on;

% 添加数值标签
hold on;
for b = 1:num_bands
    if tolerance_1deg(b) > 0
        text(b-0.15, tolerance_1deg(b)+0.3, sprintf('%.1f', tolerance_1deg(b)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    end
    if tolerance_2deg(b) > 0
        text(b+0.15, tolerance_2deg(b)+0.3, sprintf('%.1f', tolerance_2deg(b)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    end
end
hold off;

% 添加无人机震动参考线
yline(1, 'r:', 'LineWidth', 2, 'Label', '典型无人机振动上限');

sgtitle('多频段雷达抗震动性能对比', 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig, fullfile(output_folder, 'fig_多频段抗震动对比.png'));
saveas(fig, fullfile(output_folder, 'fig_多频段抗震动对比.eps'), 'epsc');

%% 输出结论
fprintf('\n============================================================\n');
fprintf('                        实验结论\n');
fprintf('============================================================\n\n');

fprintf('【各频段抗震动能力】\n');
fprintf('频段     | 波长lambda | RMSE<1deg容许 | RMSE<2deg容许 | 阵列尺寸\n');
fprintf('---------|------------|---------------|---------------|----------\n');

for b = 1:num_bands
    array_size = (num_elements-1) * band_lambda(b) / 2 * 100;  % 阵列尺寸(cm)
    fprintf('%-8s | %7.1f cm |   %5.1f cm    |   %5.1f cm    | %6.0f cm\n', ...
        band_names{b}, band_lambda(b)*100, tolerance_1deg(b), tolerance_2deg(b), array_size);
end

fprintf('\n【关键发现】\n');
if tolerance_1deg(1) > 0
    improvement_1deg = tolerance_1deg(2) / tolerance_1deg(1);
else
    improvement_1deg = Inf;
end
if tolerance_2deg(1) > 0
    improvement_2deg = tolerance_2deg(2) / tolerance_2deg(1);
else
    improvement_2deg = Inf;
end
fprintf('  米波相比S波段，容许震动提升: %.0fx (1 deg精度) / %.0fx (2 deg精度)\n', ...
    improvement_1deg, improvement_2deg);

fprintf('\n【实际应用指导】\n');
fprintf('  典型无人机振动: 0.1-1 cm RMS\n');

if tolerance_1deg(2) >= 1
    fprintf('  [OK] VHF米波雷达可容忍典型飞行震动 (容许%.1fcm, 无人机约1cm)\n', ...
        tolerance_1deg(2));
else
    fprintf('  [!] VHF米波雷达需轻微振动抑制\n');
end

if tolerance_1deg(1) >= 1
    fprintf('  [OK] S波段雷达可容忍典型飞行震动\n');
else
    fprintf('  [X] S波段雷达需要振动抑制措施 (容许%.1fcm, 无人机约1cm)\n', ...
        tolerance_1deg(1));
end

fprintf('\n【米波雷达优势与代价】\n');
fprintf('  [+] 优势: 抗震动能力强，对平台要求低\n');
fprintf('  [-] 代价: 阵列物理尺寸大 (8元ULA: S波段%.0fcm vs 米波%.0fcm)\n', ...
    (num_elements-1) * band_lambda(1) / 2 * 100, ...
    (num_elements-1) * band_lambda(2) / 2 * 100);

%% 保存
results_save.rmse = results_rmse;
results_save.bias = results_bias;
results_save.vib_lambda = results_vib_lambda;
results_save.band_names = band_names;
results_save.band_lambda = band_lambda;
results_save.vibration_cm = vibration_cm_range;
save(fullfile(output_folder, 'experiment_results.mat'), 'results_save');

fprintf('\n实验完成！结果保存在: %s\n', output_folder);
diary off;

%% ═══════════════════════════════════════════════════════════════════════════
%  辅助函数
%% ═══════════════════════════════════════════════════════════════════════════
function traj = get_vibrated_trajectory(t, v, t_axis, vib_offset)
    [~, idx] = min(abs(t_axis - t));
    if idx > length(vib_offset)
        idx = length(vib_offset);
    end
    actual_y = v * t + vib_offset(idx);
    traj = struct('position', [0, actual_y, 0], 'orientation', [0, 0, 0]);
end
