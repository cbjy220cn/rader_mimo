%% ═══════════════════════════════════════════════════════════════════════════
%  实验：飞行震动鲁棒性测试
%  验证：雷达整体部署时，阵元相对位置不变，但存在轨迹偏差
%  模型：实际位置 = 理论位置 + 随机振动偏差
%% ═══════════════════════════════════════════════════════════════════════════
clear; clc; close all;

% 添加类文件路径
addpath('asset');

% 创建带时间戳的输出文件夹
script_name = 'experiment_vibration_robustness';
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
output_folder = fullfile('validation_results', [script_name '_' timestamp]);
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

% 初始化日志
log_file = fullfile(output_folder, 'experiment_log.txt');
diary(log_file);

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║           飞行震动鲁棒性实验                                    ║\n');
fprintf('║  验证：轨迹偏差对DOA估计精度的影响                             ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');
fprintf('输出目录: %s\n\n', output_folder);

%% 参数设置
c = physconst('LightSpeed');
fc = 3e9;
lambda = c / fc;
d = lambda / 2;

radar_params = struct('fc', fc, 'lambda', lambda);

% 目标参数
target_phi = 30;
target_range = 500;
snr_db = 10;

% 阵列参数
num_elements = 8;
x_pos = ((0:num_elements-1) - (num_elements-1)/2) * d;
elements = [x_pos', zeros(num_elements,1), zeros(num_elements,1)];

% 运动参数
v = 5;  % m/s
T_obs = 0.5;
num_snapshots = 64;
t_axis = linspace(0, T_obs, num_snapshots);

% 震动参数设置（不同程度的位置偏差）
vibration_std_range = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0] * lambda;  % 标准差（单位：波长）
num_trials = 50;

fprintf('【实验设置】\n');
fprintf('  波长: %.2f cm\n', lambda*100);
fprintf('  震动标准差范围: [0, %.2f]λ = [0, %.1f]cm\n', ...
    max(vibration_std_range)/lambda, max(vibration_std_range)*100);
fprintf('  蒙特卡洛试验次数: %d\n', num_trials);
fprintf('  SNR: %d dB\n\n', snr_db);

%% 搜索网格
phi_search = 0:0.1:90;
search_grid = struct('phi', phi_search);

%% 运行实验
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('开始实验\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

results_vibration = struct();
results_vibration.vibration_std = vibration_std_range;
results_vibration.rmse = zeros(size(vibration_std_range));
results_vibration.bias = zeros(size(vibration_std_range));
results_vibration.std = zeros(size(vibration_std_range));
results_vibration.success_rate = zeros(size(vibration_std_range));

% 目标
target_pos = target_range * [cosd(target_phi), sind(target_phi), 0];
target = Target(target_pos, [0,0,0], 1);

fprintf('震动标准差 | RMSE   | 偏差   | 标准差 | 成功率\n');
fprintf('-----------|--------|--------|--------|-------\n');

for vib_idx = 1:length(vibration_std_range)
    vib_std = vibration_std_range(vib_idx);
    
    errors = zeros(1, num_trials);
    estimates = zeros(1, num_trials);
    
    for trial = 1:num_trials
        rng(trial * 100 + vib_idx);
        
        % 创建理论轨迹阵列（用于DOA计算）
        array_theoretical = ArrayPlatform(elements, 1, 1:num_elements);
        array_theoretical.set_trajectory(@(t) struct(...
            'position', [0, v*t, 0], ...
            'orientation', [0, 0, 0]));
        
        % 创建实际轨迹阵列（包含震动偏差）
        % 震动模型：实际位置 = 理论位置 + 随机高斯偏差
        % 偏差在每个时刻独立，但同一时刻所有阵元偏差相同（整体震动）
        vibration_offset = randn(num_snapshots, 3) * vib_std;  % [K x 3] 每时刻的偏差
        
        array_actual = ArrayPlatform(elements, 1, 1:num_elements);
        
        % 生成快拍数据（使用实际位置）
        snapshots = zeros(num_elements, num_snapshots);
        for k = 1:num_snapshots
            t_k = t_axis(k);
            
            % 理论位置
            theoretical_pos = [0, v*t_k, 0];
            % 实际位置 = 理论位置 + 震动偏差
            actual_pos = theoretical_pos + vibration_offset(k, :);
            
            % 设置实际轨迹
            array_actual.set_trajectory(@(t) struct(...
                'position', actual_pos, ...
                'orientation', [0, 0, 0]));
            
            % 获取实际阵元位置
            actual_element_pos = array_actual.get_mimo_virtual_positions(0);
            
            % 计算信号（基于实际位置）
            for elem = 1:num_elements
                range = 2 * norm(target_pos - actual_element_pos(elem, :));
                phase = 2 * pi * range / lambda;
                snapshots(elem, k) = exp(-1j * phase);
            end
        end
        
        % 添加噪声
        signal_power = mean(abs(snapshots(:)).^2);
        noise_power = signal_power / (10^(snr_db/10));
        noise = sqrt(noise_power/2) * (randn(size(snapshots)) + 1j*randn(size(snapshots)));
        snapshots = snapshots + noise;
        
        % DOA估计（使用理论轨迹）
        estimator = DoaEstimatorSynthetic(array_theoretical, radar_params);
        try
            [~, peaks, ~] = estimator.estimate(snapshots, t_axis, search_grid, 1);
            est_phi = peaks.phi(1);
        catch
            est_phi = NaN;
        end
        
        estimates(trial) = est_phi;
        errors(trial) = est_phi - target_phi;
    end
    
    % 统计
    valid_errors = errors(~isnan(errors));
    valid_estimates = estimates(~isnan(estimates));
    
    if ~isempty(valid_errors)
        results_vibration.rmse(vib_idx) = sqrt(mean(valid_errors.^2));
        results_vibration.bias(vib_idx) = mean(valid_errors);
        results_vibration.std(vib_idx) = std(valid_errors);
        results_vibration.success_rate(vib_idx) = length(valid_errors) / num_trials * 100;
    else
        results_vibration.rmse(vib_idx) = NaN;
        results_vibration.bias(vib_idx) = NaN;
        results_vibration.std(vib_idx) = NaN;
        results_vibration.success_rate(vib_idx) = 0;
    end
    
    fprintf('   %.2fλ    | %5.2f° | %+5.2f° | %5.2f° | %5.1f%%\n', ...
        vib_std/lambda, ...
        results_vibration.rmse(vib_idx), ...
        results_vibration.bias(vib_idx), ...
        results_vibration.std(vib_idx), ...
        results_vibration.success_rate(vib_idx));
end

%% 绘图
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('生成结果图表\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

fig = figure('Position', [100, 100, 1200, 500], 'Color', 'white');
set(gcf, 'DefaultAxesFontName', 'SimHei');

% 子图1: RMSE vs 震动幅度
subplot(1, 3, 1);
vib_std_lambda = vibration_std_range / lambda;
plot(vib_std_lambda, results_vibration.rmse, '-o', 'LineWidth', 2, ...
    'MarkerSize', 8, 'MarkerFaceColor', [0.2, 0.4, 0.8]);
xlabel('震动标准差 (λ)', 'FontWeight', 'bold');
ylabel('RMSE (°)', 'FontWeight', 'bold');
title('(a) RMSE vs 震动幅度', 'FontWeight', 'bold');
grid on;
xlim([0, max(vib_std_lambda)*1.05]);

% 找到RMSE < 1度的阈值
threshold_idx = find(results_vibration.rmse > 1, 1);
if ~isempty(threshold_idx)
    hold on;
    xline(vib_std_lambda(threshold_idx-1), 'r--', 'LineWidth', 1.5);
    text(vib_std_lambda(threshold_idx-1)*1.1, max(results_vibration.rmse)*0.8, ...
        sprintf('容许范围\n≈%.2fλ', vib_std_lambda(threshold_idx-1)), ...
        'Color', 'r', 'FontWeight', 'bold');
    hold off;
end

% 子图2: 偏差与标准差
subplot(1, 3, 2);
yyaxis left;
plot(vib_std_lambda, results_vibration.bias, '-s', 'LineWidth', 2, 'MarkerSize', 8);
ylabel('偏差 (°)');

yyaxis right;
plot(vib_std_lambda, results_vibration.std, '-d', 'LineWidth', 2, 'MarkerSize', 8);
ylabel('标准差 (°)');

xlabel('震动标准差 (λ)', 'FontWeight', 'bold');
title('(b) 偏差与标准差', 'FontWeight', 'bold');
legend({'偏差', '标准差'}, 'Location', 'northwest');
grid on;
xlim([0, max(vib_std_lambda)*1.05]);

% 子图3: 成功率
subplot(1, 3, 3);
bar(vib_std_lambda, results_vibration.success_rate, 'FaceColor', [0.3, 0.7, 0.4]);
xlabel('震动标准差 (λ)', 'FontWeight', 'bold');
ylabel('成功率 (%)', 'FontWeight', 'bold');
title('(c) 估计成功率', 'FontWeight', 'bold');
ylim([0, 105]);
grid on;

sgtitle('平台震动对DOA估计的影响', 'FontSize', 14, 'FontWeight', 'bold');

%% 保存结果
save(fullfile(output_folder, 'experiment_results.mat'), 'results_vibration');
saveas(fig, fullfile(output_folder, 'fig_抗震动鲁棒性.png'));
saveas(fig, fullfile(output_folder, 'fig_抗震动鲁棒性.eps'), 'epsc');

%% 输出结论
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                        实验结论                                   \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% 找到容许的最大震动
tolerance_1deg = find(results_vibration.rmse < 1, 1, 'last');
tolerance_05deg = find(results_vibration.rmse < 0.5, 1, 'last');

fprintf('【鲁棒性评估】\n');
fprintf('  RMSE < 1°  时最大容许震动: %.2fλ = %.1f cm\n', ...
    vib_std_lambda(tolerance_1deg), vibration_std_range(tolerance_1deg)*100);
if ~isempty(tolerance_05deg)
    fprintf('  RMSE < 0.5° 时最大容许震动: %.2fλ = %.1f cm\n', ...
        vib_std_lambda(tolerance_05deg), vibration_std_range(tolerance_05deg)*100);
end

fprintf('\n【物理意义】\n');
fprintf('  波长 λ = %.2f cm\n', lambda*100);
fprintf('  容许震动幅度 ≈ %.1f cm (3σ范围 ≈ %.1f cm)\n', ...
    vibration_std_range(tolerance_1deg)*100, ...
    vibration_std_range(tolerance_1deg)*100*3);

fprintf('\n【实际应用指导】\n');
fprintf('  典型无人机振动: 0.1-1 cm RMS\n');
fprintf('  本算法容许范围: 0-%.1f cm RMS\n', vibration_std_range(tolerance_1deg)*100);
if vibration_std_range(tolerance_1deg) > 0.01
    fprintf('  ✅ 算法可容忍典型飞行震动！\n');
else
    fprintf('  ⚠️ 需要振动抑制措施\n');
end

fprintf('\n实验完成！\n');
fprintf('所有结果保存在: %s\n', output_folder);

% 关闭日志
diary off;

