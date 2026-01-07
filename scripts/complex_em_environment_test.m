%% ═══════════════════════════════════════════════════════════════════════════
%  第四章补充实验：复杂电磁环境鲁棒性测试
%  - 基于新代码架构（DoaEstimatorSynthetic + 时间平滑MUSIC）
%  - 测试杂波、干扰、多径等复杂电磁环境对DOA估计的影响
%  - 对比静态/运动阵列在复杂环境下的性能
%% ═══════════════════════════════════════════════════════════════════════════
clear; clc; close all;

% 添加类文件路径
addpath('asset');

% 创建带时间戳的输出文件夹
script_name = 'complex_em_environment_test';
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
output_folder = fullfile('validation_results', [script_name '_' timestamp]);
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

% 初始化日志
log_file = fullfile(output_folder, 'experiment_log.txt');
diary(log_file);

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║  第四章补充实验：复杂电磁环境鲁棒性测试                      ║\n');
fprintf('║  基于时间平滑MUSIC的合成孔径方法                             ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');
fprintf('输出目录: %s\n\n', output_folder);

%% ═══════════════════════════════════════════════════════════════════════════
%  雷达系统参数
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('【雷达系统参数】\n');

c = physconst('LightSpeed');
fc = 3e9;                           % 载频: 3 GHz
lambda = c / fc;                    % 波长
BW = 50e6;                          % 带宽: 50 MHz
T_chirp = 10e-3;                    % Chirp周期
slope = BW / T_chirp;
fs = 100e6;
num_adc_samples = 1024;
range_res = c / (2 * BW);

radar_params = struct();
radar_params.fc = fc;
radar_params.lambda = lambda;
radar_params.c = c;
radar_params.BW = BW;
radar_params.T_chirp = T_chirp;
radar_params.slope = slope;
radar_params.fs = fs;
radar_params.num_samples = num_adc_samples;
radar_params.range_res = range_res;

fprintf('  载频: %.2f GHz (λ = %.2f cm)\n', fc/1e9, lambda*100);
fprintf('  带宽: %.0f MHz, 距离分辨率: %.1f m\n', BW/1e6, range_res);

%% ═══════════════════════════════════════════════════════════════════════════
%  阵列参数
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n【阵列参数】\n');

d_normal = lambda / 2;              % 正常阵元间距: 0.5λ

% 使用8元圆阵（与complex_trajectory_test一致）
num_elements = 8;
R_array = lambda * 0.65;            % 圆阵半径

fprintf('  阵列类型: %d元圆阵, 半径 %.2fλ\n', num_elements, R_array/lambda);

%% ═══════════════════════════════════════════════════════════════════════════
%  目标与运动参数
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n【目标与运动参数】\n');

target_phi = 30;                    % 目标方位角
target_theta = 75;                  % 目标俯仰角
target_range = 500;                 % 目标距离

T_obs = 0.5;                        % 观测时间
num_snapshots = 64;                 % 快拍数
t_axis = linspace(0, T_obs, num_snapshots);

v_linear = 5;                       % 平移速度: 5 m/s

fprintf('  目标: φ=%.0f°, θ=%.0f°, R=%.0fm\n', target_phi, target_theta, target_range);
fprintf('  观测时间: %.1fs, 快拍数: %d\n', T_obs, num_snapshots);
fprintf('  运动速度: %.1f m/s\n', v_linear);

%% ═══════════════════════════════════════════════════════════════════════════
%  实验参数
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n【实验参数】\n');

base_snr = 20;                      % 基准SNR
num_trials = 30;                    % 蒙特卡洛次数

fprintf('  基准SNR: %d dB\n', base_snr);
fprintf('  蒙特卡洛次数: %d\n', num_trials);

% 搜索配置
search_grid_2d.theta = 60:0.5:90;
search_grid_2d.phi = 0:0.5:90;

est_options.use_smart_search = false;
est_options.use_cfar = false;

%% ═══════════════════════════════════════════════════════════════════════════
%  创建阵列函数
%% ═══════════════════════════════════════════════════════════════════════════

% 创建圆阵
function array = create_circular_array(num_elem, radius)
    angles = linspace(0, 2*pi, num_elem + 1);
    angles(end) = [];
    elements = zeros(num_elem, 3);
    elements(:, 1) = radius * cos(angles');
    elements(:, 2) = radius * sin(angles');
    tx_indices = 1;
    rx_indices = 1:num_elem;
    array = ArrayPlatform(elements, tx_indices, rx_indices);
end

%% ═══════════════════════════════════════════════════════════════════════════
%  实验1: 杂波环境测试
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('实验1: 杂波环境对DOA估计的影响\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

cnr_list = [0, 10, 20, 30, 40];  % 杂波噪比 (dB)
fprintf('  杂波噪比(CNR): ');
fprintf('%ddB ', cnr_list);
fprintf('\n');

results_clutter = struct();
results_clutter.cnr = cnr_list;
results_clutter.rmse_motion = zeros(length(cnr_list), 2);  % [theta, phi]
results_clutter.rmse_static = zeros(length(cnr_list), 2);

for cnr_idx = 1:length(cnr_list)
    cnr = cnr_list(cnr_idx);
    
    % ===== 运动阵列 =====
    array_motion = create_circular_array(num_elements, R_array);
    array_motion.set_trajectory(@(t) struct('position', [v_linear*t, 0, 0], 'orientation', [0, 0, 0]));
    
    errors_motion = zeros(num_trials, 2);
    for trial = 1:num_trials
        rng(trial * 1000 + cnr_idx);
        
        % 生成带杂波的信号
        snapshots = generate_signal_with_clutter(radar_params, array_motion, ...
            target_theta, target_phi, target_range, t_axis, base_snr, cnr);
        
        % DOA估计
        estimator = DoaEstimatorSynthetic(array_motion, radar_params);
        [~, peaks, ~] = estimator.estimate(snapshots, t_axis, search_grid_2d, 1, est_options);
        
        errors_motion(trial, :) = [peaks.theta(1) - target_theta, peaks.phi(1) - target_phi];
    end
    
    % ===== 静态阵列 =====
    array_static = create_circular_array(num_elements, R_array);
    % 不设置轨迹，保持静态
    
    errors_static = zeros(num_trials, 2);
    for trial = 1:num_trials
        rng(trial * 2000 + cnr_idx);
        
        snapshots = generate_signal_with_clutter(radar_params, array_static, ...
            target_theta, target_phi, target_range, t_axis, base_snr, cnr);
        
        % 静态MUSIC
        positions = array_static.get_mimo_virtual_positions(0);
        spectrum = music_standard_2d(snapshots, positions, search_grid_2d, lambda, 1);
        [~, idx] = max(spectrum(:));
        [theta_idx, phi_idx] = ind2sub(size(spectrum), idx);
        est_theta = search_grid_2d.theta(theta_idx);
        est_phi = search_grid_2d.phi(phi_idx);
        
        errors_static(trial, :) = [est_theta - target_theta, est_phi - target_phi];
    end
    
    results_clutter.rmse_motion(cnr_idx, :) = sqrt(mean(errors_motion.^2, 1));
    results_clutter.rmse_static(cnr_idx, :) = sqrt(mean(errors_static.^2, 1));
    
    fprintf('  CNR=%ddB: 运动[θ=%.2f°,φ=%.2f°], 静态[θ=%.2f°,φ=%.2f°]\n', cnr, ...
        results_clutter.rmse_motion(cnr_idx, 1), results_clutter.rmse_motion(cnr_idx, 2), ...
        results_clutter.rmse_static(cnr_idx, 1), results_clutter.rmse_static(cnr_idx, 2));
end

%% ═══════════════════════════════════════════════════════════════════════════
%  实验2: 有意干扰测试
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('实验2: 有意干扰对DOA估计的影响\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

jsr_list = [-10, 0, 10, 20, 30];  % 干信比 (dB)
jammer_theta = 80;                 % 干扰源俯仰角
jammer_phi = 60;                   % 干扰源方位角

fprintf('  干信比(JSR): ');
fprintf('%ddB ', jsr_list);
fprintf('\n');
fprintf('  干扰源位置: θ=%.0f°, φ=%.0f°\n', jammer_theta, jammer_phi);

results_jamming = struct();
results_jamming.jsr = jsr_list;
results_jamming.jammer_direction = [jammer_theta, jammer_phi];
results_jamming.rmse_motion = zeros(length(jsr_list), 2);
results_jamming.rmse_static = zeros(length(jsr_list), 2);

for jsr_idx = 1:length(jsr_list)
    jsr = jsr_list(jsr_idx);
    
    % ===== 运动阵列 =====
    array_motion = create_circular_array(num_elements, R_array);
    array_motion.set_trajectory(@(t) struct('position', [v_linear*t, 0, 0], 'orientation', [0, 0, 0]));
    
    errors_motion = zeros(num_trials, 2);
    for trial = 1:num_trials
        rng(trial * 3000 + jsr_idx);
        
        snapshots = generate_signal_with_jamming(radar_params, array_motion, ...
            target_theta, target_phi, target_range, ...
            jammer_theta, jammer_phi, t_axis, base_snr, jsr);
        
        estimator = DoaEstimatorSynthetic(array_motion, radar_params);
        [~, peaks, ~] = estimator.estimate(snapshots, t_axis, search_grid_2d, 1, est_options);
        
        errors_motion(trial, :) = [peaks.theta(1) - target_theta, peaks.phi(1) - target_phi];
    end
    
    % ===== 静态阵列 =====
    array_static = create_circular_array(num_elements, R_array);
    
    errors_static = zeros(num_trials, 2);
    for trial = 1:num_trials
        rng(trial * 4000 + jsr_idx);
        
        snapshots = generate_signal_with_jamming(radar_params, array_static, ...
            target_theta, target_phi, target_range, ...
            jammer_theta, jammer_phi, t_axis, base_snr, jsr);
        
        positions = array_static.get_mimo_virtual_positions(0);
        spectrum = music_standard_2d(snapshots, positions, search_grid_2d, lambda, 1);
        [~, idx] = max(spectrum(:));
        [theta_idx, phi_idx] = ind2sub(size(spectrum), idx);
        est_theta = search_grid_2d.theta(theta_idx);
        est_phi = search_grid_2d.phi(phi_idx);
        
        errors_static(trial, :) = [est_theta - target_theta, est_phi - target_phi];
    end
    
    results_jamming.rmse_motion(jsr_idx, :) = sqrt(mean(errors_motion.^2, 1));
    results_jamming.rmse_static(jsr_idx, :) = sqrt(mean(errors_static.^2, 1));
    
    fprintf('  JSR=%ddB: 运动[θ=%.2f°,φ=%.2f°], 静态[θ=%.2f°,φ=%.2f°]\n', jsr, ...
        results_jamming.rmse_motion(jsr_idx, 1), results_jamming.rmse_motion(jsr_idx, 2), ...
        results_jamming.rmse_static(jsr_idx, 1), results_jamming.rmse_static(jsr_idx, 2));
end

%% ═══════════════════════════════════════════════════════════════════════════
%  实验3: 多径效应测试
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('实验3: 多径效应对DOA估计的影响\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

multipath_gain_list = [-20, -15, -10, -5, -3];  % 多径增益 (dB)
mp_theta_offset = 10;                           % 俯仰角偏移
mp_phi_offset = 20;                             % 方位角偏移

fprintf('  多径增益: ');
fprintf('%ddB ', multipath_gain_list);
fprintf('\n');
fprintf('  多径方向偏移: Δθ=%.0f°, Δφ=%.0f°\n', mp_theta_offset, mp_phi_offset);

results_multipath = struct();
results_multipath.multipath_gain = multipath_gain_list;
results_multipath.direction_offset = [mp_theta_offset, mp_phi_offset];
results_multipath.rmse_motion = zeros(length(multipath_gain_list), 2);
results_multipath.rmse_static = zeros(length(multipath_gain_list), 2);

for mp_idx = 1:length(multipath_gain_list)
    mp_gain = multipath_gain_list(mp_idx);
    
    % ===== 运动阵列 =====
    array_motion = create_circular_array(num_elements, R_array);
    array_motion.set_trajectory(@(t) struct('position', [v_linear*t, 0, 0], 'orientation', [0, 0, 0]));
    
    errors_motion = zeros(num_trials, 2);
    for trial = 1:num_trials
        rng(trial * 5000 + mp_idx);
        
        snapshots = generate_signal_with_multipath(radar_params, array_motion, ...
            target_theta, target_phi, target_range, ...
            mp_theta_offset, mp_phi_offset, mp_gain, t_axis, base_snr);
        
        estimator = DoaEstimatorSynthetic(array_motion, radar_params);
        [~, peaks, ~] = estimator.estimate(snapshots, t_axis, search_grid_2d, 1, est_options);
        
        errors_motion(trial, :) = [peaks.theta(1) - target_theta, peaks.phi(1) - target_phi];
    end
    
    % ===== 静态阵列 =====
    array_static = create_circular_array(num_elements, R_array);
    
    errors_static = zeros(num_trials, 2);
    for trial = 1:num_trials
        rng(trial * 6000 + mp_idx);
        
        snapshots = generate_signal_with_multipath(radar_params, array_static, ...
            target_theta, target_phi, target_range, ...
            mp_theta_offset, mp_phi_offset, mp_gain, t_axis, base_snr);
        
        positions = array_static.get_mimo_virtual_positions(0);
        spectrum = music_standard_2d(snapshots, positions, search_grid_2d, lambda, 1);
        [~, idx] = max(spectrum(:));
        [theta_idx, phi_idx] = ind2sub(size(spectrum), idx);
        est_theta = search_grid_2d.theta(theta_idx);
        est_phi = search_grid_2d.phi(phi_idx);
        
        errors_static(trial, :) = [est_theta - target_theta, est_phi - target_phi];
    end
    
    results_multipath.rmse_motion(mp_idx, :) = sqrt(mean(errors_motion.^2, 1));
    results_multipath.rmse_static(mp_idx, :) = sqrt(mean(errors_static.^2, 1));
    
    fprintf('  多径增益=%ddB: 运动[θ=%.2f°,φ=%.2f°], 静态[θ=%.2f°,φ=%.2f°]\n', mp_gain, ...
        results_multipath.rmse_motion(mp_idx, 1), results_multipath.rmse_motion(mp_idx, 2), ...
        results_multipath.rmse_static(mp_idx, 1), results_multipath.rmse_static(mp_idx, 2));
end

%% ═══════════════════════════════════════════════════════════════════════════
%  实验4: 多目标分辨能力测试
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('实验4: 多目标分辨能力测试\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

angle_sep_list = [3, 5, 10, 15, 20];  % 目标角度间隔 (度)
fprintf('  目标角度间隔: ');
fprintf('%d° ', angle_sep_list);
fprintf('\n');

results_multitarget = struct();
results_multitarget.angle_sep = angle_sep_list;
results_multitarget.resolution_motion = zeros(length(angle_sep_list), 1);
results_multitarget.resolution_static = zeros(length(angle_sep_list), 1);

for sep_idx = 1:length(angle_sep_list)
    angle_sep = angle_sep_list(sep_idx);
    
    tar2_theta = target_theta;
    tar2_phi = target_phi + angle_sep;
    
    % ===== 运动阵列 =====
    array_motion = create_circular_array(num_elements, R_array);
    array_motion.set_trajectory(@(t) struct('position', [v_linear*t, 0, 0], 'orientation', [0, 0, 0]));
    
    resolved_motion = 0;
    for trial = 1:num_trials
        rng(trial * 7000 + sep_idx);
        
        snapshots = generate_multitarget_signal(radar_params, array_motion, ...
            target_theta, target_phi, tar2_theta, tar2_phi, target_range, t_axis, base_snr);
        
        estimator = DoaEstimatorSynthetic(array_motion, radar_params);
        est_options_mt = est_options;
        est_options_mt.num_targets = 2;
        [spectrum, ~, ~] = estimator.estimate(snapshots, t_axis, search_grid_2d, 2, est_options_mt);
        
        % 检测是否分辨出两个峰值
        num_peaks = count_peaks_2d(spectrum, search_grid_2d, 6);  % 6dB阈值
        if num_peaks >= 2
            resolved_motion = resolved_motion + 1;
        end
    end
    
    % ===== 静态阵列 =====
    array_static = create_circular_array(num_elements, R_array);
    
    resolved_static = 0;
    for trial = 1:num_trials
        rng(trial * 8000 + sep_idx);
        
        snapshots = generate_multitarget_signal(radar_params, array_static, ...
            target_theta, target_phi, tar2_theta, tar2_phi, target_range, t_axis, base_snr);
        
        positions = array_static.get_mimo_virtual_positions(0);
        spectrum = music_standard_2d(snapshots, positions, search_grid_2d, lambda, 2);
        
        num_peaks = count_peaks_2d(spectrum, search_grid_2d, 6);
        if num_peaks >= 2
            resolved_static = resolved_static + 1;
        end
    end
    
    results_multitarget.resolution_motion(sep_idx) = resolved_motion / num_trials * 100;
    results_multitarget.resolution_static(sep_idx) = resolved_static / num_trials * 100;
    
    fprintf('  角度间隔=%d°: 运动分辨率=%.0f%%, 静态分辨率=%.0f%%\n', angle_sep, ...
        results_multitarget.resolution_motion(sep_idx), results_multitarget.resolution_static(sep_idx));
end

%% ═══════════════════════════════════════════════════════════════════════════
%  绘图
%% ═══════════════════════════════════════════════════════════════════════════

set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultLineLineWidth', 1.5);

%% 图1: 杂波环境影响
figure('Position', [50, 50, 1000, 400], 'Color', 'white');

subplot(1, 2, 1);
hold on;
plot(cnr_list, results_clutter.rmse_motion(:, 2), '-o', 'LineWidth', 2, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'b', 'DisplayName', '运动阵列');
plot(cnr_list, results_clutter.rmse_static(:, 2), '-s', 'LineWidth', 2, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', '静态阵列');
xlabel('杂波噪比 CNR (dB)', 'FontWeight', 'bold');
ylabel('方位角RMSE (°)', 'FontWeight', 'bold');
title('(a) 杂波对方位角估计的影响', 'FontWeight', 'bold');
legend('Location', 'northwest');
grid on;

subplot(1, 2, 2);
hold on;
plot(cnr_list, results_clutter.rmse_motion(:, 1), '-o', 'LineWidth', 2, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'b', 'DisplayName', '运动阵列');
plot(cnr_list, results_clutter.rmse_static(:, 1), '-s', 'LineWidth', 2, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', '静态阵列');
xlabel('杂波噪比 CNR (dB)', 'FontWeight', 'bold');
ylabel('俯仰角RMSE (°)', 'FontWeight', 'bold');
title('(b) 杂波对俯仰角估计的影响', 'FontWeight', 'bold');
legend('Location', 'northwest');
grid on;

sgtitle('杂波环境对DOA估计精度的影响', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(output_folder, 'fig1_杂波环境影响.png'));
saveas(gcf, fullfile(output_folder, 'fig1_杂波环境影响.eps'), 'epsc');
fprintf('\n图片已保存: fig1_杂波环境影响.png\n');

%% 图2: 有意干扰影响
figure('Position', [50, 50, 1000, 400], 'Color', 'white');

subplot(1, 2, 1);
hold on;
plot(jsr_list, results_jamming.rmse_motion(:, 2), '-o', 'LineWidth', 2, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'b', 'DisplayName', '运动阵列');
plot(jsr_list, results_jamming.rmse_static(:, 2), '-s', 'LineWidth', 2, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', '静态阵列');
xlabel('干信比 JSR (dB)', 'FontWeight', 'bold');
ylabel('方位角RMSE (°)', 'FontWeight', 'bold');
title('(a) 有意干扰对方位角估计的影响', 'FontWeight', 'bold');
legend('Location', 'northwest');
grid on;

subplot(1, 2, 2);
hold on;
plot(jsr_list, results_jamming.rmse_motion(:, 1), '-o', 'LineWidth', 2, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'b', 'DisplayName', '运动阵列');
plot(jsr_list, results_jamming.rmse_static(:, 1), '-s', 'LineWidth', 2, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', '静态阵列');
xlabel('干信比 JSR (dB)', 'FontWeight', 'bold');
ylabel('俯仰角RMSE (°)', 'FontWeight', 'bold');
title('(b) 有意干扰对俯仰角估计的影响', 'FontWeight', 'bold');
legend('Location', 'northwest');
grid on;

sgtitle(sprintf('有意干扰对DOA估计精度的影响 (干扰源: θ=%.0f°, φ=%.0f°)', ...
    jammer_theta, jammer_phi), 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(output_folder, 'fig2_有意干扰影响.png'));
saveas(gcf, fullfile(output_folder, 'fig2_有意干扰影响.eps'), 'epsc');
fprintf('图片已保存: fig2_有意干扰影响.png\n');

%% 图3: 多径效应影响
figure('Position', [50, 50, 1000, 400], 'Color', 'white');

subplot(1, 2, 1);
hold on;
plot(multipath_gain_list, results_multipath.rmse_motion(:, 2), '-o', 'LineWidth', 2, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'b', 'DisplayName', '运动阵列');
plot(multipath_gain_list, results_multipath.rmse_static(:, 2), '-s', 'LineWidth', 2, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', '静态阵列');
xlabel('多径增益 (dB)', 'FontWeight', 'bold');
ylabel('方位角RMSE (°)', 'FontWeight', 'bold');
title('(a) 多径效应对方位角估计的影响', 'FontWeight', 'bold');
legend('Location', 'northwest');
grid on;
set(gca, 'XDir', 'reverse');

subplot(1, 2, 2);
hold on;
plot(multipath_gain_list, results_multipath.rmse_motion(:, 1), '-o', 'LineWidth', 2, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'b', 'DisplayName', '运动阵列');
plot(multipath_gain_list, results_multipath.rmse_static(:, 1), '-s', 'LineWidth', 2, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', '静态阵列');
xlabel('多径增益 (dB)', 'FontWeight', 'bold');
ylabel('俯仰角RMSE (°)', 'FontWeight', 'bold');
title('(b) 多径效应对俯仰角估计的影响', 'FontWeight', 'bold');
legend('Location', 'northwest');
grid on;
set(gca, 'XDir', 'reverse');

sgtitle(sprintf('多径效应对DOA估计精度的影响 (偏移: Δθ=%.0f°, Δφ=%.0f°)', ...
    mp_theta_offset, mp_phi_offset), 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(output_folder, 'fig3_多径效应影响.png'));
saveas(gcf, fullfile(output_folder, 'fig3_多径效应影响.eps'), 'epsc');
fprintf('图片已保存: fig3_多径效应影响.png\n');

%% 图4: 多目标分辨能力
figure('Position', [50, 50, 600, 400], 'Color', 'white');

hold on;
bar_data = [results_multitarget.resolution_static, results_multitarget.resolution_motion];
b = bar(angle_sep_list, bar_data);
b(1).FaceColor = [0.9, 0.4, 0.3];  % 红色 - 静态
b(2).FaceColor = [0.2, 0.6, 0.9];  % 蓝色 - 运动
xlabel('目标角度间隔 (°)', 'FontWeight', 'bold');
ylabel('分辨成功率 (%)', 'FontWeight', 'bold');
title('多目标分辨能力对比', 'FontSize', 14, 'FontWeight', 'bold');
legend({'静态阵列', '运动阵列'}, 'Location', 'southeast');
grid on;
ylim([0, 110]);

saveas(gcf, fullfile(output_folder, 'fig4_多目标分辨能力.png'));
saveas(gcf, fullfile(output_folder, 'fig4_多目标分辨能力.eps'), 'epsc');
fprintf('图片已保存: fig4_多目标分辨能力.png\n');

%% 图5: 综合性能雷达图
figure('Position', [50, 50, 600, 500], 'Color', 'white');

metrics_labels = {'抗杂波', '抗干扰', '抗多径', '分辨能力'};

% 归一化性能评分
max_rmse = 10;  % 最大RMSE参考值
motion_scores = [
    max(0, 1 - results_clutter.rmse_motion(3, 2) / max_rmse),
    max(0, 1 - results_jamming.rmse_motion(3, 2) / max_rmse),
    max(0, 1 - results_multipath.rmse_motion(3, 2) / max_rmse),
    results_multitarget.resolution_motion(2) / 100
];

static_scores = [
    max(0, 1 - results_clutter.rmse_static(3, 2) / max_rmse),
    max(0, 1 - results_jamming.rmse_static(3, 2) / max_rmse),
    max(0, 1 - results_multipath.rmse_static(3, 2) / max_rmse),
    results_multitarget.resolution_static(2) / 100
];

motion_scores = max(0, min(1, motion_scores));
static_scores = max(0, min(1, static_scores));

% 确保是行向量
motion_scores = motion_scores(:)';
static_scores = static_scores(:)';

theta_radar = linspace(0, 2*pi, length(metrics_labels) + 1);
motion_plot = [motion_scores, motion_scores(1)];
static_plot = [static_scores, static_scores(1)];

polarplot(theta_radar, motion_plot, '-o', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', 'b', 'DisplayName', '运动阵列');
hold on;
polarplot(theta_radar, static_plot, '-s', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', 'r', 'DisplayName', '静态阵列');

ax = gca;
ax.ThetaTick = rad2deg(theta_radar(1:end-1));
ax.ThetaTickLabel = metrics_labels;
ax.RLim = [0, 1];
legend('Location', 'southoutside', 'Orientation', 'horizontal');
title('复杂电磁环境综合性能对比', 'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, fullfile(output_folder, 'fig5_综合性能雷达图.png'));
saveas(gcf, fullfile(output_folder, 'fig5_综合性能雷达图.eps'), 'epsc');
fprintf('图片已保存: fig5_综合性能雷达图.png\n');

%% 保存结果
save(fullfile(output_folder, 'experiment_results.mat'), ...
    'results_clutter', 'results_jamming', 'results_multipath', 'results_multitarget');
fprintf('数据已保存: experiment_results.mat\n');

%% 结果总结
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                        结果总结                                   \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

fprintf('【杂波环境】CNR=20dB时\n');
fprintf('  运动阵列: RMSE_φ=%.2f°, RMSE_θ=%.2f°\n', ...
    results_clutter.rmse_motion(3, 2), results_clutter.rmse_motion(3, 1));
fprintf('  静态阵列: RMSE_φ=%.2f°, RMSE_θ=%.2f°\n', ...
    results_clutter.rmse_static(3, 2), results_clutter.rmse_static(3, 1));

fprintf('\n【有意干扰】JSR=10dB时\n');
fprintf('  运动阵列: RMSE_φ=%.2f°, RMSE_θ=%.2f°\n', ...
    results_jamming.rmse_motion(3, 2), results_jamming.rmse_motion(3, 1));
fprintf('  静态阵列: RMSE_φ=%.2f°, RMSE_θ=%.2f°\n', ...
    results_jamming.rmse_static(3, 2), results_jamming.rmse_static(3, 1));

fprintf('\n【多径效应】增益=-10dB时\n');
fprintf('  运动阵列: RMSE_φ=%.2f°, RMSE_θ=%.2f°\n', ...
    results_multipath.rmse_motion(3, 2), results_multipath.rmse_motion(3, 1));
fprintf('  静态阵列: RMSE_φ=%.2f°, RMSE_θ=%.2f°\n', ...
    results_multipath.rmse_static(3, 2), results_multipath.rmse_static(3, 1));

fprintf('\n【多目标分辨】5°间隔时\n');
fprintf('  运动阵列分辨率: %.0f%%\n', results_multitarget.resolution_motion(2));
fprintf('  静态阵列分辨率: %.0f%%\n', results_multitarget.resolution_static(2));

fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('实验完成！\n');
fprintf('所有结果保存在: %s\n', output_folder);
fprintf('═══════════════════════════════════════════════════════════════════\n');

diary off;

%% ═══════════════════════════════════════════════════════════════════════════
%  辅助函数：信号生成
%% ═══════════════════════════════════════════════════════════════════════════

function snapshots = generate_signal_with_clutter(radar_params, array, ...
    target_theta, target_phi, target_range, t_axis, snr_db, cnr_db)
    % 生成带杂波的信号
    
    lambda = radar_params.lambda;
    num_snapshots = length(t_axis);
    num_virtual = array.get_num_virtual_elements();
    
    % 目标方向
    target_dir = [cosd(target_phi)*sind(target_theta), ...
                  sind(target_phi)*sind(target_theta), ...
                  cosd(target_theta)];
    
    % 目标信号
    target_amp = (randn() + 1j*randn()) / sqrt(2);
    snapshots = zeros(num_virtual, num_snapshots);
    
    for k = 1:num_snapshots
        t = t_axis(k);
        positions = array.get_mimo_virtual_positions(t);
        
        for v = 1:num_virtual
            phase = 4*pi/lambda * dot(positions(v,:), target_dir);
            snapshots(v, k) = target_amp * exp(-1j * phase);
        end
    end
    
    % 杂波信号（多个随机方向散射体）
    if cnr_db > -20
        clutter_power = 10^(cnr_db/10);
        num_clutter = 20;
        
        clutter_signal = zeros(num_virtual, num_snapshots);
        for c = 1:num_clutter
            % 随机方向（主要在地面/水面方向）
            c_theta = 60 + 30*rand();  % 60-90°
            c_phi = 360*rand();
            c_dir = [cosd(c_phi)*sind(c_theta), sind(c_phi)*sind(c_theta), cosd(c_theta)];
            c_amp = sqrt(clutter_power/num_clutter) * (randn() + 1j*randn()) / sqrt(2);
            
            for k = 1:num_snapshots
                t = t_axis(k);
                positions = array.get_mimo_virtual_positions(t);
                for v = 1:num_virtual
                    phase = 4*pi/lambda * dot(positions(v,:), c_dir);
                    clutter_signal(v, k) = clutter_signal(v, k) + c_amp * exp(-1j * phase);
                end
            end
        end
        snapshots = snapshots + clutter_signal;
    end
    
    % 添加噪声
    signal_power = mean(abs(snapshots(:)).^2);
    noise_power = signal_power / (10^(snr_db/10));
    noise = sqrt(noise_power/2) * (randn(size(snapshots)) + 1j*randn(size(snapshots)));
    snapshots = snapshots + noise;
end

function snapshots = generate_signal_with_jamming(radar_params, array, ...
    target_theta, target_phi, target_range, ...
    jam_theta, jam_phi, t_axis, snr_db, jsr_db)
    % 生成带有意干扰的信号
    
    lambda = radar_params.lambda;
    num_snapshots = length(t_axis);
    num_virtual = array.get_num_virtual_elements();
    
    % 目标方向
    target_dir = [cosd(target_phi)*sind(target_theta), ...
                  sind(target_phi)*sind(target_theta), ...
                  cosd(target_theta)];
    
    % 干扰方向
    jam_dir = [cosd(jam_phi)*sind(jam_theta), ...
               sind(jam_phi)*sind(jam_theta), ...
               cosd(jam_theta)];
    
    % 目标信号
    target_amp = (randn() + 1j*randn()) / sqrt(2);
    snapshots = zeros(num_virtual, num_snapshots);
    
    for k = 1:num_snapshots
        t = t_axis(k);
        positions = array.get_mimo_virtual_positions(t);
        
        for v = 1:num_virtual
            phase = 4*pi/lambda * dot(positions(v,:), target_dir);
            snapshots(v, k) = target_amp * exp(-1j * phase);
        end
    end
    
    % 干扰信号（宽带噪声干扰）
    if jsr_db > -30
        jam_power = 10^(jsr_db/10);
        
        for k = 1:num_snapshots
            t = t_axis(k);
            positions = array.get_mimo_virtual_positions(t);
            jam_noise = sqrt(jam_power/2) * (randn(1) + 1j*randn(1));
            
            for v = 1:num_virtual
                phase = 4*pi/lambda * dot(positions(v,:), jam_dir);
                snapshots(v, k) = snapshots(v, k) + jam_noise * exp(-1j * phase);
            end
        end
    end
    
    % 添加噪声
    signal_power = 1;  % 参考目标功率
    noise_power = signal_power / (10^(snr_db/10));
    noise = sqrt(noise_power/2) * (randn(size(snapshots)) + 1j*randn(size(snapshots)));
    snapshots = snapshots + noise;
end

function snapshots = generate_signal_with_multipath(radar_params, array, ...
    target_theta, target_phi, target_range, ...
    mp_theta_offset, mp_phi_offset, mp_gain_db, t_axis, snr_db)
    % 生成带多径效应的信号
    
    lambda = radar_params.lambda;
    num_snapshots = length(t_axis);
    num_virtual = array.get_num_virtual_elements();
    
    % 直达路径方向
    target_dir = [cosd(target_phi)*sind(target_theta), ...
                  sind(target_phi)*sind(target_theta), ...
                  cosd(target_theta)];
    
    % 多径方向
    mp_theta = target_theta + mp_theta_offset;
    mp_phi = target_phi + mp_phi_offset;
    mp_dir = [cosd(mp_phi)*sind(mp_theta), ...
              sind(mp_phi)*sind(mp_theta), ...
              cosd(mp_theta)];
    
    mp_amp = 10^(mp_gain_db/20);
    mp_phase = exp(1j * 2*pi*rand());  % 随机相位
    
    % 信号
    target_amp = (randn() + 1j*randn()) / sqrt(2);
    snapshots = zeros(num_virtual, num_snapshots);
    
    for k = 1:num_snapshots
        t = t_axis(k);
        positions = array.get_mimo_virtual_positions(t);
        
        for v = 1:num_virtual
            % 直达信号
            phase_direct = 4*pi/lambda * dot(positions(v,:), target_dir);
            direct_sig = target_amp * exp(-1j * phase_direct);
            
            % 多径信号
            phase_mp = 4*pi/lambda * dot(positions(v,:), mp_dir);
            mp_sig = target_amp * mp_amp * mp_phase * exp(-1j * phase_mp);
            
            snapshots(v, k) = direct_sig + mp_sig;
        end
    end
    
    % 添加噪声
    signal_power = mean(abs(snapshots(:)).^2);
    noise_power = signal_power / (10^(snr_db/10));
    noise = sqrt(noise_power/2) * (randn(size(snapshots)) + 1j*randn(size(snapshots)));
    snapshots = snapshots + noise;
end

function snapshots = generate_multitarget_signal(radar_params, array, ...
    tar1_theta, tar1_phi, tar2_theta, tar2_phi, target_range, t_axis, snr_db)
    % 生成多目标信号
    
    lambda = radar_params.lambda;
    num_snapshots = length(t_axis);
    num_virtual = array.get_num_virtual_elements();
    
    % 目标方向
    dir1 = [cosd(tar1_phi)*sind(tar1_theta), sind(tar1_phi)*sind(tar1_theta), cosd(tar1_theta)];
    dir2 = [cosd(tar2_phi)*sind(tar2_theta), sind(tar2_phi)*sind(tar2_theta), cosd(tar2_theta)];
    
    % 信号
    amp1 = (randn() + 1j*randn()) / sqrt(2);
    amp2 = (randn() + 1j*randn()) / sqrt(2);
    snapshots = zeros(num_virtual, num_snapshots);
    
    for k = 1:num_snapshots
        t = t_axis(k);
        positions = array.get_mimo_virtual_positions(t);
        
        for v = 1:num_virtual
            phase1 = 4*pi/lambda * dot(positions(v,:), dir1);
            phase2 = 4*pi/lambda * dot(positions(v,:), dir2);
            snapshots(v, k) = amp1 * exp(-1j*phase1) + amp2 * exp(-1j*phase2);
        end
    end
    
    % 添加噪声
    signal_power = mean(abs(snapshots(:)).^2);
    noise_power = signal_power / (10^(snr_db/10));
    noise = sqrt(noise_power/2) * (randn(size(snapshots)) + 1j*randn(size(snapshots)));
    snapshots = snapshots + noise;
end

%% ═══════════════════════════════════════════════════════════════════════════
%  辅助函数：MUSIC和峰值检测
%% ═══════════════════════════════════════════════════════════════════════════

function spectrum = music_standard_2d(snapshots, positions, search_grid, lambda, num_signals)
    % 标准2D MUSIC算法
    
    theta_range = search_grid.theta;
    phi_range = search_grid.phi;
    num_theta = length(theta_range);
    num_phi = length(phi_range);
    
    % 协方差矩阵
    R = snapshots * snapshots' / size(snapshots, 2);
    
    % 特征分解
    [V, D] = eig(R);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    
    % 噪声子空间
    Vn = V(:, num_signals+1:end);
    
    % 导向矢量和谱计算
    spectrum = zeros(num_theta, num_phi);
    num_elements = size(positions, 1);
    
    for ti = 1:num_theta
        for pi = 1:num_phi
            theta = theta_range(ti);
            phi = phi_range(pi);
            
            % 方向矢量
            u = [cosd(phi)*sind(theta), sind(phi)*sind(theta), cosd(theta)];
            
            % 导向矢量
            a = zeros(num_elements, 1);
            for e = 1:num_elements
                phase = 4*pi/lambda * dot(positions(e,:), u);
                a(e) = exp(-1j * phase);
            end
            
            % MUSIC谱
            spectrum(ti, pi) = (a' * a) / (a' * (Vn * Vn') * a);
        end
    end
    
    spectrum = abs(spectrum);
end

function num_peaks = count_peaks_2d(spectrum, search_grid, threshold_db)
    % 计算2D谱中的峰值数量
    
    spectrum_db = 10*log10(abs(spectrum) + eps);
    max_val = max(spectrum_db(:));
    threshold = max_val - threshold_db;
    
    % 简单阈值检测
    binary_map = spectrum_db > threshold;
    
    % 连通域分析
    CC = bwconncomp(binary_map);
    num_peaks = CC.NumObjects;
end
