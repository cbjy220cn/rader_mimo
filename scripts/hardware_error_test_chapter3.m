%% ═══════════════════════════════════════════════════════════════════════════
%  第三章补充实验：硬件误差鲁棒性测试
%  - 基于老代码架构（先验轨迹+RDC扩展）
%  - 测试飞行定位误差对DOA估计的影响
%  - 对比有/无运动时的鲁棒性
%% ═══════════════════════════════════════════════════════════════════════════
clear; clc; close all;

% 创建带时间戳的输出文件夹
script_name = 'hardware_error_test_chapter3';
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
output_folder = fullfile('validation_results', [script_name '_' timestamp]);
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

% 初始化日志
log_file = fullfile(output_folder, 'experiment_log.txt');
diary(log_file);

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║  第三章补充实验：飞行定位误差鲁棒性测试                      ║\n');
fprintf('║  基于先验轨迹的孔径扩展方法（UCA旋转）                       ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');
fprintf('输出目录: %s\n\n', output_folder);

%% ═══════════════════════════════════════════════════════════════════════════
%  雷达系统参数（与老代码保持一致）
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('【雷达系统参数】\n');

c = physconst('LightSpeed');
BW = 50e6;                          % 带宽: 50 MHz
f0 = 3e9;                           % 载频: 3 GHz
lambda = c / f0;                    % 波长
numADC = 361;                       % ADC采样点数
numChirps = 256;                    % 每帧Chirp数
numCPI = 10;                        % CPI数
T = 10e-3;                          % PRI: 10 ms
PRF = 1/T;
F = numADC/T;                       % 采样频率
dt = 1/F;                           % 采样间隔
slope = BW/T;

fprintf('  载频: %.2f GHz (λ = %.2f cm)\n', f0/1e9, lambda*100);
fprintf('  带宽: %.0f MHz, PRI: %.0f ms\n', BW/1e6, T*1000);
fprintf('  Chirp数: %d × %d CPI = %d\n', numChirps, numCPI, numChirps*numCPI);

%% ═══════════════════════════════════════════════════════════════════════════
%  阵列参数
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n【阵列参数】\n');

numTX = 1;
numRX = 8;                          % 8元UCA
R_tx = 0;                           % 发射阵列半径（单发）
R_rx = 0.05;                        % 接收阵列半径: 5 cm

% 孔径扩展参数
N_L = 1;                            % 扩展因子（可调整测试）
N_num = 2*(N_L-1)+1;                % 扩展后的等效阵元数

fprintf('  接收阵列: %d元UCA, 半径 %.2f cm\n', numRX, R_rx*100);
fprintf('  孔径扩展因子 N_L: %d (等效 %d 个位置)\n', N_L, N_num);

%% ═══════════════════════════════════════════════════════════════════════════
%  目标参数
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n【目标参数】\n');

tar1_theta = 30;                    % 目标俯仰角
tar1_phi = 60;                      % 目标方位角
r1_radial = 660;                    % 目标距离

fprintf('  目标: θ=%.0f°, φ=%.0f°, R=%.0fm\n', tar1_theta, tar1_phi, r1_radial);

%% ═══════════════════════════════════════════════════════════════════════════
%  误差参数设置
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n【误差参数设置】\n');

% ===== 定位误差 =====
% 位置误差标准差 (单位: 米)
% 典型GPS误差: 1-5 cm (RTK), 1-10 m (普通GPS)
% 典型IMU积分误差: 随时间增长
position_error_std_list = [0, 0.005, 0.01, 0.02, 0.05];  % 0, 5mm, 1cm, 2cm, 5cm
fprintf('  位置误差标准差: ');
fprintf('%.0fmm ', position_error_std_list * 1000);
fprintf('\n');

% 姿态角误差标准差 (单位: 度)
% 典型IMU误差: 0.1-1°
attitude_error_std_list = [0, 0.1, 0.5, 1.0, 2.0];  % 0, 0.1°, 0.5°, 1°, 2°
fprintf('  姿态角误差标准差: ');
fprintf('%.1f° ', attitude_error_std_list);
fprintf('\n');

% SNR范围
snr_list = [10, 20, 30, 50];
fprintf('  SNR测试点: ');
fprintf('%ddB ', snr_list);
fprintf('\n');

% 蒙特卡洛次数
num_trials = 30;
fprintf('  蒙特卡洛次数: %d\n', num_trials);

%% ═══════════════════════════════════════════════════════════════════════════
%  实验1: 位置误差对运动阵列的影响
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('实验1: 位置误差对运动阵列DOA估计的影响\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

results_pos_error = struct();
results_pos_error.position_error_std = position_error_std_list;
results_pos_error.snr = snr_list;
results_pos_error.rmse_theta = zeros(length(position_error_std_list), length(snr_list));
results_pos_error.rmse_phi = zeros(length(position_error_std_list), length(snr_list));
results_pos_error.rmse_total = zeros(length(position_error_std_list), length(snr_list));

for pos_idx = 1:length(position_error_std_list)
    pos_error_std = position_error_std_list(pos_idx);
    
    for snr_idx = 1:length(snr_list)
        snr = snr_list(snr_idx);
        
        errors_theta = zeros(1, num_trials);
        errors_phi = zeros(1, num_trials);
        
        for trial = 1:num_trials
            rng(trial * 1000 + pos_idx * 100 + snr_idx);
            
            % 调用带误差的雷达处理函数
            [theta_est, phi_est] = radar_processing_with_error(...
                tar1_theta, tar1_phi, r1_radial, ...
                numRX, R_rx, N_L, snr, ...
                pos_error_std, 0, ...  % 位置误差, 姿态误差=0
                numADC, numChirps, numCPI, T, f0, BW, c, lambda, slope, dt);
            
            errors_theta(trial) = theta_est - tar1_theta;
            errors_phi(trial) = phi_est - tar1_phi;
        end
        
        results_pos_error.rmse_theta(pos_idx, snr_idx) = sqrt(mean(errors_theta.^2));
        results_pos_error.rmse_phi(pos_idx, snr_idx) = sqrt(mean(errors_phi.^2));
        results_pos_error.rmse_total(pos_idx, snr_idx) = sqrt(mean(errors_theta.^2 + errors_phi.^2));
        
        fprintf('  位置误差=%.0fmm, SNR=%ddB: RMSE_θ=%.2f°, RMSE_φ=%.2f°\n', ...
            pos_error_std*1000, snr, ...
            results_pos_error.rmse_theta(pos_idx, snr_idx), ...
            results_pos_error.rmse_phi(pos_idx, snr_idx));
    end
end

%% ═══════════════════════════════════════════════════════════════════════════
%  实验2: 姿态角误差对运动阵列的影响
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('实验2: 姿态角误差对运动阵列DOA估计的影响\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

results_att_error = struct();
results_att_error.attitude_error_std = attitude_error_std_list;
results_att_error.snr = snr_list;
results_att_error.rmse_theta = zeros(length(attitude_error_std_list), length(snr_list));
results_att_error.rmse_phi = zeros(length(attitude_error_std_list), length(snr_list));
results_att_error.rmse_total = zeros(length(attitude_error_std_list), length(snr_list));

for att_idx = 1:length(attitude_error_std_list)
    att_error_std = attitude_error_std_list(att_idx);
    
    for snr_idx = 1:length(snr_list)
        snr = snr_list(snr_idx);
        
        errors_theta = zeros(1, num_trials);
        errors_phi = zeros(1, num_trials);
        
        for trial = 1:num_trials
            rng(trial * 2000 + att_idx * 100 + snr_idx);
            
            % 调用带误差的雷达处理函数
            [theta_est, phi_est] = radar_processing_with_error(...
                tar1_theta, tar1_phi, r1_radial, ...
                numRX, R_rx, N_L, snr, ...
                0, att_error_std, ...  % 位置误差=0, 姿态误差
                numADC, numChirps, numCPI, T, f0, BW, c, lambda, slope, dt);
            
            errors_theta(trial) = theta_est - tar1_theta;
            errors_phi(trial) = phi_est - tar1_phi;
        end
        
        results_att_error.rmse_theta(att_idx, snr_idx) = sqrt(mean(errors_theta.^2));
        results_att_error.rmse_phi(att_idx, snr_idx) = sqrt(mean(errors_phi.^2));
        results_att_error.rmse_total(att_idx, snr_idx) = sqrt(mean(errors_theta.^2 + errors_phi.^2));
        
        fprintf('  姿态误差=%.1f°, SNR=%ddB: RMSE_θ=%.2f°, RMSE_φ=%.2f°\n', ...
            att_error_std, snr, ...
            results_att_error.rmse_theta(att_idx, snr_idx), ...
            results_att_error.rmse_phi(att_idx, snr_idx));
    end
end

%% ═══════════════════════════════════════════════════════════════════════════
%  实验3: 静态 vs 运动阵列在误差条件下的对比
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('实验3: 静态 vs 运动阵列在定位误差条件下的鲁棒性对比\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% 选择中等误差进行对比
test_pos_error = 0.01;  % 1cm位置误差
test_att_error = 0.5;   % 0.5°姿态误差
test_snr = 20;          % 20dB SNR

fprintf('测试条件: 位置误差=%.0fmm, 姿态误差=%.1f°, SNR=%ddB\n\n', ...
    test_pos_error*1000, test_att_error, test_snr);

% 运动阵列结果
errors_motion_theta = zeros(1, num_trials);
errors_motion_phi = zeros(1, num_trials);

for trial = 1:num_trials
    rng(trial * 3000);
    [theta_est, phi_est] = radar_processing_with_error(...
        tar1_theta, tar1_phi, r1_radial, ...
        numRX, R_rx, N_L, test_snr, ...
        test_pos_error, test_att_error, ...
        numADC, numChirps, numCPI, T, f0, BW, c, lambda, slope, dt, ...
        true);  % 运动模式
    errors_motion_theta(trial) = theta_est - tar1_theta;
    errors_motion_phi(trial) = phi_est - tar1_phi;
end

rmse_motion_theta = sqrt(mean(errors_motion_theta.^2));
rmse_motion_phi = sqrt(mean(errors_motion_phi.^2));

% 静态阵列结果
errors_static_theta = zeros(1, num_trials);
errors_static_phi = zeros(1, num_trials);

for trial = 1:num_trials
    rng(trial * 4000);
    [theta_est, phi_est] = radar_processing_with_error(...
        tar1_theta, tar1_phi, r1_radial, ...
        numRX, R_rx, N_L, test_snr, ...
        test_pos_error, test_att_error, ...
        numADC, numChirps, numCPI, T, f0, BW, c, lambda, slope, dt, ...
        false);  % 静态模式
    errors_static_theta(trial) = theta_est - tar1_theta;
    errors_static_phi(trial) = phi_est - tar1_phi;
end

rmse_static_theta = sqrt(mean(errors_static_theta.^2));
rmse_static_phi = sqrt(mean(errors_static_phi.^2));

fprintf('【结果对比】\n');
fprintf('  运动阵列: RMSE_θ=%.2f°, RMSE_φ=%.2f°\n', rmse_motion_theta, rmse_motion_phi);
fprintf('  静态阵列: RMSE_θ=%.2f°, RMSE_φ=%.2f°\n', rmse_static_theta, rmse_static_phi);

results_compare = struct();
results_compare.motion_rmse_theta = rmse_motion_theta;
results_compare.motion_rmse_phi = rmse_motion_phi;
results_compare.static_rmse_theta = rmse_static_theta;
results_compare.static_rmse_phi = rmse_static_phi;
results_compare.test_condition.pos_error = test_pos_error;
results_compare.test_condition.att_error = test_att_error;
results_compare.test_condition.snr = test_snr;

%% ═══════════════════════════════════════════════════════════════════════════
%  绘图
%% ═══════════════════════════════════════════════════════════════════════════

% 设置论文风格
set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultLineLineWidth', 1.5);

%% 图1: 位置误差影响
figure('Position', [50, 50, 1000, 400], 'Color', 'white');

subplot(1, 2, 1);
hold on;
colors = lines(length(snr_list));
markers = {'o', 's', 'd', '^'};
for snr_idx = 1:length(snr_list)
    plot(position_error_std_list * 1000, results_pos_error.rmse_phi(:, snr_idx), ...
        ['-' markers{snr_idx}], 'Color', colors(snr_idx,:), ...
        'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colors(snr_idx,:), ...
        'DisplayName', sprintf('SNR=%ddB', snr_list(snr_idx)));
end
xlabel('位置误差标准差 (mm)', 'FontWeight', 'bold');
ylabel('方位角RMSE (°)', 'FontWeight', 'bold');
title('(a) 位置误差对方位角估计的影响', 'FontWeight', 'bold');
legend('Location', 'northwest');
grid on;
xlim([0, max(position_error_std_list)*1000*1.1]);

subplot(1, 2, 2);
hold on;
for snr_idx = 1:length(snr_list)
    plot(position_error_std_list * 1000, results_pos_error.rmse_theta(:, snr_idx), ...
        ['-' markers{snr_idx}], 'Color', colors(snr_idx,:), ...
        'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colors(snr_idx,:), ...
        'DisplayName', sprintf('SNR=%ddB', snr_list(snr_idx)));
end
xlabel('位置误差标准差 (mm)', 'FontWeight', 'bold');
ylabel('俯仰角RMSE (°)', 'FontWeight', 'bold');
title('(b) 位置误差对俯仰角估计的影响', 'FontWeight', 'bold');
legend('Location', 'northwest');
grid on;
xlim([0, max(position_error_std_list)*1000*1.1]);

sgtitle('位置定位误差对DOA估计精度的影响', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(output_folder, 'fig1_位置误差影响.png'));
saveas(gcf, fullfile(output_folder, 'fig1_位置误差影响.eps'), 'epsc');
fprintf('\n图片已保存: fig1_位置误差影响.png\n');

%% 图2: 姿态角误差影响
figure('Position', [50, 50, 1000, 400], 'Color', 'white');

subplot(1, 2, 1);
hold on;
for snr_idx = 1:length(snr_list)
    plot(attitude_error_std_list, results_att_error.rmse_phi(:, snr_idx), ...
        ['-' markers{snr_idx}], 'Color', colors(snr_idx,:), ...
        'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colors(snr_idx,:), ...
        'DisplayName', sprintf('SNR=%ddB', snr_list(snr_idx)));
end
xlabel('姿态角误差标准差 (°)', 'FontWeight', 'bold');
ylabel('方位角RMSE (°)', 'FontWeight', 'bold');
title('(a) 姿态误差对方位角估计的影响', 'FontWeight', 'bold');
legend('Location', 'northwest');
grid on;
xlim([0, max(attitude_error_std_list)*1.1]);

subplot(1, 2, 2);
hold on;
for snr_idx = 1:length(snr_list)
    plot(attitude_error_std_list, results_att_error.rmse_theta(:, snr_idx), ...
        ['-' markers{snr_idx}], 'Color', colors(snr_idx,:), ...
        'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colors(snr_idx,:), ...
        'DisplayName', sprintf('SNR=%ddB', snr_list(snr_idx)));
end
xlabel('姿态角误差标准差 (°)', 'FontWeight', 'bold');
ylabel('俯仰角RMSE (°)', 'FontWeight', 'bold');
title('(b) 姿态误差对俯仰角估计的影响', 'FontWeight', 'bold');
legend('Location', 'northwest');
grid on;
xlim([0, max(attitude_error_std_list)*1.1]);

sgtitle('姿态测量误差对DOA估计精度的影响', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(output_folder, 'fig2_姿态误差影响.png'));
saveas(gcf, fullfile(output_folder, 'fig2_姿态误差影响.eps'), 'epsc');
fprintf('图片已保存: fig2_姿态误差影响.png\n');

%% 图3: 静态vs运动阵列对比（柱状图）
figure('Position', [50, 50, 600, 400], 'Color', 'white');

comparison_data = [rmse_static_phi, rmse_static_theta; rmse_motion_phi, rmse_motion_theta];
b = bar(comparison_data);
b(1).FaceColor = [0.2, 0.6, 0.9];  % 蓝色 - 方位角
b(2).FaceColor = [0.9, 0.4, 0.3];  % 红色 - 俯仰角

set(gca, 'XTickLabel', {'静态阵列', '运动阵列'});
ylabel('RMSE (°)', 'FontWeight', 'bold');
title(sprintf('定位误差条件下的鲁棒性对比\n(位置误差=%.0fmm, 姿态误差=%.1f°, SNR=%ddB)', ...
    test_pos_error*1000, test_att_error, test_snr), 'FontWeight', 'bold');
legend({'方位角', '俯仰角'}, 'Location', 'northeast');
grid on;

% 添加数值标注
for i = 1:2
    for j = 1:2
        text(i, comparison_data(i,j) + 0.1, sprintf('%.2f°', comparison_data(i,j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    end
end

saveas(gcf, fullfile(output_folder, 'fig3_静态vs运动对比.png'));
saveas(gcf, fullfile(output_folder, 'fig3_静态vs运动对比.eps'), 'epsc');
fprintf('图片已保存: fig3_静态vs运动对比.png\n');

%% 保存结果
save(fullfile(output_folder, 'experiment_results.mat'), ...
    'results_pos_error', 'results_att_error', 'results_compare');
fprintf('数据已保存: experiment_results.mat\n');

%% 结果总结
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                        结果总结                                   \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

fprintf('【位置误差影响】\n');
fprintf('  - 位置误差从0增加到5cm时:\n');
fprintf('    方位角RMSE增加: %.2f° → %.2f° (SNR=20dB)\n', ...
    results_pos_error.rmse_phi(1, 2), results_pos_error.rmse_phi(end, 2));
fprintf('    俯仰角RMSE增加: %.2f° → %.2f° (SNR=20dB)\n', ...
    results_pos_error.rmse_theta(1, 2), results_pos_error.rmse_theta(end, 2));

fprintf('\n【姿态误差影响】\n');
fprintf('  - 姿态误差从0增加到2°时:\n');
fprintf('    方位角RMSE增加: %.2f° → %.2f° (SNR=20dB)\n', ...
    results_att_error.rmse_phi(1, 2), results_att_error.rmse_phi(end, 2));
fprintf('    俯仰角RMSE增加: %.2f° → %.2f° (SNR=20dB)\n', ...
    results_att_error.rmse_theta(1, 2), results_att_error.rmse_theta(end, 2));

fprintf('\n【关键发现】\n');
fprintf('  1. 位置误差对方位角估计影响较大\n');
fprintf('  2. 姿态误差直接影响旋转扩展的准确性\n');
fprintf('  3. 高SNR可以部分抵消定位误差的影响\n');

fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('实验完成！\n');
fprintf('所有结果保存在: %s\n', output_folder);
fprintf('═══════════════════════════════════════════════════════════════════\n');

diary off;

%% ═══════════════════════════════════════════════════════════════════════════
%  核心处理函数：带误差的雷达信号处理
%% ═══════════════════════════════════════════════════════════════════════════

function [theta_est, phi_est] = radar_processing_with_error(...
    tar_theta, tar_phi, r_radial, ...
    numRX, R_rx, N_L, snr, ...
    pos_error_std, att_error_std, ...
    numADC, numChirps, numCPI, T, f0, BW, c, lambda, slope, dt, ...
    use_motion)
    % 带定位误差的雷达信号处理函数
    % 基于老代码架构，但添加了误差模型
    %
    % 输入:
    %   tar_theta, tar_phi, r_radial - 目标参数
    %   numRX, R_rx, N_L - 阵列参数
    %   snr - 信噪比
    %   pos_error_std - 位置误差标准差 (m)
    %   att_error_std - 姿态角误差标准差 (度)
    %   use_motion - 是否使用运动模式
    
    if nargin < 21
        use_motion = true;
    end
    
    numTX = 1;
    N_num = 2*(N_L-1)+1;
    
    % 时间轴
    N = numChirps * numADC * numCPI;
    t = linspace(0, T*numChirps*numCPI, N);
    t_onePulse = 0:dt:dt*numADC-dt;
    
    % 目标位置
    r1_x = cosd(tar_phi)*sind(tar_theta)*r_radial;
    r1_y = sind(tar_phi)*sind(tar_theta)*r_radial;
    r1_z = cosd(tar_theta)*r_radial;
    r1 = [r1_x, r1_y, r1_z];
    
    tar1_loc = zeros(length(t), 3);
    tar1_loc(:,1) = r1(1);
    tar1_loc(:,2) = r1(2);
    tar1_loc(:,3) = r1(3);
    
    % 阵列旋转角速度
    if use_motion
        tr_vel = 2*pi/N_num/numRX/T;
    else
        tr_vel = 0;
    end
    
    % ===== 添加定位误差 =====
    % 位置误差: 影响阵列中心位置
    pos_error = pos_error_std * randn(1, 3);
    
    % 姿态误差: 影响旋转角度（随时间缓慢漂移）
    % 使用AR(1)模型模拟姿态漂移
    att_error = zeros(length(t), 1);
    if att_error_std > 0
        att_error(1) = att_error_std * randn();
        for k = 2:length(t)
            att_error(k) = 0.99 * att_error(k-1) + 0.1 * att_error_std * randn();
        end
    end
    att_error_rad = deg2rad(att_error);
    
    % 发射天线位置（单发，位于中心）
    tx_loc_t = cell(numTX, 1);
    for i = 1:numTX
        tx_loc_t{i} = zeros(length(t), 3);
        tx_loc_t{i}(:,1) = 0 + pos_error(1);
        tx_loc_t{i}(:,2) = 0 + pos_error(2);
        tx_loc_t{i}(:,3) = 0 + pos_error(3);
    end
    
    % 接收天线位置（UCA，带误差）
    theta_rx = linspace(0, 2*pi, numRX+1); theta_rx(end) = [];
    rx_loc_t = cell(numRX, 1);
    for i = 1:numRX
        rx_loc_t{i} = zeros(length(t), 3);
        % 理想旋转角度 + 姿态误差
        actual_angle = theta_rx(i) + tr_vel*t' + att_error_rad;
        rx_loc_t{i}(:,1) = R_rx*cos(actual_angle) + pos_error(1);
        rx_loc_t{i}(:,2) = R_rx*sin(actual_angle) + pos_error(2);
        rx_loc_t{i}(:,3) = 0 + pos_error(3);
    end
    
    % 计算时延
    delays_tar1 = cell(numTX, numRX);
    for i = 1:numTX
        for j = 1:numRX
            delays_tar1{i,j} = (vecnorm(tar1_loc-rx_loc_t{j}, 2, 2) + ...
                vecnorm(tar1_loc-tx_loc_t{i}, 2, 2)) / c;
        end
    end
    
    % 生成信号
    phase_func = @(tx, fx) 2*pi*(fx.*tx + slope/2*tx.^2);
    mixed = cell(numTX, numRX);
    
    for i = 1:numTX
        for j = 1:numRX
            signal_1 = zeros(1, numChirps*numCPI*numADC);
            for k = 1:numChirps*numCPI
                phase_t = phase_func(t_onePulse, f0);
                phase_1 = phase_func(t_onePulse - delays_tar1{i,j}(k*numADC), f0);
                signal_1((k-1)*numADC+1:k*numADC) = exp(1j*(phase_t - phase_1));
            end
            mixed{i,j} = awgn(signal_1, snr, 'measured');
        end
    end
    
    % 构建RDC
    RDC = reshape(cat(3, mixed{:}), numADC, numChirps*numCPI, numRX*numTX);
    
    % 孔径扩展
    RDC_plus = zeros(numADC, (numChirps-2*(N_L-1))*numCPI, numRX*numTX*(2*(N_L-1)+1));
    for i = 1:numCPI
        for j = 1:numChirps-2*(N_L-1)
            for k = 1:numRX*numTX
                RDC_plus(:,j+(i-1)*(numChirps-2*(N_L-1)),N_L+(k-1)*(2*(N_L-1)+1)) = ...
                    RDC(:,j+(N_L-1)+(i-1)*(numChirps-2*(N_L-1)),k);
                for l = 1:N_L-1
                    RDC_plus(:,j+(i-1)*(numChirps-2*(N_L-1)),N_L+l+(k-1)*(2*(N_L-1)+1)) = ...
                        RDC(:,j+(N_L-1)+l+(i-1)*(numChirps-2*(N_L-1)),k);
                    RDC_plus(:,j+(i-1)*(numChirps-2*(N_L-1)),N_L-l+(k-1)*(2*(N_L-1)+1)) = ...
                        RDC(:,j+(N_L-1)-l+(i-1)*(numChirps-2*(N_L-1)),k);
                end
            end
        end
    end
    
    % 旋转修正
    numChirps_new = numChirps - 2*(N_L-1);
    RDC_plus_rot = zeros(numADC, numChirps_new*numCPI, numRX*numTX*(2*(N_L-1)+1));
    if tr_vel == 0
        RDC_plus_rot = RDC_plus;
        delta_phi = 0;
    else
        for i = 1:numChirps_new
            RDC_plus_rot(:,i,:) = circshift(RDC_plus(:,i,:), (i-1), 3);
        end
        delta_phi = 360/N_num/numRX;
    end
    
    % 导向矢量计算（使用理想阵列位置）
    theta_circle = linspace(0, 2*pi, numRX*(2*(N_L-1)+1)+1); theta_circle(end) = [];
    circle_loc = cell(numRX*(2*(N_L-1)+1), 1);
    for i = 1:numRX*(2*(N_L-1)+1)
        circle_loc{i} = [R_rx*cos(theta_circle(i)), R_rx*sin(theta_circle(i)), 0];
    end
    
    x_coords = cellfun(@(c) c(1), circle_loc);
    y_coords = cellfun(@(c) c(2), circle_loc);
    X = x_coords(:);
    Y = y_coords(:);
    
    % 搜索范围
    ang_phi = -180:0.5:180;
    ang_theta = 0:0.5:90;
    
    num_theta = length(ang_theta);
    num_phi = length(ang_phi);
    [theta_grid, phi_grid] = ndgrid(ang_theta, ang_phi);
    theta_list = theta_grid(:);
    phi_list = phi_grid(:);
    
    A_sind = sind(phi_list) .* sind(theta_list);
    A_cosd = cosd(phi_list) .* sind(theta_list);
    
    coords = [Y, X];
    A_matrix = [A_sind, A_cosd];
    phase_sv = (2 * pi / lambda) * (coords * A_matrix.');
    a1 = exp(-1i * phase_sv);
    
    % MUSIC算法
    RDC_plus_rot_frame = RDC_plus_rot(:, 1:numChirps_new, :);
    A = reshape(squeeze(RDC_plus_rot_frame(:,1:10,:)), numADC*10, numRX*numTX*(2*(N_L-1)+1));
    Rxx = (A'*A);
    
    [Q, D] = eig(Rxx);
    [~, I] = sort(diag(D), 'descend');
    Q = Q(:, I);
    Qn = Q(:, 2:end);
    
    QnQn = Qn * Qn';
    A_all = a1;
    numerator = sum(conj(A_all) .* A_all, 1);
    QnQn_A = QnQn * A_all;
    denominator = sum(conj(A_all) .* QnQn_A, 1);
    music_spectrum_2D = reshape(numerator ./ denominator, num_theta, num_phi);
    
    % 方位角修正
    ang_phi_shifted = mod(ang_phi + delta_phi + 180 + 180, 360) - 180;
    [ang_phi_sorted, sort_idx] = sort(ang_phi_shifted);
    spectrum_shifted = squeeze(music_spectrum_2D);
    spectrum_shifted = spectrum_shifted(:, sort_idx);
    
    % 找峰值
    spectrum_data = 10 * log10(abs(spectrum_shifted));
    [max_val, max_idx] = max(spectrum_data(:));
    [row, col] = ind2sub(size(spectrum_data), max_idx);
    
    theta_est = ang_theta(row);
    phi_est = ang_phi_sorted(col);
end

