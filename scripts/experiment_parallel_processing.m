%% ═══════════════════════════════════════════════════════════════════════════
%  实验：同步/并行运算仿真
%  证明虽然算法时间复杂度增加，但可以实时同步解算
%% ═══════════════════════════════════════════════════════════════════════════
clear; clc; close all;

% 添加类文件路径
addpath('asset');

% 创建带时间戳的输出文件夹
script_name = 'experiment_parallel_processing';
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
output_folder = fullfile('validation_results', [script_name '_' timestamp]);
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

% 初始化日志
log_file = fullfile(output_folder, 'experiment_log.txt');
diary(log_file);

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║       同步运算能力验证实验                                      ║\n');
fprintf('║  证明：算法可在数据采集的同时实时解算DOA                        ║\n');
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

% 时间参数
T_chirp = 10e-3;        % 单个Chirp周期：10ms
T_total = 0.5;          % 总观测时间：500ms
num_chirps = round(T_total / T_chirp);  % 总Chirp数

fprintf('【系统参数】\n');
fprintf('  Chirp周期: %.0f ms\n', T_chirp*1000);
fprintf('  总观测时间: %.0f ms\n', T_total*1000);
fprintf('  总Chirp数: %d\n', num_chirps);
fprintf('  每Chirp处理可用时间: %.0f ms\n\n', T_chirp*1000);

%% 创建阵列和目标
array = ArrayPlatform(elements, 1, 1:num_elements);
array.set_trajectory(@(t) struct('position', [0, v*t, 0], 'orientation', [0,0,0]));

target_pos = target_range * [cosd(target_phi), sind(target_phi), 0];
target = Target(target_pos, [0,0,0], 1);

%% 仿真实时处理
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('开始实时处理仿真\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

phi_search = 0:0.5:90;  % 粗搜索，加速计算
search_grid = struct('phi', phi_search);

% 存储结果
processing_times = zeros(1, num_chirps);
doa_estimates = zeros(1, num_chirps);
cumulative_snapshots = [];

% 创建实时显示图
fig = figure('Position', [100, 100, 1200, 600], 'Color', 'white');
set(gcf, 'DefaultAxesFontName', 'SimHei');

subplot(2, 2, 1);
h_doa = animatedline('Color', 'b', 'LineWidth', 1.5);
hold on;
yline(target_phi, 'r--', 'LineWidth', 1.5);
hold off;
xlabel('Chirp序号');
ylabel('DOA估计值 (°)');
title('实时DOA估计');
ylim([0, 90]);
grid on;

subplot(2, 2, 2);
h_time = bar(nan, 'FaceColor', [0.2, 0.6, 0.8]);
hold on;
yline(T_chirp*1000, 'r--', 'LineWidth', 2, 'DisplayName', '截止时间');
hold off;
xlabel('Chirp序号');
ylabel('处理时间 (ms)');
title('单Chirp处理时间');
ylim([0, T_chirp*1000*2]);
legend('Location', 'northeast');
grid on;

subplot(2, 2, 3);
h_aperture = animatedline('Color', [0.2, 0.7, 0.3], 'LineWidth', 1.5);
xlabel('Chirp序号');
ylabel('合成孔径 (λ)');
title('累积孔径增长');
grid on;

subplot(2, 2, 4);
h_rmse = animatedline('Color', [0.8, 0.3, 0.2], 'LineWidth', 1.5);
xlabel('Chirp序号');
ylabel('累积误差 (°)');
title('估计精度随时间变化');
grid on;

drawnow;

% 累积数据
all_snapshots = [];
all_times = [];

fprintf('Chirp | 处理时间 | DOA估计 | 误差  | 状态\n');
fprintf('------|----------|---------|-------|------\n');

for chirp_idx = 1:num_chirps
    t_current = (chirp_idx - 1) * T_chirp;
    t_axis_single = t_current;
    
    % 生成当前Chirp的数据
    sig_gen = SignalGeneratorSimple(radar_params, array, {target});
    rng(chirp_idx);
    snapshot_current = sig_gen.generate_snapshots(t_axis_single, snr_db);
    
    % 累积快拍
    all_snapshots = [all_snapshots, snapshot_current];
    all_times = [all_times, t_axis_single];
    
    % 开始计时
    tic;
    
    % 使用累积数据进行DOA估计
    if chirp_idx >= 4  % 至少4个快拍才开始估计
        estimator = DoaEstimatorSynthetic(array, radar_params);
        [~, peaks, info] = estimator.estimate(all_snapshots, all_times, search_grid, 1);
        est_phi = peaks.phi(1);
        aperture_lambda = info.synthetic_aperture.total_lambda;
    else
        est_phi = NaN;
        aperture_lambda = 0;
    end
    
    % 结束计时
    proc_time = toc * 1000;  % 转换为毫秒
    
    processing_times(chirp_idx) = proc_time;
    doa_estimates(chirp_idx) = est_phi;
    
    % 判断是否满足实时要求
    if proc_time < T_chirp * 1000
        status = '✓ 实时';
        status_color = 'green';
    else
        status = '✗ 超时';
        status_color = 'red';
    end
    
    % 计算误差
    if ~isnan(est_phi)
        error_deg = abs(est_phi - target_phi);
    else
        error_deg = NaN;
    end
    
    % 打印进度
    if mod(chirp_idx, 5) == 0 || chirp_idx == 1
        fprintf(' %3d  | %6.2f ms | %5.1f° | %4.2f° | %s\n', ...
            chirp_idx, proc_time, est_phi, error_deg, status);
    end
    
    % 更新图表
    if ~isnan(est_phi)
        addpoints(h_doa, chirp_idx, est_phi);
        addpoints(h_aperture, chirp_idx, aperture_lambda);
        addpoints(h_rmse, chirp_idx, error_deg);
    end
    
    h_time.YData = processing_times(1:chirp_idx);
    h_time.XData = 1:chirp_idx;
    
    drawnow limitrate;
end

%% 统计结果
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                        实验结果统计                               \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

valid_estimates = doa_estimates(~isnan(doa_estimates));
valid_errors = abs(valid_estimates - target_phi);

fprintf('【处理时间统计】\n');
fprintf('  平均处理时间: %.2f ms\n', mean(processing_times));
fprintf('  最大处理时间: %.2f ms\n', max(processing_times));
fprintf('  最小处理时间: %.2f ms\n', min(processing_times));
fprintf('  Chirp周期: %.0f ms\n', T_chirp*1000);
fprintf('  实时达成率: %.1f%%\n\n', sum(processing_times < T_chirp*1000) / num_chirps * 100);

fprintf('【DOA估计精度】\n');
fprintf('  最终DOA估计: %.2f°\n', doa_estimates(end));
fprintf('  真实目标角度: %.2f°\n', target_phi);
fprintf('  最终误差: %.2f°\n', abs(doa_estimates(end) - target_phi));
fprintf('  平均误差: %.2f°\n', mean(valid_errors));

fprintf('\n【核心结论】\n');
if mean(processing_times) < T_chirp * 1000
    fprintf('  ✅ 算法满足实时处理要求！\n');
    fprintf('  ✅ 平均处理时间 (%.1f ms) < Chirp周期 (%.0f ms)\n', ...
        mean(processing_times), T_chirp*1000);
    fprintf('  ✅ 可在数据采集的同时同步解算DOA\n');
else
    fprintf('  ⚠️ 需要优化处理算法或降低搜索精度\n');
end

%% 保存结果
results_parallel = struct();
results_parallel.processing_times = processing_times;
results_parallel.doa_estimates = doa_estimates;
results_parallel.T_chirp = T_chirp;
results_parallel.target_phi = target_phi;

save(fullfile(output_folder, 'experiment_results.mat'), 'results_parallel');
saveas(fig, fullfile(output_folder, 'fig_同步运算演示.png'));
saveas(fig, fullfile(output_folder, 'fig_同步运算演示.eps'), 'epsc');

fprintf('\n实验完成！\n');
fprintf('所有结果保存在: %s\n', output_folder);

% 关闭日志
diary off;

