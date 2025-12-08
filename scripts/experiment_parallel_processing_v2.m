%% ═══════════════════════════════════════════════════════════════════════════
%  实验：同步/并行运算仿真 v2.0 - 滑动窗口优化版
%  解决方案：使用固定窗口大小，保证O(1)时间复杂度
%% ═══════════════════════════════════════════════════════════════════════════
clear; clc; close all;

addpath('asset');

% 创建输出文件夹
script_name = 'experiment_parallel_v2';
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
output_folder = fullfile('validation_results', [script_name '_' timestamp]);
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

log_file = fullfile(output_folder, 'experiment_log.txt');
diary(log_file);

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║       同步运算能力验证 v2.0 - 滑动窗口优化版                    ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');
fprintf('输出目录: %s\n\n', output_folder);

%% 参数设置
c = physconst('LightSpeed');
fc = 3e9;
lambda = c / fc;
d = lambda / 2;
radar_params = struct('fc', fc, 'lambda', lambda);

target_phi = 30;
target_range = 500;
snr_db = 10;

num_elements = 8;
x_pos = ((0:num_elements-1) - (num_elements-1)/2) * d;
elements = [x_pos', zeros(num_elements,1), zeros(num_elements,1)];

v = 5;
T_chirp = 10e-3;
T_total = 0.5;
num_chirps = round(T_total / T_chirp);

% 【关键优化】滑动窗口大小（固定协方差矩阵维度）
WINDOW_SIZE = 16;  % 只使用最近16个快拍

fprintf('【系统参数】\n');
fprintf('  Chirp周期: %.0f ms\n', T_chirp*1000);
fprintf('  总Chirp数: %d\n', num_chirps);
fprintf('  滑动窗口大小: %d (固定处理复杂度)\n\n', WINDOW_SIZE);

%% 创建阵列和目标
array = ArrayPlatform(elements, 1, 1:num_elements);
array.set_trajectory(@(t) struct('position', [0, v*t, 0], 'orientation', [0,0,0]));

target_pos = target_range * [cosd(target_phi), sind(target_phi), 0];
target = Target(target_pos, [0,0,0], 1);

%% 仿真
phi_search = 0:0.5:90;
search_grid = struct('phi', phi_search);

processing_times = zeros(1, num_chirps);
doa_estimates = zeros(1, num_chirps);

% 滑动窗口缓冲区
snapshot_buffer = [];
time_buffer = [];

fprintf('Chirp | 窗口大小 | 处理时间 | DOA估计 | 误差  | 状态\n');
fprintf('------|----------|----------|---------|-------|------\n');

for chirp_idx = 1:num_chirps
    t_current = (chirp_idx - 1) * T_chirp;
    
    % 生成当前快拍
    sig_gen = SignalGeneratorSimple(radar_params, array, {target});
    rng(chirp_idx);
    snapshot_current = sig_gen.generate_snapshots(t_current, snr_db);
    
    % 【滑动窗口】添加新数据，移除旧数据
    snapshot_buffer = [snapshot_buffer, snapshot_current];
    time_buffer = [time_buffer, t_current];
    
    if size(snapshot_buffer, 2) > WINDOW_SIZE
        snapshot_buffer = snapshot_buffer(:, end-WINDOW_SIZE+1:end);
        time_buffer = time_buffer(end-WINDOW_SIZE+1:end);
    end
    
    % 计时
    tic;
    
    if size(snapshot_buffer, 2) >= 4
        estimator = DoaEstimatorSynthetic(array, radar_params);
        [~, peaks, ~] = estimator.estimate(snapshot_buffer, time_buffer, search_grid, 1);
        est_phi = peaks.phi(1);
    else
        est_phi = NaN;
    end
    
    proc_time = toc * 1000;
    
    processing_times(chirp_idx) = proc_time;
    doa_estimates(chirp_idx) = est_phi;
    
    if proc_time < T_chirp * 1000
        status = '✓ 实时';
    else
        status = '✗ 超时';
    end
    
    if ~isnan(est_phi)
        error_deg = abs(est_phi - target_phi);
    else
        error_deg = NaN;
    end
    
    if mod(chirp_idx, 5) == 0 || chirp_idx == 1
        fprintf(' %3d  |    %2d    | %6.2f ms | %5.1f° | %4.2f° | %s\n', ...
            chirp_idx, size(snapshot_buffer, 2), proc_time, est_phi, error_deg, status);
    end
end

%% 统计
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                        实验结果统计                               \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

valid_estimates = doa_estimates(~isnan(doa_estimates));
valid_errors = abs(valid_estimates - target_phi);

realtime_rate = sum(processing_times < T_chirp*1000) / num_chirps * 100;

fprintf('【处理时间统计】\n');
fprintf('  平均处理时间: %.2f ms\n', mean(processing_times));
fprintf('  最大处理时间: %.2f ms\n', max(processing_times));
fprintf('  Chirp周期: %.0f ms\n', T_chirp*1000);
fprintf('  实时达成率: %.1f%% (优化前: ~36%%)\n\n', realtime_rate);

fprintf('【DOA估计精度】\n');
fprintf('  最终DOA估计: %.2f°\n', doa_estimates(end));
fprintf('  真实角度: %.2f°\n', target_phi);
fprintf('  最终误差: %.2f°\n', abs(doa_estimates(end) - target_phi));

fprintf('\n【核心结论】\n');
if realtime_rate > 90
    fprintf('  ✅ 滑动窗口优化成功！实时达成率 %.1f%%\n', realtime_rate);
else
    fprintf('  ⚠️ 需进一步优化（增大窗口或使用GPU）\n');
end

%% 绘图
fig = figure('Position', [100, 100, 1000, 400], 'Color', 'white');
set(gcf, 'DefaultAxesFontName', 'SimHei');

subplot(1, 2, 1);
bar(processing_times, 'FaceColor', [0.2, 0.6, 0.8]);
hold on;
yline(T_chirp*1000, 'r--', 'LineWidth', 2);
hold off;
xlabel('Chirp序号');
ylabel('处理时间 (ms)');
title(sprintf('滑动窗口=%d: 处理时间稳定', WINDOW_SIZE));
legend({'处理时间', '截止时间'}, 'Location', 'northeast');
grid on;

subplot(1, 2, 2);
plot(1:num_chirps, doa_estimates, 'b-', 'LineWidth', 1.5);
hold on;
yline(target_phi, 'r--', 'LineWidth', 1.5);
hold off;
xlabel('Chirp序号');
ylabel('DOA估计值 (°)');
title('实时DOA估计');
legend({'估计值', '真实值'}, 'Location', 'northeast');
ylim([0, 90]);
grid on;

sgtitle('滑动窗口优化: O(1)时间复杂度', 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig, fullfile(output_folder, 'fig_滑动窗口优化.png'));
saveas(fig, fullfile(output_folder, 'fig_滑动窗口优化.eps'), 'epsc');

%% 保存
results = struct();
results.processing_times = processing_times;
results.doa_estimates = doa_estimates;
results.window_size = WINDOW_SIZE;
results.realtime_rate = realtime_rate;

save(fullfile(output_folder, 'experiment_results.mat'), 'results');

fprintf('\n实验完成！结果保存在: %s\n', output_folder);
diary off;


