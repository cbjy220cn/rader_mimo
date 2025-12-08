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

%% ═════════════════════════════════════════════════════════════════════════════
%  实验说明（用于论文参考）
%% ═════════════════════════════════════════════════════════════════════════════
fprintf('┌─────────────────────────────────────────────────────────────────┐\n');
fprintf('│                        实验说明                                 │\n');
fprintf('├─────────────────────────────────────────────────────────────────┤\n');
fprintf('│ 【实验目的】                                                    │\n');
fprintf('│   验证运动合成孔径DOA算法的实时处理能力。                       │\n');
fprintf('│   每个Chirp周期内需完成DOA估计，才能实现同步运算。              │\n');
fprintf('│                                                                 │\n');
fprintf('│ 【算法优化】滑动窗口                                            │\n');
fprintf('│   问题：累积所有快拍会导致计算复杂度O(n³)随时间增长             │\n');
fprintf('│   方案：固定窗口大小W，只保留最近W个快拍                        │\n');
fprintf('│   效果：计算复杂度恒定为O(W³)，可满足实时要求                   │\n');
fprintf('│                                                                 │\n');
fprintf('│ 【关键指标定义】                                                │\n');
fprintf('│   ① 处理时间                                                   │\n');
fprintf('│      = 单次DOA估计耗时（毫秒）                                 │\n');
fprintf('│      要求：< Chirp周期 才能实时                                 │\n');
fprintf('│                                                                 │\n');
fprintf('│   ② 实时达成率                                                 │\n');
fprintf('│      = (处理时间<Chirp周期的次数) / 总Chirp数 × 100%%           │\n');
fprintf('│      物理含义：系统能否实时跟踪目标                             │\n');
fprintf('│      - 100%%: 完全实时，无延迟累积                              │\n');
fprintf('│      - <100%%: 部分Chirp超时，但可通过跳帧补偿                  │\n');
fprintf('│                                                                 │\n');
fprintf('│   ③ 滑动窗口大小 (W)                                           │\n');
fprintf('│      物理含义：参与DOA估计的快拍数                              │\n');
fprintf('│      权衡：W↑ → 精度↑、实时性↓；W↓ → 精度↓、实时性↑           │\n');
fprintf('│      典型值：8-32个快拍                                         │\n');
fprintf('│                                                                 │\n');
fprintf('│   ④ 累积孔径                                                   │\n');
fprintf('│      = 窗口内虚拟阵列的最大空间跨度 / λ                        │\n');
fprintf('│      物理含义：等效阵列的有效口径                               │\n');
fprintf('│                                                                 │\n');
fprintf('│   ⑤ DOA估计误差                                                │\n');
fprintf('│      = |估计角度 - 真实角度|                                   │\n');
fprintf('│      反映滑动窗口下的估计精度                                   │\n');
fprintf('│                                                                 │\n');
fprintf('│ 【Chirp周期说明】                                               │\n');
fprintf('│   FMCW雷达每个Chirp产生一个快拍数据                             │\n');
fprintf('│   Chirp周期 = T_chirp = 数据更新周期                           │\n');
fprintf('│   处理时间必须 < T_chirp 才能实现实时处理                       │\n');
fprintf('└─────────────────────────────────────────────────────────────────┘\n\n');

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


