%% ═══════════════════════════════════════════════════════════════════════════
%  实验：实时处理能力验证 v5.0 - 2D滑动窗口版 (时间平滑MUSIC)
%  解决方案：使用固定窗口大小，保证O(1)时间复杂度
%  算法：时间平滑MUSIC（解决合成孔径单快拍秩-1问题）
%  特性：
%    - 使用2D DOA估计（θ俯仰角 + φ方位角）
%    - 4×4 URA（16阵元）+ y平移
%    - 预生成数据保证信号相干性
%    - 滑动窗口保证实时性
%% ═══════════════════════════════════════════════════════════════════════════
clear; clc; close all;

addpath('asset');

% 创建输出文件夹
script_name = 'experiment_realtime_2d';
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
output_folder = fullfile('validation_results', [script_name '_' timestamp]);
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

log_file = fullfile(output_folder, 'experiment_log.txt');
diary(log_file);

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║   实时处理能力验证 v5.0 - 2D滑动窗口版                       ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');
fprintf('输出目录: %s\n\n', output_folder);

%% 参数设置
c = physconst('LightSpeed');
fc = 3e9;
lambda = c / fc;
d = lambda / 2;
radar_params = struct('fc', fc, 'lambda', lambda);

target_phi = 45;       % 方位角
target_theta = 75;     % 俯仰角（非水平面）
target_range = 500;
snr_db = 10;

% 使用4×4 URA（16阵元）
num_x = 4; num_y = 4;
elements = [];
for iy = 1:num_y
    for ix = 1:num_x
        x = (ix - 1 - (num_x-1)/2) * d;
        y = (iy - 1 - (num_y-1)/2) * d;
        elements = [elements; x, y, 0];
    end
end
num_elements = size(elements, 1);

v = 5;           % y方向平移速度
T_chirp = 50e-3; % Chirp周期（放宽到50ms，精度优先）
T_total = 2.0;   % 总观测时间
num_chirps = round(T_total / T_chirp);

% 【关键优化】滑动窗口大小
WINDOW_SIZE = 16;  % 使用16个快拍（0.8秒数据）

% 计算孔径
static_aperture_x = (num_x - 1) * d;
static_aperture_y = (num_y - 1) * d;
synthetic_aperture_y = v * (WINDOW_SIZE * T_chirp);  % 窗口内的合成孔径

fprintf('【系统参数】\n');
fprintf('  阵列: %d×%d URA (%d阵元)\n', num_x, num_y, num_elements);
fprintf('  目标: φ=%.0f°, θ=%.0f°\n', target_phi, target_theta);
fprintf('  静态孔径: x=%.2fλ, y=%.2fλ\n', static_aperture_x/lambda, static_aperture_y/lambda);
fprintf('  窗口合成孔径: y=%.1fλ (%.0fms窗口)\n', synthetic_aperture_y/lambda, WINDOW_SIZE*T_chirp*1000);
fprintf('  Chirp周期: %.0f ms\n', T_chirp*1000);
fprintf('  总Chirp数: %d\n', num_chirps);
fprintf('  滑动窗口大小: %d\n\n', WINDOW_SIZE);

%% 创建阵列和目标
array = ArrayPlatform(elements, 1, 1:num_elements);
array.set_trajectory(@(t) struct('position', [0, v*t, 0], 'orientation', [0,0,0]));

% 3D目标位置
target_pos = target_range * [sind(target_theta)*cosd(target_phi), ...
                              sind(target_theta)*sind(target_phi), ...
                              cosd(target_theta)];
target = Target(target_pos, [0,0,0], 1);

%% 预生成所有快拍数据（保证目标幅度一致性）
fprintf('【预生成数据】\n');
t_all = (0:num_chirps-1) * T_chirp;
sig_gen = SignalGeneratorSimple(radar_params, array, {target});
rng(12345);  % 固定种子，保证可重复性
all_snapshots = sig_gen.generate_snapshots(t_all, snr_db);
fprintf('  完成！共%d个快拍\n\n', num_chirps);

%% 仿真 - 2D搜索（精度优先，使用细网格）
% 搜索网格 - 1°步长，精度更高
phi_search = 20:1:70;      % 1°步长
theta_search = 60:1:90;    % 1°步长
search_grid.phi = phi_search;
search_grid.theta = theta_search;

fprintf('【搜索配置】\n');
fprintf('  φ搜索: %d点 (%.0f°-%.0f°, %.0f°步长)\n', length(phi_search), min(phi_search), max(phi_search), phi_search(2)-phi_search(1));
fprintf('  θ搜索: %d点 (%.0f°-%.0f°, %.0f°步长)\n', length(theta_search), min(theta_search), max(theta_search), theta_search(2)-theta_search(1));
fprintf('  总搜索点: %d\n\n', length(phi_search)*length(theta_search));

est_options.search_mode = '2d';

processing_times = zeros(1, num_chirps);
doa_estimates_phi = zeros(1, num_chirps);
doa_estimates_theta = zeros(1, num_chirps);
doa_errors_phi = zeros(1, num_chirps);
doa_errors_theta = zeros(1, num_chirps);

fprintf('Chirp | 窗口 | 时间(ms) | φ估计 | θ估计 | φ误差 | θ误差 | 状态\n');
fprintf('------|------|----------|-------|-------|-------|-------|------\n');

for chirp_idx = 1:num_chirps
    % 【滑动窗口】从预生成数据中取窗口
    window_start = max(1, chirp_idx - WINDOW_SIZE + 1);
    window_end = chirp_idx;
    
    snapshot_buffer = all_snapshots(:, window_start:window_end);
    time_buffer = t_all(window_start:window_end);
    
    % 计时
    tic;
    
    if size(snapshot_buffer, 2) >= 8  % 至少8个快拍才开始估计
        estimator = DoaEstimatorSynthetic(array, radar_params);
        [~, peaks, ~] = estimator.estimate(snapshot_buffer, time_buffer, search_grid, 1, est_options);
        est_phi = peaks.phi(1);
        est_theta = peaks.theta(1);
    else
        est_phi = NaN;
        est_theta = NaN;
    end
    
    proc_time = toc * 1000;
    
    processing_times(chirp_idx) = proc_time;
    doa_estimates_phi(chirp_idx) = est_phi;
    doa_estimates_theta(chirp_idx) = est_theta;
    
    if ~isnan(est_phi)
        error_phi = abs(est_phi - target_phi);
        error_theta = abs(est_theta - target_theta);
        doa_errors_phi(chirp_idx) = error_phi;
        doa_errors_theta(chirp_idx) = error_theta;
    else
        error_phi = NaN;
        error_theta = NaN;
        doa_errors_phi(chirp_idx) = NaN;
        doa_errors_theta(chirp_idx) = NaN;
    end
    
    if proc_time < T_chirp * 1000
        status = '✓实时';
    else
        status = '✗超时';
    end
    
    if mod(chirp_idx, 10) == 0 || chirp_idx == 1
        fprintf(' %3d  |  %2d  | %7.2f  | %5.1f | %5.1f | %5.2f | %5.2f | %s\n', ...
            chirp_idx, size(snapshot_buffer, 2), proc_time, est_phi, est_theta, error_phi, error_theta, status);
    end
end

%% 统计
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                        实验结果统计                               \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

valid_phi = doa_estimates_phi(~isnan(doa_estimates_phi));
valid_theta = doa_estimates_theta(~isnan(doa_estimates_theta));
valid_errors_phi = doa_errors_phi(~isnan(doa_errors_phi));
valid_errors_theta = doa_errors_theta(~isnan(doa_errors_theta));

realtime_rate = sum(processing_times < T_chirp*1000) / num_chirps * 100;

fprintf('【处理时间统计】\n');
fprintf('  平均处理时间: %.2f ms\n', mean(processing_times));
fprintf('  最大处理时间: %.2f ms\n', max(processing_times));
fprintf('  Chirp周期: %.0f ms\n', T_chirp*1000);
fprintf('  实时达成率: %.1f%%\n\n', realtime_rate);

fprintf('【2D DOA估计精度】\n');
fprintf('  最终φ估计: %.2f° (真值: %.0f°, 误差: %.2f°)\n', ...
    doa_estimates_phi(end), target_phi, abs(doa_estimates_phi(end) - target_phi));
fprintf('  最终θ估计: %.2f° (真值: %.0f°, 误差: %.2f°)\n', ...
    doa_estimates_theta(end), target_theta, abs(doa_estimates_theta(end) - target_theta));
fprintf('  平均φ误差: %.2f°\n', mean(valid_errors_phi));
fprintf('  平均θ误差: %.2f°\n', mean(valid_errors_theta));
fprintf('  稳定后φ均值: %.2f° (后50%%数据)\n', mean(valid_phi(end-floor(length(valid_phi)/2):end)));
fprintf('  稳定后θ均值: %.2f° (后50%%数据)\n', mean(valid_theta(end-floor(length(valid_theta)/2):end)));

fprintf('\n【核心结论】\n');
if mean(valid_errors_phi) < 2 && mean(valid_errors_theta) < 3
    fprintf('  ✅ 2D精度达标！φ误差 %.2f°, θ误差 %.2f°\n', ...
        mean(valid_errors_phi), mean(valid_errors_theta));
    if realtime_rate > 80
        fprintf('  ✅ 实时性也达标！达成率 %.1f%%\n', realtime_rate);
    else
        fprintf('  ⚠️ 实时性需优化（达成率%.1f%%）\n', realtime_rate);
    end
else
    fprintf('  ⚠️ 精度未达标，需检查配置\n');
end

%% 绘图
fig = figure('Position', [100, 100, 1400, 400], 'Color', 'white');
set(gcf, 'DefaultAxesFontName', 'SimHei');

subplot(1, 4, 1);
bar(processing_times, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'none');
hold on;
yline(T_chirp*1000, 'r--', 'LineWidth', 2);
hold off;
xlabel('Chirp序号', 'FontWeight', 'bold');
ylabel('处理时间 (ms)', 'FontWeight', 'bold');
title(sprintf('处理时间 (窗口=%d)', WINDOW_SIZE), 'FontWeight', 'bold');
legend({'处理时间', '截止时间'}, 'Location', 'northeast');
grid on;

subplot(1, 4, 2);
plot(1:num_chirps, doa_estimates_phi, 'b-', 'LineWidth', 1.5);
hold on;
yline(target_phi, 'r--', 'LineWidth', 2);
hold off;
xlabel('Chirp序号', 'FontWeight', 'bold');
ylabel('φ估计值 (°)', 'FontWeight', 'bold');
title('实时φ估计', 'FontWeight', 'bold');
legend({'估计值', '真实值'}, 'Location', 'southeast');
ylim([20, 70]);
grid on;

subplot(1, 4, 3);
plot(1:num_chirps, doa_estimates_theta, 'b-', 'LineWidth', 1.5);
hold on;
yline(target_theta, 'r--', 'LineWidth', 2);
hold off;
xlabel('Chirp序号', 'FontWeight', 'bold');
ylabel('θ估计值 (°)', 'FontWeight', 'bold');
title('实时θ估计', 'FontWeight', 'bold');
legend({'估计值', '真实值'}, 'Location', 'southeast');
ylim([60, 90]);
grid on;

subplot(1, 4, 4);
plot(1:num_chirps, doa_errors_phi, 'b-', 'LineWidth', 1);
hold on;
plot(1:num_chirps, doa_errors_theta, 'r-', 'LineWidth', 1);
yline(2, 'g--', 'LineWidth', 1.5);
hold off;
xlabel('Chirp序号', 'FontWeight', 'bold');
ylabel('估计误差 (°)', 'FontWeight', 'bold');
title('估计误差', 'FontWeight', 'bold');
legend({'φ误差', 'θ误差', '2°线'}, 'Location', 'northeast');
grid on;

sgtitle(sprintf('2D滑动窗口实时DOA估计 (目标φ=%.0f°, θ=%.0f°)', target_phi, target_theta), 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig, fullfile(output_folder, 'fig_实时估计.png'));
saveas(fig, fullfile(output_folder, 'fig_实时估计.eps'), 'epsc');

%% 收敛分析图
fig2 = figure('Position', [100, 100, 1000, 400], 'Color', 'white');
set(gcf, 'DefaultAxesFontName', 'SimHei');

subplot(1, 2, 1);
% 计算滑动平均误差
window_avg = 10;
smoothed_phi = movmean(doa_errors_phi, window_avg, 'omitnan');
smoothed_theta = movmean(doa_errors_theta, window_avg, 'omitnan');

plot(1:num_chirps, doa_errors_phi, 'b-', 'LineWidth', 0.5, 'Color', [0.7, 0.7, 1]);
hold on;
plot(1:num_chirps, smoothed_phi, 'b-', 'LineWidth', 2);
yline(2, 'g--', 'LineWidth', 1.5);
xline(WINDOW_SIZE, 'k:', 'LineWidth', 1.5);
hold off;
xlabel('Chirp序号', 'FontWeight', 'bold');
ylabel('φ误差 (°)', 'FontWeight', 'bold');
title('φ误差收敛分析', 'FontWeight', 'bold');
legend({'瞬时误差', '滑动平均', '2°参考线', '窗口填满'}, 'Location', 'northeast');
grid on;

subplot(1, 2, 2);
plot(1:num_chirps, doa_errors_theta, 'r-', 'LineWidth', 0.5, 'Color', [1, 0.7, 0.7]);
hold on;
plot(1:num_chirps, smoothed_theta, 'r-', 'LineWidth', 2);
yline(2, 'g--', 'LineWidth', 1.5);
xline(WINDOW_SIZE, 'k:', 'LineWidth', 1.5);
hold off;
xlabel('Chirp序号', 'FontWeight', 'bold');
ylabel('θ误差 (°)', 'FontWeight', 'bold');
title('θ误差收敛分析', 'FontWeight', 'bold');
legend({'瞬时误差', '滑动平均', '2°参考线', '窗口填满'}, 'Location', 'northeast');
grid on;

sgtitle('2D DOA估计误差收敛分析', 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig2, fullfile(output_folder, 'fig_收敛分析.png'));

%% 保存
results = struct();
results.processing_times = processing_times;
results.doa_estimates_phi = doa_estimates_phi;
results.doa_estimates_theta = doa_estimates_theta;
results.doa_errors_phi = doa_errors_phi;
results.doa_errors_theta = doa_errors_theta;
results.window_size = WINDOW_SIZE;
results.realtime_rate = realtime_rate;
results.target_phi = target_phi;
results.target_theta = target_theta;
results.mean_error_phi = mean(valid_errors_phi);
results.mean_error_theta = mean(valid_errors_theta);

save(fullfile(output_folder, 'experiment_results.mat'), 'results');

fprintf('\n实验完成！结果保存在: %s\n', output_folder);
diary off;
