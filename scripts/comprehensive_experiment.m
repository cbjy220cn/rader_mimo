%% ═══════════════════════════════════════════════════════════════
%  完整实验：阵列配置 × 运动轨迹 × 速度 - 找最优组合
%  
%  实验设计：
%    - 4种阵列形状：ULA、L型、十字型、方阵
%    - 5种运动轨迹：静止、绕中心旋转、绕边缘旋转、平移、旋转+平移
%    - SNR范围：-10:2:20 dB（16个点）
%    - 搜索精度：0.02° (最终精度)
%    - 蒙特卡洛：100次试验
%  
%  支持断点续传和实时保存
%  
%  作者：基于ISA-MUSIC的合成孔径雷达系统
%  时间：2025-11-23
%% ═══════════════════════════════════════════════════════════════

clear; clc;
addpath('..');

%% 进度文件
progress_file = 'comprehensive_experiment_progress.mat';

fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║  完整实验：阵列×轨迹×速度系统性对比（高精度）          ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

%% 雷达参数
c = 3e8;
f0 = 3e9;
lambda = c / f0;

fprintf('📡 雷达: f₀=%.2f GHz, λ=%.1f cm\n\n', f0/1e9, lambda*100);

%% 实验参数（完整版 - 高精度）
num_elements = 8;           % 所有阵列统一8个阵元
num_snapshots = 64;         % 快拍数（完整版，更多快拍）
num_trials = 100;           % 蒙特卡洛试验次数（完整版）
snr_range = -10:2:20;       % SNR: -10 to 20 dB, step 2 dB（16个点）
final_search_step = 0.02;   % 最终搜索精度 0.02°

% 智能搜索策略
coarse_step = 2.0;          % 粗搜索 2°
fine_step = 0.02;           % 细搜索 0.02°
roi_margin = 10;            % ROI边界 ±10°

% 目标设置
target_theta = 30;  % deg
target_phi = 60;    % deg
target_range = 1000; % m

fprintf('实验配置（完整版 - 高精度）:\n');
fprintf('  阵元数: %d\n', num_elements);
fprintf('  快拍数: %d\n', num_snapshots);
fprintf('  Monte Carlo: %d次\n', num_trials);
fprintf('  SNR范围: %d:%d:%d dB (%d个点)\n', ...
    snr_range(1), snr_range(2)-snr_range(1), snr_range(end), length(snr_range));
fprintf('  搜索策略: %.1f° (粗) → %.2f° (细)\n', coarse_step, fine_step);
fprintf('  目标: θ=%.0f°, φ=%.0f°, R=%.0f m\n\n', target_theta, target_phi, target_range);

%% 阵列配置定义（参考quick_validation_experiment.m）
array_configs = {
    % 名称,       生成函数,                          物理尺寸(m)
    'ULA一字型',  @(N,d) generate_ula(N, d),         0.7*lambda
    'L型阵列',    @(N,d) generate_l_array(N, d),     0.7*lambda
    '十字型',     @(N,d) generate_cross_array(N, d), 0.7*lambda
    '方阵URA',    @(N,d) generate_ura(N, d),         0.7*lambda
};

%% 运动轨迹定义
v_base = 5.0;  % 基准速度 5 m/s
motion_configs = {
    % 名称,           速度,    轨迹函数
    '静止基准',       0,       @(t, R) motion_static()
    '绕中心旋转',     v_base,  @(t, R) motion_rotate_center(t, R, v_base)
    '绕边缘旋转',     v_base,  @(t, R) motion_rotate_edge(t, R, v_base)
    '直线平移',       v_base,  @(t, R) motion_linear(t, v_base)
    '旋转+平移',      v_base,  @(t, R) motion_rotate_translate(t, R, v_base)
};

n_arrays = size(array_configs, 1);
n_motions = size(motion_configs, 1);
n_snr = length(snr_range);
total_configs = n_arrays * n_motions;

fprintf('阵列配置: %d种\n', n_arrays);
for i = 1:n_arrays
    fprintf('  %d. %s\n', i, array_configs{i,1});
end
fprintf('\n');

fprintf('运动轨迹: %d种\n', n_motions);
for i = 1:n_motions
    fprintf('  %d. %s (v=%.1f m/s)\n', i, motion_configs{i,1}, motion_configs{i,2});
end
fprintf('\n');

fprintf('总配置数: %d × %d = %d\n', n_arrays, n_motions, total_configs);
fprintf('总SNR点数: %d\n', n_snr);
fprintf('总试验次数: %d × %d × %d = %d\n', ...
    total_configs, n_snr, num_trials, total_configs * n_snr * num_trials);

% 预估时间（每次DOA估计约0.5秒）
est_time_per_trial = 0.5;  % 秒
total_est_time = total_configs * n_snr * num_trials * est_time_per_trial / 3600;
fprintf('预计总耗时: %.1f 小时\n\n', total_est_time);

fprintf('⚠️  这是一个长时间实验！请确保:\n');
fprintf('   1. 电脑不会休眠\n');
fprintf('   2. 支持断点续传（可随时Ctrl+C中断）\n');
fprintf('   3. 每完成一个SNR点自动保存\n\n');

%% 检查是否有进度文件（断点续传）
if exist(progress_file, 'file')
    fprintf('发现进度文件，加载中...\n');
    load(progress_file);
    fprintf('✓ 已恢复进度: 数组%d/%d, 轨迹%d/%d, SNR点%d/%d\n\n', ...
        current_arr_idx, n_arrays, current_mot_idx, n_motions, current_snr_idx, n_snr);
else
    % 初始化结果结构
    results = struct();
    results.array_names = array_configs(:,1);
    results.motion_names = motion_configs(:,1);
    results.snr_range = snr_range;
    results.rmse = nan(n_arrays, n_motions, n_snr);
    results.mean_error = nan(n_arrays, n_motions, n_snr);
    results.std_error = nan(n_arrays, n_motions, n_snr);
    results.computation_time = zeros(n_arrays, n_motions, n_snr);
    results.config = struct(...
        'num_elements', num_elements, ...
        'num_snapshots', num_snapshots, ...
        'num_trials', num_trials, ...
        'target', [target_theta, target_phi, target_range], ...
        'search_resolution', [coarse_step, fine_step]);
    
    % 进度控制
    current_arr_idx = 1;
    current_mot_idx = 1;
    current_snr_idx = 1;
end

%% 搜索网格（粗搜索用）
search_grid.theta = 0:coarse_step:90;
search_grid.phi = 0:coarse_step:180;

%% 雷达参数结构
radar_params.c = c;
radar_params.f0 = f0;
radar_params.bandwidth = 100e6;
radar_params.range_res = c / (2 * radar_params.bandwidth);

%% 主实验循环
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('开始完整实验\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

tic_total = tic;

try
    for arr_idx = current_arr_idx:n_arrays
        array_name = array_configs{arr_idx, 1};
        array_func = array_configs{arr_idx, 2};
        array_size = array_configs{arr_idx, 3};
        
        % 生成阵列
        array_pos = array_func(num_elements, array_size);
        array_radius = max(sqrt(sum(array_pos.^2, 1)));
        
        % 确定运动轨迹起始索引
        start_mot_idx = (arr_idx == current_arr_idx) * current_mot_idx + ...
                        (arr_idx > current_arr_idx) * 1;
        
        for mot_idx = start_mot_idx:n_motions
            motion_name = motion_configs{mot_idx, 1};
            motion_speed = motion_configs{mot_idx, 2};
            motion_func = motion_configs{mot_idx, 3};
            
            config_num = (arr_idx-1)*n_motions + mot_idx;
            
            fprintf('╔════════════════════════════════════════════════════════╗\n');
            fprintf('║ [配置 %2d/%2d] %s + %s\n', ...
                config_num, total_configs, array_name, motion_name);
            fprintf('║ 阵列半径: %.2f cm | 速度: %.1f m/s\n', array_radius*100, motion_speed);
            fprintf('╚════════════════════════════════════════════════════════╝\n');
            
            % 确定SNR起始索引
            start_snr_idx = (arr_idx == current_arr_idx && mot_idx == current_mot_idx) * current_snr_idx + ...
                           (arr_idx > current_arr_idx || mot_idx > current_mot_idx) * 1;
            
            for snr_idx = start_snr_idx:n_snr
                snr_db = snr_range(snr_idx);
                
                fprintf('\n  SNR = %+3d dB [%2d/%2d]: ', snr_db, snr_idx, n_snr);
                tic_snr = tic;
                
                errors = zeros(num_trials, 1);
                
                % 蒙特卡洛试验
                for trial = 1:num_trials
                    if mod(trial, 20) == 0
                        fprintf('%d', trial);
                    elseif mod(trial, 10) == 0
                        fprintf('.');
                    end
                    
                    % 生成运动平台
                    platform = ArrayPlatform_Motion(...
                        array_pos, ...
                        @(t) motion_func(t, array_radius), ...
                        lambda);
                    
                    % 生成信号
                    sig_gen = SignalGenerator(platform);
                    [snapshots, ~] = sig_gen.generate_snapshots(...
                        num_snapshots, ...
                        [target_theta; target_phi], ...
                        [target_range], ...
                        snr_db, ...
                        radar_params);
                    
                    % DOA估计（智能两步搜索）
                    if motion_speed > 0
                        estimator = DoaEstimatorIncoherent_FIXED(platform, ...
                            'num_segments', 8, 'verbose', false);
                    else
                        % 静止用相干MUSIC（更快）
                        estimator = DoaEstimatorIncoherent_FIXED(platform, ...
                            'num_segments', 1, 'verbose', false);
                    end
                    
                    [~, doa_estimates] = estimator.estimate_doa_smart(...
                        snapshots, 1, search_grid, ...
                        struct('coarse_step', coarse_step, 'fine_step', fine_step, ...
                               'roi_margin', roi_margin));
                    
                    % 计算误差
                    est_theta = doa_estimates.theta(1);
                    est_phi = doa_estimates.phi(1);
                    error = sqrt((est_theta - target_theta)^2 + (est_phi - target_phi)^2);
                    errors(trial) = error;
                end
                
                % 统计结果
                rmse = sqrt(mean(errors.^2));
                mean_err = mean(errors);
                std_err = std(errors);
                elapsed = toc(tic_snr);
                
                results.rmse(arr_idx, mot_idx, snr_idx) = rmse;
                results.mean_error(arr_idx, mot_idx, snr_idx) = mean_err;
                results.std_error(arr_idx, mot_idx, snr_idx) = std_err;
                results.computation_time(arr_idx, mot_idx, snr_idx) = elapsed;
                
                fprintf(' → RMSE=%.3f° (μ=%.3f°, σ=%.3f°) [%.1fs]\n', ...
                    rmse, mean_err, std_err, elapsed);
                
                % 实时保存（每完成一个SNR点）
                current_arr_idx = arr_idx;
                current_mot_idx = mot_idx;
                current_snr_idx = snr_idx + 1;
                save(progress_file, 'results', 'current_arr_idx', 'current_mot_idx', 'current_snr_idx', ...
                     'array_configs', 'motion_configs');
            end
            
            % 重置SNR索引
            current_snr_idx = 1;
        end
        
        % 重置运动索引
        current_mot_idx = 1;
    end
    
    fprintf('\n╔════════════════════════════════════════════════════════╗\n');
    fprintf('║  所有实验完成！总耗时: %.1f 小时\n', toc(tic_total)/3600);
    fprintf('╚════════════════════════════════════════════════════════╝\n\n');
    
catch ME
    fprintf('\n\n⚠️  实验中断: %s\n', ME.message);
    fprintf('进度已保存到: %s\n', progress_file);
    fprintf('重新运行脚本即可从断点继续\n');
    rethrow(ME);
end

%% 保存最终结果
final_results_file = sprintf('comprehensive_experiment_results_%s.mat', ...
    datestr(now, 'yyyymmdd_HHMMSS'));
save(final_results_file, 'results');
fprintf('✓ 最终结果已保存: %s\n\n', final_results_file);

%% 结果分析和可视化
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('结果分析\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

analyze_and_visualize_results(results);

fprintf('✓ 所有图片已保存到当前目录\n\n');

%% 生成最优配置报告
generate_optimal_config_report(results);

% 删除进度文件（实验已完成）
if exist(progress_file, 'file')
    delete(progress_file);
    fprintf('\n✓ 进度文件已清理\n');
end

fprintf('\n╔════════════════════════════════════════════════════════╗\n');
fprintf('║  实验系统完成！                                        ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n');

%% ═══════════════════════════════════════════════════════════════
%  阵列生成函数
%% ═══════════════════════════════════════════════════════════════

function pos = generate_ula(N, spacing)
    x = (0:N-1) * spacing - (N-1)*spacing/2;
    pos = [x; zeros(1,N); zeros(1,N)];
end

function pos = generate_l_array(N, spacing)
    N_arm = ceil(N/2);
    x_arm = (0:N_arm-1) * spacing;
    pos_x = [x_arm; zeros(1,N_arm); zeros(1,N_arm)];
    y_arm = (1:N-N_arm) * spacing;
    pos_y = [zeros(1,N-N_arm); y_arm; zeros(1,N-N_arm)];
    pos = [pos_x, pos_y];
end

function pos = generate_cross_array(N, spacing)
    N_arm = ceil(N/4);
    arm_len = (1:N_arm) * spacing;
    pos_px = [arm_len; zeros(1,N_arm); zeros(1,N_arm)];
    pos_nx = [-arm_len; zeros(1,N_arm); zeros(1,N_arm)];
    N_remain = N - 3*N_arm;
    arm_len_y = (1:N_remain) * spacing;
    pos_py = [zeros(1,N_remain); arm_len_y; zeros(1,N_remain)];
    pos_ny = [zeros(1,N_arm); -arm_len; zeros(1,N_arm)];
    pos_0 = [0; 0; 0];
    pos = [pos_0, pos_px, pos_nx, pos_py, pos_ny];
    pos = pos(:, 1:N);
end

function pos = generate_ura(N, spacing)
    N_side = ceil(sqrt(N));
    [X, Y] = meshgrid(0:N_side-1, 0:N_side-1);
    X = X(:)' * spacing - (N-1)*spacing/2;
    Y = Y(:)' * spacing - (N-1)*spacing/2;
    pos = [X(1:N); Y(1:N); zeros(1,N)];
end

%% ═══════════════════════════════════════════════════════════════
%  运动轨迹函数
%% ═══════════════════════════════════════════════════════════════

function state = motion_static()
    state.position = [0; 0; 0];
    state.velocity = [0; 0; 0];
    state.rotation_matrix = eye(3);
end

function state = motion_rotate_center(t, R, v)
    omega = v / (2*pi*R);
    angle = omega * 2*pi * t;
    state.position = [0; 0; 0];
    state.velocity = [0; 0; 0];
    state.rotation_matrix = [cos(angle), -sin(angle), 0;
                            sin(angle),  cos(angle), 0;
                            0,           0,          1];
end

function state = motion_rotate_edge(t, R, v)
    r_orbit = 2*R;
    omega = v / r_orbit;
    angle = omega * t;
    state.position = [r_orbit*cos(angle); r_orbit*sin(angle); 0];
    state.velocity = [-r_orbit*omega*sin(angle); r_orbit*omega*cos(angle); 0];
    state.rotation_matrix = eye(3);
end

function state = motion_linear(t, v)
    state.position = [v*t; 0; 0];
    state.velocity = [v; 0; 0];
    state.rotation_matrix = eye(3);
end

function state = motion_rotate_translate(t, R, v)
    omega = v / (4*pi*R);
    angle = omega * 2*pi * t;
    state.position = [v*t; 0; 0];
    state.velocity = [v; 0; 0];
    state.rotation_matrix = [cos(angle), -sin(angle), 0;
                            sin(angle),  cos(angle), 0;
                            0,           0,          1];
end

%% ═══════════════════════════════════════════════════════════════
%  结果分析和可视化
%% ═══════════════════════════════════════════════════════════════

function analyze_and_visualize_results(results)
    n_arrays = length(results.array_names);
    n_motions = length(results.motion_names);
    
    % 图1: RMSE vs SNR曲线（每个阵列的最优运动）
    figure('Position', [100, 100, 1600, 1000]);
    
    subplot(2,3,1);
    hold on; grid on;
    colors = lines(n_arrays);
    for arr_idx = 1:n_arrays
        rmse_best = squeeze(min(results.rmse(arr_idx, :, :), [], 2));
        plot(results.snr_range, rmse_best, 'o-', 'LineWidth', 2, ...
            'Color', colors(arr_idx,:), 'DisplayName', results.array_names{arr_idx});
    end
    xlabel('SNR (dB)'); ylabel('RMSE (度)');
    title('RMSE vs SNR（最优运动）');
    legend('Location', 'best');
    set(gca, 'YScale', 'log');
    
    % 图2: 静止 vs 运动改善倍数
    subplot(2,3,2);
    high_snr_idx = find(results.snr_range == 10, 1);
    if isempty(high_snr_idx), high_snr_idx = length(results.snr_range); end
    
    improvement = zeros(n_arrays, n_motions-1);
    for arr_idx = 1:n_arrays
        static_rmse = results.rmse(arr_idx, 1, high_snr_idx);
        for mot_idx = 2:n_motions
            improvement(arr_idx, mot_idx-1) = static_rmse / results.rmse(arr_idx, mot_idx, high_snr_idx);
        end
    end
    
    bar(improvement');
    set(gca, 'XTickLabel', results.motion_names(2:end));
    ylabel('改善倍数');
    title(sprintf('性能改善 (SNR=%ddB)', results.snr_range(high_snr_idx)));
    legend(results.array_names, 'Location', 'best');
    grid on;
    
    % 图3: 热力图 - 阵列 × 运动
    subplot(2,3,3);
    rmse_matrix = results.rmse(:, :, high_snr_idx);
    imagesc(log10(rmse_matrix));
    colorbar;
    title(sprintf('log10(RMSE) 热力图 (SNR=%ddB)', results.snr_range(high_snr_idx)));
    set(gca, 'XTick', 1:n_motions, 'XTickLabel', results.motion_names, 'XTickLabelRotation', 45);
    set(gca, 'YTick', 1:n_arrays, 'YTickLabel', results.array_names);
    
    % 图4: 不同运动轨迹对比（选最优阵列）
    subplot(2,3,4);
    hold on; grid on;
    colors_mot = lines(n_motions);
    [~, best_arr_idx] = min(mean(results.rmse(:, :, :), [2,3]));
    for mot_idx = 1:n_motions
        rmse_curve = squeeze(results.rmse(best_arr_idx, mot_idx, :));
        plot(results.snr_range, rmse_curve, 'o-', 'LineWidth', 2, ...
            'Color', colors_mot(mot_idx,:), 'DisplayName', results.motion_names{mot_idx});
    end
    xlabel('SNR (dB)'); ylabel('RMSE (度)');
    title(sprintf('运动轨迹对比 (%s)', results.array_names{best_arr_idx}));
    legend('Location', 'best');
    set(gca, 'YScale', 'log');
    
    % 图5: 低SNR性能对比
    subplot(2,3,5);
    low_snr_idx = find(results.snr_range == -5, 1);
    if isempty(low_snr_idx), low_snr_idx = 1; end
    
    rmse_low = results.rmse(:, :, low_snr_idx);
    bar(rmse_low);
    set(gca, 'XTickLabel', results.array_names);
    ylabel('RMSE (度)');
    title(sprintf('低SNR性能 (SNR=%ddB)', results.snr_range(low_snr_idx)));
    legend(results.motion_names, 'Location', 'best');
    grid on;
    
    % 图6: 计算效率对比
    subplot(2,3,6);
    avg_time = mean(results.computation_time, 3);
    bar(avg_time);
    set(gca, 'XTickLabel', results.array_names);
    ylabel('平均时间 (秒)');
    title('计算效率对比');
    legend(results.motion_names, 'Location', 'best');
    grid on;
    
    saveas(gcf, 'comprehensive_results_overview.png');
    
    % 更多详细图表
    plot_detailed_snr_curves(results);
    plot_improvement_heatmaps(results);
end

function plot_detailed_snr_curves(results)
    % 为每个阵列配置绘制详细的SNR曲线
    n_arrays = length(results.array_names);
    
    figure('Position', [100, 100, 1600, 1000]);
    for arr_idx = 1:n_arrays
        subplot(2, 2, arr_idx);
        hold on; grid on;
        colors = lines(length(results.motion_names));
        
        for mot_idx = 1:length(results.motion_names)
            rmse_curve = squeeze(results.rmse(arr_idx, mot_idx, :));
            plot(results.snr_range, rmse_curve, 'o-', 'LineWidth', 2, ...
                'Color', colors(mot_idx,:), 'DisplayName', results.motion_names{mot_idx});
        end
        
        xlabel('SNR (dB)'); ylabel('RMSE (度)');
        title(results.array_names{arr_idx});
        legend('Location', 'best');
        set(gca, 'YScale', 'log');
    end
    
    sgtitle('各阵列配置详细RMSE曲线');
    saveas(gcf, 'comprehensive_results_detailed_snr.png');
end

function plot_improvement_heatmaps(results)
    % 绘制改善倍数热力图（多个SNR点）
    n_arrays = length(results.array_names);
    n_motions = length(results.motion_names);
    
    % 选择几个关键SNR点
    key_snr_values = [-5, 0, 5, 10, 15, 20];
    key_snr_indices = arrayfun(@(s) find(results.snr_range == s, 1), key_snr_values);
    key_snr_indices = key_snr_indices(~isempty(key_snr_indices));
    
    figure('Position', [100, 100, 1600, 1000]);
    for i = 1:length(key_snr_indices)
        snr_idx = key_snr_indices(i);
        
        improvement = zeros(n_arrays, n_motions-1);
        for arr_idx = 1:n_arrays
            static_rmse = results.rmse(arr_idx, 1, snr_idx);
            for mot_idx = 2:n_motions
                improvement(arr_idx, mot_idx-1) = static_rmse / results.rmse(arr_idx, mot_idx, snr_idx);
            end
        end
        
        subplot(2, 3, i);
        imagesc(improvement);
        colorbar;
        caxis([1, max(improvement(:))]);
        title(sprintf('改善倍数 (SNR=%ddB)', results.snr_range(snr_idx)));
        set(gca, 'XTick', 1:n_motions-1, 'XTickLabel', results.motion_names(2:end), 'XTickLabelRotation', 45);
        set(gca, 'YTick', 1:n_arrays, 'YTickLabel', results.array_names);
    end
    
    sgtitle('性能改善倍数热力图（不同SNR）');
    saveas(gcf, 'comprehensive_results_improvement_heatmaps.png');
end

%% ═══════════════════════════════════════════════════════════════
%  生成最优配置报告
%% ═══════════════════════════════════════════════════════════════

function generate_optimal_config_report(results)
    fprintf('╔════════════════════════════════════════════════════════╗\n');
    fprintf('║              最优配置报告                              ║\n');
    fprintf('╚════════════════════════════════════════════════════════╝\n\n');
    
    % 在不同SNR下找最优组合
    test_snr_values = [0, 5, 10, 15];
    
    for snr_val = test_snr_values
        snr_idx = find(results.snr_range == snr_val, 1);
        if isempty(snr_idx), continue; end
        
        [min_rmse, min_idx] = min(results.rmse(:, :, snr_idx), [], 'all', 'linear');
        [best_arr_idx, best_mot_idx] = ind2sub([length(results.array_names), length(results.motion_names)], min_idx);
        
        fprintf('SNR = %+3d dB:\n', snr_val);
        fprintf('  最优组合: %s + %s\n', ...
            results.array_names{best_arr_idx}, results.motion_names{best_mot_idx});
        fprintf('  RMSE: %.3f°\n', min_rmse);
        
        % 对比静止
        static_rmse = results.rmse(best_arr_idx, 1, snr_idx);
        improvement = static_rmse / min_rmse;
        fprintf('  相比静止改善: %.1fx\n\n', improvement);
    end
    
    % 全局最优（平均所有SNR）
    avg_rmse = mean(results.rmse, 3);
    [min_avg_rmse, min_idx] = min(avg_rmse(:));
    [best_arr_idx, best_mot_idx] = ind2sub(size(avg_rmse), min_idx);
    
    fprintf('全局最优（所有SNR平均）:\n');
    fprintf('  最优组合: %s + %s\n', ...
        results.array_names{best_arr_idx}, results.motion_names{best_mot_idx});
    fprintf('  平均RMSE: %.3f°\n\n', min_avg_rmse);
    
    % 保存报告到文件
    report_file = sprintf('optimal_config_report_%s.txt', datestr(now, 'yyyymmdd_HHMMSS'));
    fid = fopen(report_file, 'w');
    fprintf(fid, '实验最优配置报告\n');
    fprintf(fid, '生成时间: %s\n\n', datestr(now));
    fprintf(fid, '全局最优组合:\n');
    fprintf(fid, '  阵列: %s\n', results.array_names{best_arr_idx});
    fprintf(fid, '  运动: %s\n', results.motion_names{best_mot_idx});
    fprintf(fid, '  平均RMSE: %.3f°\n', min_avg_rmse);
    fclose(fid);
    
    fprintf('✓ 报告已保存: %s\n', report_file);
end

