%% ═══════════════════════════════════════════════════════════════
%  快速验证实验：阵列配置 × 运动轨迹 × 速度组合
%  目标：验证所有配置能正常运行（小规模测试）
%  
%  作者：基于ISA-MUSIC的合成孔径雷达系统
%  时间：2025-11-23
%% ═══════════════════════════════════════════════════════════════

clear; clc;
addpath('..');

fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║   快速验证：阵列×轨迹×速度组合（预计10-20分钟）        ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

%% 雷达参数
c = 3e8;
f0 = 3e9;
lambda = c / f0;

fprintf('📡 雷达: f₀=%.2f GHz, λ=%.1f cm\n\n', f0/1e9, lambda*100);

%% 实验参数（快速验证版）
num_elements = 8;           % 所有阵列统一8个阵元
num_snapshots = 16;         % ⚠️ 快拍数≈2×阵元数（避免噪声子空间过大导致谱平坦）
num_trials = 10;            % 蒙特卡洛试验次数（快速版，降低到10次）
snr_range = [0, 10];        % SNR快速测试点（只测2个点）
search_step = 1.0;          % 搜索步长（快速版，改为1°）

% 目标设置（改为接近xy平面，让各种阵列都能有效工作）
target_theta = 80;  % deg (接近xy平面：sin(80°)≈0.985)
target_phi = 45;    % deg  (x和y分量相等)
target_range = 1000; % m
% 方向矢量 u = [sin(80°)*cos(45°), sin(80°)*sin(45°), cos(80°)]
%            ≈ [0.697, 0.697, 0.174]
% ULA(x轴)、L型、十字型都能有效分辨

fprintf('实验配置（快速验证）:\n');
fprintf('  阵元数: %d\n', num_elements);
fprintf('  快拍数: %d\n', num_snapshots);
fprintf('  Monte Carlo: %d次\n', num_trials);
fprintf('  SNR测试点: %s dB\n', mat2str(snr_range));
fprintf('  搜索步长: %.2f°\n\n', search_step);

%% 阵列配置定义
array_configs = {
    % 名称,       生成函数,                          阵元间距(m)
    % 增大间距以获得更大的总基线（总长度 = (N-1) × 间距）
    'ULA一字型',  @(N,d) generate_ula(N, d),         1.5*lambda  % 1.5λ间距 → 总长10.5λ
    'L型阵列',    @(N,d) generate_l_array(N, d),     1.5*lambda
    '十字型',     @(N,d) generate_cross_array(N, d), 1.5*lambda
    '方阵URA',    @(N,d) generate_ura(N, d),         1.5*lambda
};

fprintf('阵列配置: %d种\n', size(array_configs, 1));
for i = 1:size(array_configs, 1)
    fprintf('  %d. %s\n', i, array_configs{i,1});
end
fprintf('\n');

%% 运动轨迹定义
% 基准速度：5 m/s（无人机巡航速度）
v_base = 5.0;
t_obs = (num_snapshots - 1) * 0.1;  % 观测时间（假设100ms快拍间隔）

motion_configs = {
    % 名称,           速度,    轨迹函数
    '静止基准',       0,       @(t, R) motion_static()
    '绕中心旋转',     v_base,  @(t, R) motion_rotate_center(t, R, v_base)
    '绕边缘旋转',     v_base,  @(t, R) motion_rotate_edge(t, R, v_base)
    '直线平移',       v_base,  @(t, R) motion_linear(t, v_base)
    '旋转+平移',      v_base,  @(t, R) motion_rotate_translate(t, R, v_base)
};

fprintf('运动轨迹: %d种\n', size(motion_configs, 1));
for i = 1:size(motion_configs, 1)
    fprintf('  %d. %s (v=%.1f m/s)\n', i, motion_configs{i,1}, motion_configs{i,2});
end
fprintf('\n');

%% 总组合数
total_configs = size(array_configs, 1) * size(motion_configs, 1);
fprintf('总组合数: %d × %d = %d\n', size(array_configs, 1), size(motion_configs, 1), total_configs);
fprintf('预计耗时: %.1f 分钟（快速验证）\n\n', total_configs * length(snr_range) * num_trials * 0.5 / 60);

%% 初始化结果存储
results = struct();
results.array_names = array_configs(:,1);
results.motion_names = motion_configs(:,1);
results.snr_range = snr_range;
results.rmse = zeros(size(array_configs,1), size(motion_configs,1), length(snr_range));
results.computation_time = zeros(size(array_configs,1), size(motion_configs,1));

%% 搜索网格和智能搜索配置
search_grid.theta = 0:search_step:90;
search_grid.phi = 0:search_step:180;

% 快速验证：使用粗网格（不用智能搜索，避免复杂度）
USE_SMART_SEARCH = false;  % 快速验证用粗网格即可
fprintf('✓ 快速验证模式：搜索步长 %.1f°\n\n', search_step);

%% 雷达参数结构（完整）
radar_params.c = c;
radar_params.f0 = f0;
radar_params.fc = f0;
radar_params.lambda = lambda;
radar_params.bandwidth = 100e6;
radar_params.BW = 100e6;
radar_params.range_res = c / (2 * 100e6);
radar_params.fs = 36100;
radar_params.T_chirp = 10e-3;
radar_params.slope = 5e12;
radar_params.num_samples = 361;

%% 开始实验
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('开始快速验证实验\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

config_idx = 0;
tic_total = tic;

for arr_idx = 1:size(array_configs, 1)
    array_name = array_configs{arr_idx, 1};
    array_func = array_configs{arr_idx, 2};
    array_size = array_configs{arr_idx, 3};
    
    % 生成阵列
    array_pos = array_func(num_elements, array_size);
    array_radius = max(sqrt(sum(array_pos.^2, 2)));  % 用于运动轨迹（N×3格式）
    
    for mot_idx = 1:size(motion_configs, 1)
        config_idx = config_idx + 1;
        motion_name = motion_configs{mot_idx, 1};
        motion_speed = motion_configs{mot_idx, 2};
        motion_func = motion_configs{mot_idx, 3};
        
        fprintf('[%2d/%2d] %s + %s\n', config_idx, total_configs, array_name, motion_name);
        fprintf('        阵列半径: %.2f cm | 速度: %.1f m/s\n', array_radius*100, motion_speed);
        
        tic_config = tic;
        
        % 对每个SNR点进行测试
        for snr_idx = 1:length(snr_range)
            snr_db = snr_range(snr_idx);
            
            errors = zeros(num_trials, 1);
            
            for trial = 1:num_trials
                % 生成运动平台
                platform = ArrayPlatform(array_pos, 1, 1:size(array_pos,1));
                platform = platform.set_trajectory(@(t) motion_func(t, array_radius));
                
                % 创建目标
                target_pos = [target_range * sind(target_theta) * cosd(target_phi), ...
                             target_range * sind(target_theta) * sind(target_phi), ...
                             target_range * cosd(target_theta)];
                targets = {Target(target_pos, [0,0,0], 1)};
                
                % 生成快拍
                t_axis = (0:num_snapshots-1) * radar_params.T_chirp;
                sig_gen = SignalGenerator(radar_params, platform, targets);
                snapshots = sig_gen.generate_snapshots(t_axis, snr_db);
                
                % DOA估计（简化版：直接用粗网格）
                if motion_speed > 0
                    % 运动阵列：使用非相干MUSIC
                    estimator = DoaEstimatorIncoherent_FIXED(platform, radar_params);
                    options.verbose = false;
                    options.weighting = 'uniform';
                    options.num_segments = 4;  % 降低分段数加速
                    spectrum = estimator.estimate_incoherent_music(snapshots, t_axis, 1, search_grid, options);
                else
                    % 静态阵列：使用传统MUSIC
                    estimator = DoaEstimator(platform, radar_params);
                    spectrum = estimator.estimate_gmusic(snapshots, t_axis, 1, search_grid);
                end
                
                % 简单峰值检测
                [max_val, max_idx] = max(spectrum(:));
                [theta_idx, phi_idx] = ind2sub(size(spectrum), max_idx);
                est_theta = search_grid.theta(theta_idx);
                est_phi = search_grid.phi(phi_idx);
                
                % 【调试】第一次试验输出估计值
                if trial == 1 && snr_idx == 1
                    fprintf('          [调试] 估计: θ=%.1f°, φ=%.1f° | 真实: θ=%.1f°, φ=%.1f°\n', ...
                        est_theta, est_phi, target_theta, target_phi);
                end
                
                % 计算误差（简单欧氏距离）
                error = sqrt((est_theta - target_theta)^2 + (est_phi - target_phi)^2);
                errors(trial) = error;
            end
            
            % 计算RMSE
            rmse = sqrt(mean(errors.^2));
            results.rmse(arr_idx, mot_idx, snr_idx) = rmse;
            
            fprintf('        SNR=%+3d dB: RMSE=%.2f° ', snr_db, rmse);
            if rmse < 1.0
                fprintf('✅\n');
            elseif rmse < 3.0
                fprintf('✓\n');
            else
                fprintf('⚠️\n');
            end
        end
        
        elapsed = toc(tic_config);
        results.computation_time(arr_idx, mot_idx) = elapsed;
        fprintf('        耗时: %.1f 秒\n\n', elapsed);
    end
end

total_time = toc(tic_total);

%% 显示结果摘要
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('快速验证完成！总耗时: %.1f 分钟\n', total_time/60);
fprintf('═══════════════════════════════════════════════════════\n\n');

%% 分析最优组合（在SNR=10dB时）
high_snr_idx = find(snr_range == 10);
if ~isempty(high_snr_idx)
    rmse_at_high_snr = results.rmse(:, :, high_snr_idx);
    [min_rmse, min_idx] = min(rmse_at_high_snr(:));
    [best_arr_idx, best_mot_idx] = ind2sub(size(rmse_at_high_snr), min_idx);
    
    fprintf('🏆 最优组合（SNR=10dB）:\n');
    fprintf('   阵列: %s\n', results.array_names{best_arr_idx});
    fprintf('   轨迹: %s\n', results.motion_names{best_mot_idx});
    fprintf('   RMSE: %.2f°\n\n', min_rmse);
end

%% 对比静止 vs 运动
fprintf('静止 vs 运动对比（SNR=10dB）:\n');
for arr_idx = 1:size(array_configs, 1)
    static_rmse = results.rmse(arr_idx, 1, high_snr_idx);  % 第1个是静止
    motion_rmse = results.rmse(arr_idx, 2:end, high_snr_idx);
    best_motion_rmse = min(motion_rmse);
    improvement = static_rmse / best_motion_rmse;
    
    fprintf('  %s: %.2f° → %.2f° (改善%.1fx)\n', ...
        array_configs{arr_idx,1}, static_rmse, best_motion_rmse, improvement);
end
fprintf('\n');

%% 保存结果
save('quick_validation_results.mat', 'results');
fprintf('✓ 结果已保存: quick_validation_results.mat\n\n');

%% 生成可视化
fprintf('生成对比图...\n');
visualize_quick_results(results);
fprintf('✓ 图片已保存\n\n');

fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║  验证完成！如果结果合理，可运行完整实验                ║\n');
fprintf('║  完整实验: comprehensive_experiment.m                  ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n');

%% ═══════════════════════════════════════════════════════════════
%  辅助函数：阵列生成
%% ═══════════════════════════════════════════════════════════════

function pos = generate_ula(N, spacing)
    % 一字型均匀线阵（沿x轴）
    % 返回: N × 3 矩阵（与 ArrayPlatform 兼容）
    x = ((0:N-1) * spacing - (N-1)*spacing/2)';
    pos = [x, zeros(N,1), zeros(N,1)];
end

function pos = generate_l_array(N, spacing)
    % L型阵列（x轴和y轴各一半）
    % 返回: N × 3 矩阵
    N_arm = ceil(N/2);
    % x轴臂
    x_arm = ((0:N_arm-1) * spacing)';
    pos_x = [x_arm, zeros(N_arm,1), zeros(N_arm,1)];
    % y轴臂（不包括原点，避免重复）
    y_arm = ((1:N-N_arm) * spacing)';
    pos_y = [zeros(N-N_arm,1), y_arm, zeros(N-N_arm,1)];
    pos = [pos_x; pos_y];
end

function pos = generate_cross_array(N, spacing)
    % 十字型阵列（x轴和y轴对称）
    % 返回: N × 3 矩阵
    N_arm = ceil(N/4);
    % +x方向
    arm_len = ((1:N_arm) * spacing)';
    pos_px = [arm_len, zeros(N_arm,1), zeros(N_arm,1)];
    % -x方向
    pos_nx = [-arm_len, zeros(N_arm,1), zeros(N_arm,1)];
    % +y方向
    N_remain = N - 3*N_arm;
    if N_remain > 0
        arm_len_y = ((1:N_remain) * spacing)';
        pos_py = [zeros(N_remain,1), arm_len_y, zeros(N_remain,1)];
    else
        pos_py = zeros(0,3);
    end
    % -y方向
    pos_ny = [zeros(N_arm,1), -arm_len, zeros(N_arm,1)];
    % 原点
    pos_0 = [0, 0, 0];
    pos = [pos_0; pos_px; pos_nx; pos_py; pos_ny];
    % 取前N个
    pos = pos(1:N, :);
end

function pos = generate_ura(N, spacing)
    % 方阵（尽量接近正方形）
    % 返回: N × 3 矩阵
    N_side = ceil(sqrt(N));
    [X, Y] = meshgrid(0:N_side-1, 0:N_side-1);
    X = X(:) * spacing - (N_side-1)*spacing/2;
    Y = Y(:) * spacing - (N_side-1)*spacing/2;
    pos = [X(1:N), Y(1:N), zeros(N,1)];
end

%% ═══════════════════════════════════════════════════════════════
%  辅助函数：运动轨迹
%% ═══════════════════════════════════════════════════════════════

function state = motion_static()
    % 静止（参考 comprehensive_validation_FIXED.m 的格式）
    state.position = [0, 0, 0];
    state.orientation = [0, 0, 0];
end

function state = motion_rotate_center(t, R, v)
    % 绕自身中心旋转（纯旋转，不平移）
    omega_dps = 360 / (2*pi*R / v);  % 度/秒
    angle_deg = omega_dps * t;
    
    state.position = [0, 0, 0];
    state.orientation = [0, 0, angle_deg];
end

function state = motion_rotate_edge(t, R, v)
    % 绕边缘旋转（圆周运动）
    r_orbit = 2*R;  % 轨道半径
    omega = v / r_orbit;
    angle = omega * t;
    
    state.position = [r_orbit*cos(angle), r_orbit*sin(angle), 0];
    state.orientation = [0, 0, 0];  % 阵列不自转
end

function state = motion_linear(t, v)
    % 直线平移（沿x轴）
    state.position = [v*t, 0, 0];
    state.orientation = [0, 0, 0];
end

function state = motion_rotate_translate(t, R, v)
    % 旋转+平移（螺旋运动）
    omega_dps = 360 / (4*pi*R / v);  % 降低旋转速度
    angle_deg = omega_dps * t;
    
    state.position = [v*t, 0, 0];  % 平移
    state.orientation = [0, 0, angle_deg];
end

%% ═══════════════════════════════════════════════════════════════
%  可视化函数
%% ═══════════════════════════════════════════════════════════════

function visualize_quick_results(results)
    figure('Position', [100, 100, 1400, 800]);
    
    % 准备数据
    n_arrays = length(results.array_names);
    n_motions = length(results.motion_names);
    n_snr = length(results.snr_range);
    
    % 子图1: RMSE vs SNR（选择最优组合）
    subplot(2,2,1);
    hold on; grid on;
    colors = lines(n_arrays);
    for arr_idx = 1:n_arrays
        % 找每个阵列的最优运动
        rmse_curve = squeeze(min(results.rmse(arr_idx, :, :), [], 2));
        plot(results.snr_range, rmse_curve, 'o-', 'LineWidth', 2, ...
            'Color', colors(arr_idx,:), 'DisplayName', results.array_names{arr_idx});
    end
    xlabel('SNR (dB)'); ylabel('RMSE (度)');
    title('最优运动下的RMSE对比');
    legend('Location', 'best');
    set(gca, 'YScale', 'log');
    
    % 子图2: 静止 vs 最优运动（SNR=10dB）
    subplot(2,2,2);
    high_snr_idx = find(results.snr_range == 10);
    if isempty(high_snr_idx), high_snr_idx = length(results.snr_range); end
    
    static_rmse = results.rmse(:, 1, high_snr_idx);
    best_motion_rmse = squeeze(min(results.rmse(:, 2:end, high_snr_idx), [], 2));
    
    x = 1:n_arrays;
    bar(x - 0.15, static_rmse, 0.3, 'FaceColor', [0.7 0.7 0.7], 'DisplayName', '静止');
    hold on;
    bar(x + 0.15, best_motion_rmse, 0.3, 'FaceColor', [0.2 0.6 0.8], 'DisplayName', '最优运动');
    set(gca, 'XTick', x, 'XTickLabel', results.array_names);
    ylabel('RMSE (度)');
    title(sprintf('静止 vs 运动 (SNR=%d dB)', results.snr_range(high_snr_idx)));
    legend();
    grid on;
    
    % 子图3: 热力图（阵列×运动，SNR=10dB）
    subplot(2,2,3);
    rmse_matrix = results.rmse(:, :, high_snr_idx);
    imagesc(rmse_matrix);
    colorbar;
    set(gca, 'XTick', 1:n_motions, 'XTickLabel', results.motion_names, 'XTickLabelRotation', 45);
    set(gca, 'YTick', 1:n_arrays, 'YTickLabel', results.array_names);
    title(sprintf('RMSE热力图 (SNR=%d dB)', results.snr_range(high_snr_idx)));
    xlabel('运动轨迹');
    ylabel('阵列配置');
    
    % 子图4: 性能改善倍数
    subplot(2,2,4);
    improvement_matrix = static_rmse ./ results.rmse(:, 2:end, high_snr_idx);
    imagesc(improvement_matrix);
    colorbar;
    set(gca, 'XTick', 1:n_motions-1, 'XTickLabel', results.motion_names(2:end), 'XTickLabelRotation', 45);
    set(gca, 'YTick', 1:n_arrays, 'YTickLabel', results.array_names);
    title('性能改善倍数 (运动/静止)');
    xlabel('运动轨迹');
    ylabel('阵列配置');
    
    saveas(gcf, 'quick_validation_comparison.png');
end

