%% 复杂运动模式快速测试 - 减少快拍数以加快测试
% 使用circle_move.m的参数，但快拍数减少10倍以加快验证
clear; clc; close all;

fprintf('=== 复杂运动合成孔径雷达DOA估算测试（快速版）===\n\n');

%% 雷达参数
c = physconst('LightSpeed');
BW = 50e6;
f0 = 3000e6;
lambda = c/f0;
T = 10e-3;

radar_params.fc = f0;
radar_params.c = c;
radar_params.lambda = lambda;
radar_params.fs = 36100;  % 简化
radar_params.T_chirp = T;
radar_params.slope = BW/T;
radar_params.BW = BW;
radar_params.num_samples = 361;
radar_params.range_res = c / (2 * BW);

fprintf('雷达参数: f0=%.2f GHz, λ=%.3f m, 距离分辨率=%.2f m\n\n', f0/1e9, lambda, radar_params.range_res);

%% 阵列配置（圆形阵列）
numRX = 8;
R_rx = 0.05;
theta_rx = linspace(0, 2*pi, numRX+1); 
theta_rx(end) = [];

rx_elements = zeros(numRX, 3);
for i = 1:numRX
    rx_elements(i,:) = [R_rx*cos(theta_rx(i)), R_rx*sin(theta_rx(i)), 0];
end

fprintf('阵列: 8元圆形阵列，半径=%.3f m\n\n', R_rx);

%% 目标设置
r1_radial = 660;
tar1_theta = 30;
tar1_phi = 60;

r1_x = cosd(tar1_phi)*sind(tar1_theta)*r1_radial;
r1_y = sind(tar1_phi)*sind(tar1_theta)*r1_radial;
r1_z = cosd(tar1_theta)*r1_radial;
target_pos = [r1_x, r1_y, r1_z];

target = Target(target_pos, [0, 0, 0], 1);  % 静止目标简化

fprintf('目标: 距离=%.1f m, theta=%.1f°, phi=%.1f°\n\n', r1_radial, tar1_theta, tar1_phi);

%% 时间轴设置（减少快拍数）
num_snapshots = 256;  % 从2560减少到256（减少10倍）
t_axis = (0:num_snapshots-1) * T;
total_time = t_axis(end);

fprintf('⚡ 快速模式: %d个快拍 (总时间=%.2f s)\n\n', num_snapshots, total_time);

%% 搜索网格
search_grid.theta = 0:1:90;  % 粗网格以加快速度
search_grid.phi = 0:1:180;

%% 测试不同的运动模式
motion_patterns = {
    '静止', ...
    '均速旋转(1圈)', ...
    '快速旋转(2圈)', ...
    '螺旋上升'
};

num_patterns = length(motion_patterns);

fprintf('=== 测试 %d 种运动模式 ===\n\n', num_patterns);

for pattern_idx = 1:num_patterns
    pattern_name = motion_patterns{pattern_idx};
    fprintf('【%d/%d】 %s\n', pattern_idx, num_patterns, pattern_name);
    
    % 根据运动模式定义轨迹
    switch pattern_idx
        case 1  % 静止
            trajectory_func = @(t) struct('position', [0,0,0], 'orientation', [0,0,0]);
            
        case 2  % 均速旋转
            omega_dps = 360 / total_time;
            trajectory_func = @(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]);
            fprintf('  旋转: %.1f°/s (%.1f圈)\n', omega_dps, total_time*omega_dps/360);
            
        case 3  % 快速旋转
            omega_dps = 720 / total_time;
            trajectory_func = @(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]);
            fprintf('  旋转: %.1f°/s (%.1f圈)\n', omega_dps, total_time*omega_dps/360);
            
        case 4  % 螺旋上升
            omega_dps = 360 / total_time;
            v_up = 0.1;
            trajectory_func = @(t) struct('position', [0, 0, v_up*t], ...
                'orientation', [0, 0, omega_dps * t]);
            fprintf('  螺旋: %.1f°/s + %.2f m/s上升\n', omega_dps, v_up);
    end
    
    % 创建阵列
    array_platform = ArrayPlatform(rx_elements, 1, 1:numRX);
    array_platform = array_platform.set_trajectory(trajectory_func);
    
    % 生成信号
    fprintf('  [1/3] 生成信号...\n');
    sig_gen = SignalGenerator(radar_params, array_platform, {target});
    snapshots = sig_gen.generate_snapshots(t_axis, inf);
    
    % DOA估算
    fprintf('  [2/3] DOA估算（非相干MUSIC）...\n');
    estimator = DoaEstimatorIncoherent(array_platform, radar_params);
    options.verbose = false;
    options.weighting = 'uniform';
    
    tic;
    spectrum = estimator.estimate_incoherent_music(snapshots, t_axis, 1, search_grid, options);
    compute_time = toc;
    
    fprintf('  [3/3] 峰值搜索...\n');
    [theta_est, phi_est, peak_val] = DoaEstimatorIncoherent.find_peaks(spectrum, search_grid, 1);
    
    % 显示结果
    fprintf('  ✓ 估算: theta=%.1f° (误差%.1f°), phi=%.1f° (误差%.1f°)\n', ...
        theta_est, theta_est-tar1_theta, phi_est, phi_est-tar1_phi);
    fprintf('  ✓ 计算时间: %.2f s\n', compute_time);
    
    if abs(theta_est-tar1_theta) < 3 && abs(phi_est-tar1_phi) < 3
        fprintf('  ✅ 精度优秀\n\n');
    elseif abs(theta_est-tar1_theta) < 5 && abs(phi_est-tar1_phi) < 5
        fprintf('  ✓ 精度良好\n\n');
    else
        fprintf('  ⚠️  精度需改善\n\n');
    end
end

fprintf('=== 快速测试完成 ===\n');
fprintf('如果结果良好，可以运行完整版 test_complex_motion.m\n');

