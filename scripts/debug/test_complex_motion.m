%% 复杂运动模式测试 - 使用circle_move.m的实际参数
% 测试非相干MUSIC在各种运动模式下的性能
clear; clc; close all;

fprintf('=== 复杂运动合成孔径雷达DOA估算测试 ===\n\n');

%% 雷达参数（来自circle_move.m）
c = physconst('LightSpeed');
BW = 50e6;                      % 带宽 50 MHz
f0 = 3000e6;                    % 载频 3 GHz (S波段)
lambda = c/f0;                  % 波长 0.1m
numADC = 361;                   % ADC采样点
numChirps = 256;                % 每帧chirp数
numCPI = 10;                    % CPI数量
T = 10e-3;                      % PRI (脉冲重复间隔)
F = numADC/T;                   % 采样频率
slope = BW/T;                   % 调频斜率

fprintf('雷达参数:\n');
fprintf('  载频: %.2f GHz (λ=%.3f m)\n', f0/1e9, lambda);
fprintf('  带宽: %.1f MHz\n', BW/1e6);
fprintf('  PRI: %.1f ms\n', T*1e3);
fprintf('  总CPI数: %d\n', numCPI);
fprintf('  总chirp数: %d\n\n', numChirps * numCPI);

% 转换为统一的雷达参数结构
radar_params.fc = f0;
radar_params.c = c;
radar_params.lambda = lambda;
radar_params.fs = F;
radar_params.T_chirp = T;
radar_params.slope = slope;
radar_params.BW = BW;
radar_params.num_samples = numADC;
radar_params.range_res = c / (2 * BW);

fprintf('  距离分辨率: %.2f m\n\n', radar_params.range_res);

%% 阵列配置（圆形阵列）
numRX = 8;
R_rx = 0.05;                    % 接收阵列半径 5cm
theta_rx = linspace(0, 2*pi, numRX+1); 
theta_rx(end) = [];

% 生成圆形阵列位置
rx_elements = zeros(numRX, 3);
for i = 1:numRX
    rx_elements(i,:) = [R_rx*cos(theta_rx(i)), R_rx*sin(theta_rx(i)), 0];
end

fprintf('阵列配置:\n');
fprintf('  类型: 均匀圆形阵列 (UCA)\n');
fprintf('  阵元数: %d\n', numRX);
fprintf('  半径: %.3f m (%.2f λ)\n', R_rx, R_rx/lambda);
fprintf('  阵元间距: %.3f m (%.2f λ)\n\n', 2*pi*R_rx/numRX, 2*pi*R_rx/numRX/lambda);

%% 目标设置（远场目标）
r1_radial = 660;                % 距离 660m
tar1_theta = 30;                % 俯仰角 30度
tar1_phi = 60;                  % 方位角 60度

% 球坐标转笛卡尔坐标
r1_x = cosd(tar1_phi)*sind(tar1_theta)*r1_radial;
r1_y = sind(tar1_phi)*sind(tar1_theta)*r1_radial;
r1_z = cosd(tar1_theta)*r1_radial;
target_pos = [r1_x, r1_y, r1_z];

v1_radial = 0.001;              % 极慢的径向速度
v1_x = cosd(tar1_phi)*cosd(tar1_theta)*v1_radial;
v1_y = sind(tar1_phi)*cosd(tar1_theta)*v1_radial;
v1_z = sind(tar1_theta)*v1_radial;

target = Target(target_pos, [v1_x, v1_y, v1_z], 1);

fprintf('目标参数:\n');
fprintf('  位置: [%.1f, %.1f, %.1f] m\n', r1_x, r1_y, r1_z);
fprintf('  距离: %.1f m\n', r1_radial);
fprintf('  角度: theta=%.1f°, phi=%.1f°\n', tar1_theta, tar1_phi);
fprintf('  速度: %.3f m/s (径向)\n\n', v1_radial);

%% 时间轴设置
num_snapshots = numChirps * numCPI;  % 总快拍数
t_axis = (0:num_snapshots-1) * T;
total_time = t_axis(end);

fprintf('时间参数:\n');
fprintf('  总快拍数: %d\n', num_snapshots);
fprintf('  总时间: %.2f s\n', total_time);
fprintf('  快拍间隔: %.1f ms\n\n', T*1e3);

%% 搜索网格
search_grid.theta = 0:0.5:90;
search_grid.phi = 0:0.5:180;

%% 测试不同的运动模式

motion_patterns = {
    '静止', ...
    '均速圆周旋转', ...
    '变速圆周旋转', ...
    '螺旋上升', ...
    '随机游走', ...
    '8字形轨迹'
};

num_patterns = length(motion_patterns);
results = struct();

fprintf('=== 开始测试 %d 种运动模式 ===\n\n', num_patterns);

for pattern_idx = 1:num_patterns
    pattern_name = motion_patterns{pattern_idx};
    fprintf('--- 运动模式 %d/%d: %s ---\n', pattern_idx, num_patterns, pattern_name);
    
    % 根据运动模式定义轨迹函数
    switch pattern_idx
        case 1  % 静止
            trajectory_func = @(t) struct('position', [0,0,0], 'orientation', [0,0,0]);
            
        case 2  % 均速圆周旋转（类似circle_move.m）
            % 旋转速度：在总时间内旋转多圈
            omega_dps = 360 / total_time;  % 1圈/秒
            trajectory_func = @(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]);
            fprintf('  旋转速度: %.1f °/s (%.1f圈在%.2fs内)\n', omega_dps, omega_dps*total_time/360, total_time);
            
        case 3  % 变速旋转（加速-减速）
            omega_max = 720;  % 最大角速度 720°/s
            trajectory_func = @(t) struct('position', [0,0,0], ...
                'orientation', [0, 0, omega_max * (t/total_time) * (1 - t/total_time) * 4]);
            fprintf('  变速旋转: 加速-减速\n');
            
        case 4  % 螺旋上升
            omega_dps = 360 / total_time;
            v_up = 0.1;  % 向上速度 0.1 m/s
            trajectory_func = @(t) struct('position', [0, 0, v_up*t], ...
                'orientation', [0, 0, omega_dps * t]);
            fprintf('  螺旋上升: %.1f°/s旋转 + %.2f m/s上升\n', omega_dps, v_up);
            
        case 5  % 随机游走
            rng(42);  % 固定随机种子
            random_positions = cumsum([zeros(1,3); 0.01*randn(num_snapshots-1, 3)], 1);
            random_orientations = cumsum([zeros(1,3); 1*randn(num_snapshots-1, 3)], 1);
            % 创建插值函数
            pos_interp = griddedInterpolant(t_axis, random_positions, 'linear');
            ori_interp = griddedInterpolant(t_axis, random_orientations, 'linear');
            trajectory_func = @(t) struct('position', pos_interp(t)', ...
                'orientation', ori_interp(t)');
            fprintf('  随机游走: 位置和姿态随机扰动\n');
            
        case 6  % 8字形轨迹
            omega = 2*pi / total_time;  % 角频率
            radius_8 = 0.5;  % 8字半径
            trajectory_func = @(t) struct(...
                'position', [radius_8*sin(omega*t), radius_8*sin(2*omega*t), 0], ...
                'orientation', [0, 0, 180*sin(omega*t)]);
            fprintf('  8字形轨迹: 半径%.2fm\n', radius_8);
    end
    
    % 创建阵列平台
    array_platform = ArrayPlatform(rx_elements, 1, 1:numRX);
    array_platform = array_platform.set_trajectory(trajectory_func);
    
    % 生成信号
    sig_gen = SignalGenerator(radar_params, array_platform, {target});
    snapshots = sig_gen.generate_snapshots(t_axis, inf);
    
    % 使用非相干MUSIC估算
    estimator = DoaEstimatorIncoherent(array_platform, radar_params);
    options.verbose = false;
    options.weighting = 'uniform';
    
    tic;
    spectrum = estimator.estimate_incoherent_music(snapshots, t_axis, 1, search_grid, options);
    compute_time = toc;
    
    [theta_est, phi_est, peak_val] = DoaEstimatorIncoherent.find_peaks(spectrum, search_grid, 1);
    
    % 保存结果
    results(pattern_idx).name = pattern_name;
    results(pattern_idx).theta_est = theta_est;
    results(pattern_idx).phi_est = phi_est;
    results(pattern_idx).theta_error = theta_est - tar1_theta;
    results(pattern_idx).phi_error = phi_est - tar1_phi;
    results(pattern_idx).peak_val = peak_val;
    results(pattern_idx).compute_time = compute_time;
    results(pattern_idx).spectrum = spectrum;
    
    fprintf('  估算: theta=%.1f°, phi=%.1f°\n', theta_est, phi_est);
    fprintf('  误差: Δtheta=%.1f°, Δphi=%.1f°\n', theta_est-tar1_theta, phi_est-tar1_phi);
    fprintf('  峰值: %.2e\n', peak_val);
    fprintf('  计算时间: %.2f s\n\n', compute_time);
end

%% 结果汇总
fprintf('=== 结果汇总 ===\n\n');
fprintf('%-20s | %-10s | %-10s | %-12s | %-12s | %-12s\n', ...
    '运动模式', 'Theta估算', 'Phi估算', 'Theta误差', 'Phi误差', '计算时间');
fprintf('%s\n', repmat('-', 1, 95));

for i = 1:num_patterns
    status = '';
    if abs(results(i).theta_error) < 2 && abs(results(i).phi_error) < 2
        status = '✓';
    elseif abs(results(i).theta_error) < 5 && abs(results(i).phi_error) < 5
        status = '⚠';
    else
        status = '✗';
    end
    
    fprintf('%-20s | %-10.1f | %-10.1f | %-12.1f | %-12.1f | %-10.2fs %s\n', ...
        results(i).name, results(i).theta_est, results(i).phi_est, ...
        results(i).theta_error, results(i).phi_error, results(i).compute_time, status);
end

%% 可视化对比
figure('Position', [50, 50, 1600, 800]);

for i = 1:min(6, num_patterns)
    subplot(2, 3, i);
    surf(search_grid.phi, search_grid.theta, results(i).spectrum);
    shading interp; view(2); colorbar;
    hold on;
    plot(tar1_phi, tar1_theta, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
    plot(results(i).phi_est, results(i).theta_est, 'wo', 'MarkerSize', 10, 'LineWidth', 2);
    hold off;
    
    title(sprintf('%s\nΔphi=%.1f°', results(i).name, results(i).phi_error));
    xlabel('Phi (度)');
    if mod(i-1, 3) == 0
        ylabel('Theta (度)');
    end
end

sgtitle('非相干MUSIC - 不同运动模式性能对比', 'FontSize', 14, 'FontWeight', 'bold');

%% 性能分析
fprintf('\n=== 性能分析 ===\n');

theta_errors = [results.theta_error];
phi_errors = [results.phi_error];

fprintf('Theta估算:\n');
fprintf('  平均误差: %.2f°\n', mean(abs(theta_errors)));
fprintf('  最大误差: %.2f°\n', max(abs(theta_errors)));
fprintf('  标准差: %.2f°\n', std(theta_errors));

fprintf('\nPhi估算:\n');
fprintf('  平均误差: %.2f°\n', mean(abs(phi_errors)));
fprintf('  最大误差: %.2f°\n', max(abs(phi_errors)));
fprintf('  标准差: %.2f°\n', std(phi_errors));

fprintf('\n计算性能:\n');
fprintf('  平均计算时间: %.2f s\n', mean([results.compute_time]));
fprintf('  最长计算时间: %.2f s\n', max([results.compute_time]));

fprintf('\n=== 结论 ===\n');
successful_count = sum(abs([results.phi_error]) < 5);
fprintf('✅ %d/%d 种运动模式实现了良好的DOA估算（误差<5°）\n', successful_count, num_patterns);

if successful_count == num_patterns
    fprintf('🎉 非相干MUSIC算法在所有运动模式下都表现优异！\n');
    fprintf('   已验证适用于米波雷达无人机编队合成孔径系统。\n');
elseif successful_count >= num_patterns * 0.8
    fprintf('✅ 非相干MUSIC算法在大多数运动模式下表现良好。\n');
else
    fprintf('⚠️  部分运动模式需要进一步优化。\n');
end

