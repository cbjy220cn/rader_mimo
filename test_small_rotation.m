%% 测试小角度旋转和静止目标
clear; clc;

fprintf('=== 小角度旋转测试（静止目标）===\n\n');

% 雷达参数
radar_params.fc = 77e9;
radar_params.c = physconst('LightSpeed');
radar_params.lambda = radar_params.c / radar_params.fc;
radar_params.fs = 20e6;
radar_params.T_chirp = 50e-6;
radar_params.slope = 100e12;
radar_params.BW = radar_params.slope * radar_params.T_chirp;
radar_params.num_samples = radar_params.T_chirp * radar_params.fs;
radar_params.range_res = radar_params.c / (2 * radar_params.BW);

% 4x2 URA
num_elements = 8;
spacing = radar_params.lambda / 2;
Nx = 4;
Ny = 2;
x_coords = (-(Nx-1)/2:(Nx-1)/2) * spacing;
y_coords = (-(Ny-1)/2:(Ny-1)/2) * spacing;
[X, Y] = meshgrid(x_coords, y_coords);
rect_elements = [X(:), Y(:), zeros(num_elements, 1)];

% 静止目标
target_range = 20;
theta_true = 30;
phi_true = 20;
target_pos = [target_range * sind(theta_true) * cosd(phi_true), ...
              target_range * sind(theta_true) * sind(phi_true), ...
              target_range * cosd(theta_true)];
target = Target(target_pos, [0, 0, 0], 1);  % ← 零速度！

fprintf('目标: [%.2f, %.2f, %.2f] m\n', target_pos(1), target_pos(2), target_pos(3));
fprintf('真实角度: theta=%.1f°, phi=%.1f°\n', theta_true, phi_true);
fprintf('目标速度: [0, 0, 0] m/s (静止)\n\n');

% 搜索网格
search_grid.theta = 0:0.5:90;
search_grid.phi = -90:0.5:90;

% 测试不同的旋转角度
rotation_angles = [0, 10, 30, 60, 90, 180];
results = struct();

for idx = 1:length(rotation_angles)
    total_rotation = rotation_angles(idx);
    fprintf('--- 测试：旋转%.0f度 ---\n', total_rotation);
    
    % 时间轴
    num_snapshots = 64;
    T_chirp = 1e-4;
    t_axis = (0:num_snapshots-1) * T_chirp;
    
    if total_rotation == 0
        % 静态
        omega_dps = 0;
    else
        omega_dps = total_rotation / t_axis(end);
    end
    
    % 创建旋转阵列
    array_rotating = ArrayPlatform(rect_elements, 1, 1:num_elements);
    rotation_traj = @(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]);
    array_rotating = array_rotating.set_trajectory(rotation_traj);
    
    % 生成信号和估算
    sig_gen = SignalGenerator(radar_params, array_rotating, {target});
    snapshots = sig_gen.generate_snapshots(t_axis, inf);
    
    estimator = DoaEstimator(array_rotating, radar_params);
    spectrum = estimator.estimate_gmusic(snapshots, t_axis, 1, search_grid);
    [theta_est, phi_est, peak_val] = DoaEstimator.find_peaks(spectrum, search_grid, 1);
    
    % 保存结果
    results(idx).rotation = total_rotation;
    results(idx).theta_est = theta_est;
    results(idx).phi_est = phi_est;
    results(idx).theta_error = theta_est - theta_true;
    results(idx).phi_error = phi_est - phi_true;
    results(idx).peak_val = peak_val;
    
    fprintf('估算: theta=%.1f°, phi=%.1f°\n', theta_est, phi_est);
    fprintf('误差: Δtheta=%.1f°, Δphi=%.1f°\n', theta_est-theta_true, phi_est-phi_true);
    fprintf('峰值: %.2f\n\n', peak_val);
end

% 结果对比表格
fprintf('=== 旋转角度 vs 估算精度 ===\n');
fprintf('%-15s | %-12s | %-12s | %-15s | %-15s | %-12s\n', ...
    '旋转角度', 'Theta估算', 'Phi估算', 'Theta误差', 'Phi误差', '峰值');
fprintf('%s\n', repmat('-', 1, 95));
for idx = 1:length(results)
    fprintf('%-15.0f | %-12.1f | %-12.1f | %-15.1f | %-15.1f | %-12.2f\n', ...
        results(idx).rotation, results(idx).theta_est, results(idx).phi_est, ...
        results(idx).theta_error, results(idx).phi_error, results(idx).peak_val);
end

% 找出phi误差何时开始变大
fprintf('\n分析：\n');
for idx = 1:length(results)
    if abs(results(idx).phi_error) > 10
        fprintf('❌ 旋转%.0f度时phi误差达到%.1f°，超过阈值\n', ...
            results(idx).rotation, results(idx).phi_error);
    else
        fprintf('✓ 旋转%.0f度时phi误差为%.1f°，可接受\n', ...
            results(idx).rotation, results(idx).phi_error);
    end
end

