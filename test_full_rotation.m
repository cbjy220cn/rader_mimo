%% 测试完整旋转（360度+）的合成孔径效果
clear; clc;

fprintf('=== 完整旋转合成孔径测试 ===\n\n');

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

% 目标
target_range = 20;
theta_true = 30;
phi_true = 20;
target_pos = [target_range * sind(theta_true) * cosd(phi_true), ...
              target_range * sind(theta_true) * sind(phi_true), ...
              target_range * cosd(theta_true)];
target = Target(target_pos, [0, 0, 0], 1);

fprintf('目标: theta=%.1f°, phi=%.1f°\n\n', theta_true, phi_true);

% 搜索网格
search_grid.theta = 0:0.5:90;
search_grid.phi = -90:0.5:90;

% 测试不同的完整旋转圈数
rotation_tests = [360, 720, 1080];  % 1圈、2圈、3圈

fprintf('%-15s | %-12s | %-12s | %-15s | %-15s | %-12s\n', ...
    '旋转角度', 'Theta估算', 'Phi估算', 'Theta误差', 'Phi误差', '峰值');
fprintf('%s\n', repmat('-', 1, 95));

for total_rotation = rotation_tests
    % 时间轴
    num_snapshots = 128;
    T_chirp = 1e-4;
    t_axis = (0:num_snapshots-1) * T_chirp;
    omega_dps = total_rotation / t_axis(end);
    
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
    
    fprintf('%-15.0f | %-12.1f | %-12.1f | %-15.1f | %-15.1f | %-12.2f\n', ...
        total_rotation, theta_est, phi_est, ...
        theta_est-theta_true, phi_est-phi_true, peak_val);
end

fprintf('\n分析：\n');
fprintf('如果完整旋转（≥360°）能显著改善精度，说明可以用"合成圆形阵列"方法。\n');
fprintf('如果仍然失败，说明需要使用非相干积累方法。\n');

