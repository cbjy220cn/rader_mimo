%% 深度调试：对比旋转阵列的信号相位和导向矢量相位
clear; clc;

fprintf('=== 旋转阵列相位深度调试 ===\n\n');

% 使用简化参数以便理解
radar_params.fc = 77e9;
radar_params.c = physconst('LightSpeed');
radar_params.lambda = radar_params.c / radar_params.fc;
radar_params.fs = 20e6;
radar_params.T_chirp = 50e-6;
radar_params.slope = 100e12;
radar_params.BW = radar_params.slope * radar_params.T_chirp;
radar_params.num_samples = radar_params.T_chirp * radar_params.fs;
radar_params.range_res = radar_params.c / (2 * radar_params.BW);

% 简单的ULA（只沿x轴，便于理解）
num_elements = 4;
spacing = radar_params.lambda / 2;
x_coords = (-(num_elements-1)/2:(num_elements-1)/2) * spacing;
elements = [x_coords', zeros(num_elements, 1), zeros(num_elements, 1)];

% 目标
target_range = 20;
theta_true = 30;
phi_true = 20;
target_pos = [target_range * sind(theta_true) * cosd(phi_true), ...
              target_range * sind(theta_true) * sind(phi_true), ...
              target_range * cosd(theta_true)];
target = Target(target_pos, [0,0,0], 1);

fprintf('目标: [%.2f, %.2f, %.2f] m, theta=%.1f°, phi=%.1f°\n\n', ...
    target_pos(1), target_pos(2), target_pos(3), theta_true, phi_true);

% 只用几个快拍
num_snapshots = 5;
total_rotation = 45;  % 旋转45度（适中的旋转）
T_chirp = 1e-4;
t_axis = (0:num_snapshots-1) * T_chirp;
omega_dps = total_rotation / t_axis(end);

fprintf('旋转: %d快拍, %.1f°总旋转, %.1f°/s\n', num_snapshots, total_rotation, omega_dps);
fprintf('时间: ');
for k = 1:num_snapshots
    fprintf('t%d=%.4fs(%.1f°) ', k, t_axis(k), omega_dps*t_axis(k));
end
fprintf('\n\n');

% 创建旋转阵列
array_rotating = ArrayPlatform(elements, 1, 1:num_elements);
rotation_traj = @(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]);
array_rotating = array_rotating.set_trajectory(rotation_traj);

% 生成信号（无噪声）
sig_gen = SignalGenerator(radar_params, array_rotating, {target});
snapshots = sig_gen.generate_snapshots(t_axis, inf);

fprintf('生成的快拍矩阵: %d阵元 x %d快拍\n\n', size(snapshots, 1), size(snapshots, 2));

% 构建理论导向矢量
estimator = DoaEstimator(array_rotating, radar_params);
u_target = [sind(theta_true)*cosd(phi_true); ...
            sind(theta_true)*sind(phi_true); ...
            cosd(theta_true)];
A_theory = estimator.build_steering_matrix(t_axis, u_target);

fprintf('导向矢量: %d阵元 x %d快拍\n\n', size(A_theory, 1), size(A_theory, 2));

% === 详细对比每个阵元、每个快拍的相位 ===
fprintf('=== 逐元素相位对比 ===\n');
fprintf('(信号包含随机baseband相位，需要归一化)\n\n');

% 计算baseband相位偏移（使用第一个快拍的第一个元素）
phase_offset = angle(snapshots(1, 1)) - angle(A_theory(1, 1));
fprintf('Baseband相位偏移: %.4f rad\n\n', phase_offset);

% 创建对比表格
fprintf('%-8s | %-8s | %-15s | %-15s | %-15s | %-10s\n', ...
    '快拍', '阵元', '信号相位', '导向矢量相位', '信号(归一化)', '相位差');
fprintf('%s\n', repmat('-', 1, 100));

max_phase_error = 0;
for k = 1:num_snapshots
    for m = 1:min(num_elements, 4)  % 只显示前4个阵元
        phase_signal = angle(snapshots(m, k));
        phase_steering = angle(A_theory(m, k));
        phase_signal_normalized = phase_signal - phase_offset;
        phase_diff = mod(phase_signal_normalized - phase_steering + pi, 2*pi) - pi;
        max_phase_error = max(max_phase_error, abs(phase_diff));
        
        fprintf('%-8d | %-8d | %-15.6f | %-15.6f | %-15.6f | %-10.6f\n', ...
            k, m, phase_signal, phase_steering, phase_signal_normalized, phase_diff);
    end
    if k < num_snapshots
        fprintf('%s\n', repmat('-', 1, 100));
    end
end

fprintf('\n最大相位误差: %.6f rad (%.2f°)\n', max_phase_error, rad2deg(max_phase_error));

if max_phase_error < 0.1
    fprintf('\n✅ 相位匹配良好！\n');
else
    fprintf('\n❌ 相位存在显著误差！\n');
    fprintf('   这解释了为什么MUSIC谱不收敛。\n');
end

% === 打印虚拟阵元位置信息 ===
fprintf('\n=== 虚拟阵元位置随时间变化 ===\n');
for k = 1:num_snapshots
    virtual_pos = array_rotating.get_mimo_virtual_positions(t_axis(k));
    fprintf('t=%d (yaw=%.1f°):\n', k, omega_dps*t_axis(k));
    for m = 1:min(size(virtual_pos, 1), 4)
        fprintf('  元素%d: [%.6f, %.6f, %.6f]\n', m, virtual_pos(m,1), virtual_pos(m,2), virtual_pos(m,3));
    end
end

% === 计算相位演化率 ===
fprintf('\n=== 相位随时间演化 ===\n');
fprintf('(观察第1个阵元的相位变化)\n\n');

phase_signal_elem1 = angle(snapshots(1, :)) - phase_offset;
phase_steering_elem1 = angle(A_theory(1, :));

fprintf('%-8s | %-15s | %-15s | %-15s\n', '快拍', '信号相位', '导向矢量相位', '差值');
fprintf('%s\n', repmat('-', 1, 60));
for k = 1:num_snapshots
    phase_diff = mod(phase_signal_elem1(k) - phase_steering_elem1(k) + pi, 2*pi) - pi;
    fprintf('%-8d | %-15.6f | %-15.6f | %-15.6f\n', ...
        k, phase_signal_elem1(k), phase_steering_elem1(k), phase_diff);
end

% 计算相位变化率
if num_snapshots > 1
    dphi_signal = unwrap(phase_signal_elem1);
    dphi_steering = unwrap(phase_steering_elem1);
    
    rate_signal = (dphi_signal(end) - dphi_signal(1)) / t_axis(end);
    rate_steering = (dphi_steering(end) - dphi_steering(1)) / t_axis(end);
    
    fprintf('\n相位变化率（元素1）:\n');
    fprintf('  信号: %.2f rad/s\n', rate_signal);
    fprintf('  导向矢量: %.2f rad/s\n', rate_steering);
    fprintf('  差异: %.2f rad/s (%.1f%%)\n', rate_signal - rate_steering, ...
        100*abs(rate_signal - rate_steering)/abs(rate_steering));
end

