%% 测试导向矢量与信号的一致性
% 这个脚本验证SignalGenerator产生的信号相位
% 与DoaEstimator的导向矢量是否匹配
clear; clc;

fprintf('=== 导向矢量一致性测试 ===\n\n');

% 简化的雷达参数
radar_params.fc = 77e9;
radar_params.c = physconst('LightSpeed');
radar_params.lambda = radar_params.c / radar_params.fc;
radar_params.fs = 20e6;
radar_params.T_chirp = 50e-6;
radar_params.slope = 100e12;
radar_params.BW = radar_params.slope * radar_params.T_chirp;
radar_params.num_samples = radar_params.T_chirp * radar_params.fs;
radar_params.range_res = radar_params.c / (2 * radar_params.BW);

% 简单的ULA
num_elements = 4;
spacing = radar_params.lambda / 2;
x_coords = (0:num_elements-1) * spacing;
rect_elements = [x_coords', zeros(num_elements, 1), zeros(num_elements, 1)];

% 静态阵列（先测试静态情况）
array_platform = ArrayPlatform(rect_elements, 1, 1:num_elements);
array_platform = array_platform.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

% 简单目标
range_target = 50;  % 米
theta_target = 30;  % 度
phi_target = 0;     % 度（在xz平面内）
target_pos = [range_target * sind(theta_target) * cosd(phi_target), ...
              range_target * sind(theta_target) * sind(phi_target), ...
              range_target * cosd(theta_target)];
target = Target(target_pos, [0,0,0], 1);

fprintf('目标位置: [%.2f, %.2f, %.2f] m\n', target_pos(1), target_pos(2), target_pos(3));
fprintf('目标角度: theta=%.1f°, phi=%.1f°\n\n', theta_target, phi_target);

% 生成信号（无噪声）
sig_gen = SignalGenerator(radar_params, array_platform, {target});
t_axis = 0:1e-4:1e-3;  % 只用几个快拍
num_snapshots = length(t_axis);
snapshots = sig_gen.generate_snapshots(t_axis, inf);  % 无噪声

fprintf('生成的快拍矩阵大小: %d x %d\n', size(snapshots, 1), size(snapshots, 2));
fprintf('快拍相位 (第1个虚拟元素):\n');
for k = 1:min(5, num_snapshots)
    fprintf('  t=%.4f: 相位=%.4f rad, 幅度=%.4f\n', t_axis(k), angle(snapshots(1,k)), abs(snapshots(1,k)));
end
fprintf('\n');

% 构建理论导向矢量
fprintf('构建理论导向矢量...\n');
estimator = DoaEstimator(array_platform, radar_params);
u_target = [sind(theta_target)*cosd(phi_target); ...
            sind(theta_target)*sind(phi_target); ...
            cosd(theta_target)];
A_theory = estimator.build_steering_matrix(t_axis, u_target);

fprintf('导向矢量大小: %d x %d\n', size(A_theory, 1), size(A_theory, 2));
fprintf('导向矢量相位 (第1个虚拟元素):\n');
for k = 1:min(5, num_snapshots)
    fprintf('  t=%.4f: 相位=%.4f rad, 幅度=%.4f\n', t_axis(k), angle(A_theory(1,k)), abs(A_theory(1,k)));
end
fprintf('\n');

% 检查相位一致性
% 注意：快拍包含baseband随机相位，我们需要去除它
fprintf('=== 相位一致性检查 ===\n');
% 使用第一个快拍作为参考，计算相位差
phase_snapshots = angle(snapshots);
phase_steering = angle(A_theory);

% 计算第一个快拍的相位差（用于对齐baseband相位）
phase_offset = phase_snapshots(:, 1) - phase_steering(:, 1);
fprintf('第一个快拍的相位偏移(baseband): %.4f rad (元素1)\n', phase_offset(1));

% 检查后续快拍的相位演化是否一致
fprintf('\n相位演化对比（已移除baseband offset）:\n');
fprintf('%-10s | %-15s | %-15s | %-10s\n', '快拍', '信号相位', '导向矢量相位', '误差');
fprintf('%s\n', repmat('-', 1, 60));
for k = 1:min(10, num_snapshots)
    % 移除baseband offset
    phase_signal_corrected = phase_snapshots(1, k) - phase_offset(1);
    phase_steering_k = phase_steering(1, k);
    
    % 计算相位差（考虑2π周期性）
    phase_diff = mod(phase_signal_corrected - phase_steering_k + pi, 2*pi) - pi;
    
    fprintf('%-10d | %-15.6f | %-15.6f | %-10.6f\n', k, phase_signal_corrected, phase_steering_k, phase_diff);
end

% 计算平均误差
num_virtual_elements = size(snapshots, 1);
all_phase_diffs = zeros(num_virtual_elements, num_snapshots);
for k = 1:num_snapshots
    phase_signal_corrected = phase_snapshots(:, k) - phase_offset;
    phase_diff = mod(phase_signal_corrected - phase_steering(:, k) + pi, 2*pi) - pi;
    all_phase_diffs(:, k) = phase_diff;
end

mean_phase_error = mean(abs(all_phase_diffs(:)));
max_phase_error = max(abs(all_phase_diffs(:)));

fprintf('\n平均相位误差: %.6f rad (%.2f°)\n', mean_phase_error, rad2deg(mean_phase_error));
fprintf('最大相位误差: %.6f rad (%.2f°)\n', max_phase_error, rad2deg(max_phase_error));

if mean_phase_error < 0.01
    fprintf('\n✅ 相位一致性良好！信号和导向矢量匹配。\n');
else
    fprintf('\n❌ 相位一致性差！存在%.2f°的平均误差。\n', rad2deg(mean_phase_error));
    fprintf('   这可能导致DOA估算错误。\n');
end

