function fig_handle = run_rotation_vs_circular_test(radar_params, num_elements, targets, search_grid)
%RUN_ROTATION_VS_CIRCULAR_TEST 对比不同阵列类型的分辨率。
%
%   对比一个静态圆形阵列、一个旋转矩形阵列以及一个静态矩形阵列
%   在分辨两个临近目标时的性能。
%
%   输入:
%       radar_params   - 包含雷达参数的结构体。
%       num_elements   - 所有阵列的阵元数量。
%       targets        - 包含两个目标对象的元胞数组。
%       search_grid    - 定义了theta和phi搜索空间的结构体。
%
%   输出:
%       fig_handle     - 指向包含对比图窗的句柄。

fprintf('--- 开始旋转阵列 vs 圆形阵列分辨率对比测试 ---\n\n');

lambda = physconst('LightSpeed') / radar_params.fc;
spacing = lambda / 2;

% --- 仿真时间与旋转参数设置 ---
num_snapshots = 128;
T_chirp = 1e-4; % 假设的快拍间隔，与FMCW的chirp周期相关
t_axis = (0:num_snapshots-1) * T_chirp;
total_time = t_axis(end);
total_rotation_angle_deg = 180; % 总共旋转10度
omega_dps = total_rotation_angle_deg / total_time; % 旋转角速度 (度/秒)

% --- 1. 静态圆形阵列定义 ---
fprintf('1. 评估静态圆形阵列...\n');
circumference = num_elements * spacing;
radius = circumference / (2 * pi);
angles = (0:num_elements-1).' * (2*pi/num_elements);
circ_elements = [radius * cos(angles), radius * sin(angles), zeros(num_elements, 1)];
array_circ_static = ArrayPlatform(circ_elements, 1, 1:num_elements);
array_circ_static = array_circ_static.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

sig_gen_circ = SignalGenerator(radar_params, array_circ_static, targets);
snapshots_circ = sig_gen_circ.generate_snapshots(t_axis, inf);
estimator_circ = DoaEstimator(array_circ_static, radar_params);
num_targets = length(targets); % 根据实际目标数量动态设置
spectrum_circ = estimator_circ.estimate_gmusic(snapshots_circ, t_axis, num_targets, search_grid);

% --- 定量结果分析 ---
fprintf('静态圆形阵列估算结果:\n');
[theta_peaks_circ, phi_peaks_circ, peak_vals_circ] = DoaEstimator.find_peaks(spectrum_circ, search_grid, num_targets);
for i = 1:length(theta_peaks_circ)
    fprintf('  目标 %d: Theta = %.2f 度, Phi = %.2f 度 (谱峰值: %.2f)\n', i, theta_peaks_circ(i), phi_peaks_circ(i), peak_vals_circ(i));
end
fprintf('\n');


% --- 2. 旋转矩形阵列定义 (原为线性阵列) ---
fprintf('2. 评估旋转矩形阵列（使用非相干MUSIC）...\n');
% 定义一个 4x2 URA 来消除物理模糊性
Nx = 4;
Ny = 2;
assert(Nx * Ny == num_elements, '为URA指定的阵元数与总阵元数不匹配。');
x_coords = (-(Nx-1)/2:(Nx-1)/2) * spacing;
y_coords = (-(Ny-1)/2:(Ny-1)/2) * spacing;
[X, Y] = meshgrid(x_coords, y_coords);
rect_elements = [X(:), Y(:), zeros(num_elements, 1)];

array_rect_rotating = ArrayPlatform(rect_elements, 1, 1:num_elements);
rotation_trajectory = @(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]);
array_rect_rotating = array_rect_rotating.set_trajectory(rotation_trajectory);

sig_gen_rot = SignalGenerator(radar_params, array_rect_rotating, targets);
snapshots_rot = sig_gen_rot.generate_snapshots(t_axis, inf);

% 使用非相干MUSIC（适合旋转阵列）
estimator_rot = DoaEstimatorIncoherent(array_rect_rotating, radar_params);
options_rot.verbose = false;
options_rot.weighting = 'uniform';
spectrum_rot = estimator_rot.estimate_incoherent_music(snapshots_rot, t_axis, num_targets, search_grid, options_rot);

% --- 定量结果分析 ---
fprintf('旋转矩形阵列估算结果（非相干MUSIC）:\n');
[theta_peaks_rot, phi_peaks_rot, peak_vals_rot] = DoaEstimatorIncoherent.find_peaks(spectrum_rot, search_grid, num_targets);
for i = 1:length(theta_peaks_rot)
    fprintf('  目标 %d: Theta = %.2f 度, Phi = %.2f 度 (谱峰值: %.2f)\n', i, theta_peaks_rot(i), phi_peaks_rot(i), peak_vals_rot(i));
end
fprintf('\n');


% --- [虚拟阵元排序调试] ---
fprintf('\n--- [虚拟阵元排序验证] ---\n');
fprintf('在t=0时刻，虚拟阵元位置：\n');
virtual_pos_t0 = array_rect_rotating.get_mimo_virtual_positions(t_axis(1));
for i = 1:min(4, size(virtual_pos_t0, 1))  % 只打印前4个
    fprintf('  虚拟元素%d: [%.6f, %.6f, %.6f]\n', i, virtual_pos_t0(i,1), virtual_pos_t0(i,2), virtual_pos_t0(i,3));
end
fprintf('(共%d个虚拟元素)\n\n', size(virtual_pos_t0, 1));

% --- [深度调试] 模块 ---
fprintf('\n--- [深度调试] 开始旋转阵列内部变量验证 ---\n');
% 为简化，仅使用一个目标
target_debug = targets{1}; 
sig_gen_debug = SignalGenerator(radar_params, array_rect_rotating, {target_debug});
% 以调试模式运行，获取所有中间变量
[~, debug_info] = sig_gen_debug.generate_snapshots(t_axis, inf, true);

% --- 理论计算 ---
fprintf('\n--- [深度调试] 阶段 1: 理论值计算 ---\n');
% 这是之前错误的根源：一个硬编码的、错误的目标位置。我们现在删除它。
% target_pos_debug = [20, 20, 20]; 

% 正确的做法是直接从传入的targets对象中获取位置
target_pos_t0 = targets{1}.get_position_at(t_axis(1));
target_pos_t_end = targets{1}.get_position_at(t_axis(end));

% 获取轨迹函数
trajectory_func = array_rect_rotating.get_trajectory();

% t=0 时刻的理论值
platform_state_t0 = trajectory_func(t_axis(1));
[tx_pos_t0_theory, rx_pos_t0_theory] = get_theory_tx_rx_pos(rect_elements, platform_state_t0);
range_t0_theory = norm(tx_pos_t0_theory - target_pos_t0) + norm(rx_pos_t0_theory - target_pos_t0);
tau_t0_actual = range_t0_theory / radar_params.c;  % 实际信号的tau
geom_phase_t0_theory = -2 * pi * radar_params.fc * tau_t0_actual;
rvp_phase_t0_actual = pi * radar_params.slope * tau_t0_actual^2;

% --- 关键修正：计算range bin对应的tau（用于RVP校正） ---
range_one_way = norm(target_pos_t0 - platform_state_t0.position);  % 单程距离
range_bin = round(range_one_way / radar_params.range_res) + 1;
% range bin对应的频率和tau
freq_bin = (range_bin - 1) * radar_params.fs / radar_params.num_samples;
tau_bin = freq_bin / radar_params.slope;  % range bin的理论tau
rvp_phase_t0_correction = pi * radar_params.slope * tau_bin^2;  % RVP校正项

% 最终相位 = start_phase（含actual tau）- RVP校正（含bin tau）+ baseband
final_phase_t0_theory = geom_phase_t0_theory + rvp_phase_t0_actual - rvp_phase_t0_correction + debug_info.t0.baseband_phase_component;
rvp_phase_t0_theory = rvp_phase_t0_actual;  % 保持输出变量名不变

% t=end 时刻的理论值
platform_state_t_end = trajectory_func(t_axis(end));
[tx_pos_t_end_theory, rx_pos_t_end_theory] = get_theory_tx_rx_pos(rect_elements, platform_state_t_end);
range_t_end_theory = norm(tx_pos_t_end_theory - target_pos_t_end) + norm(rx_pos_t_end_theory - target_pos_t_end);
tau_t_end_actual = range_t_end_theory / radar_params.c;
geom_phase_t_end_theory = -2 * pi * radar_params.fc * tau_t_end_actual;
rvp_phase_t_end_actual = pi * radar_params.slope * tau_t_end_actual^2;

% RVP校正使用相同的range bin（目标位置在CPI期间基本不变）
% 已经在t=0时计算了range_bin，这里直接使用相同的tau_bin
final_phase_t_end_theory = geom_phase_t_end_theory + rvp_phase_t_end_actual - rvp_phase_t0_correction + debug_info.t_end.baseband_phase_component;
rvp_phase_t_end_theory = rvp_phase_t_end_actual;  % 保持输出变量名不变

% --- 对比打印 ---
fprintf('\n--- [深度调试] 阶段 2: 内部变量对账 ---\n');
fprintf('%-25s | %-20s | %-20s | %-15s\n', '变量', '理论值', '实际值 (จาก SigGen)', '误差');
fprintf([repmat('-', 1, 85) '\n']);

% t=0 对比
print_comparison('t=0: TX Pos X', tx_pos_t0_theory(1), debug_info.t0.tx_pos(1));
print_comparison('t=0: RX Pos X', rx_pos_t0_theory(1), debug_info.t0.rx_pos(1));
print_comparison('t=0: Range (m)', range_t0_theory, debug_info.t0.range);
print_comparison('t=0: Tau (s)', tau_t0_theory, debug_info.t0.tau);
print_comparison('t=0: Geom Phase (rad)', geom_phase_t0_theory, debug_info.t0.geom_phase_component);
print_comparison('t=0: RVP Phase (rad)', rvp_phase_t0_theory, debug_info.t0.rvp_phase_component);
fprintf([repmat('-', 1, 85) '\n']);

% t=end 对比
print_comparison('t=end: TX Pos X', tx_pos_t_end_theory(1), debug_info.t_end.tx_pos(1));
print_comparison('t=end: RX Pos X', rx_pos_t_end_theory(1), debug_info.t_end.rx_pos(1));
print_comparison('t=end: Range (m)', range_t_end_theory, debug_info.t_end.range);
print_comparison('t=end: Tau (s)', tau_t_end_theory, debug_info.t_end.tau);
print_comparison('t=end: Geom Phase (rad)', geom_phase_t_end_theory, debug_info.t_end.geom_phase_component);
print_comparison('t=end: RVP Phase (rad)', rvp_phase_t_end_theory, debug_info.t_end.rvp_phase_component);
fprintf([repmat('-', 1, 85) '\n']);

% 最终相位对比
final_phase_t0_actual = angle(debug_info.t0.final_complex_val);
final_phase_t_end_actual = angle(debug_info.t_end.final_complex_val);
print_comparison('Final Phase @t=0 (rad)', mod(final_phase_t0_theory, 2*pi), mod(final_phase_t0_actual, 2*pi));
print_comparison('Final Phase @t=end (rad)', mod(final_phase_t_end_theory, 2*pi), mod(final_phase_t_end_actual, 2*pi));
phase_diff_theory = unwrap([final_phase_t0_theory, final_phase_t_end_theory]);
phase_diff_actual = unwrap([final_phase_t0_actual, final_phase_t_end_actual]);
print_comparison('Phase Diff (rad)', phase_diff_theory(2)-phase_diff_theory(1), phase_diff_actual(2)-phase_diff_actual(1));

fprintf('--- [深度调试] 结束相位验证 ---\n\n');
% --- [调试] 模块结束 ---


% --- 3. 静态矩形阵列定义 (用于基准对比) ---
fprintf('3. 评估静态矩形阵列...\n');
array_rect_static = ArrayPlatform(rect_elements, 1, 1:num_elements); % 使用相同的物理阵元
array_rect_static = array_rect_static.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

sig_gen_stat = SignalGenerator(radar_params, array_rect_static, targets);
snapshots_stat = sig_gen_stat.generate_snapshots(t_axis, inf);
estimator_stat = DoaEstimator(array_rect_static, radar_params);
spectrum_stat = estimator_stat.estimate_gmusic(snapshots_stat, t_axis, num_targets, search_grid);

% --- 定量结果分析 ---
fprintf('静态矩形阵列估算结果:\n');
[theta_peaks_stat, phi_peaks_stat, peak_vals_stat] = DoaEstimator.find_peaks(spectrum_stat, search_grid, num_targets);
for i = 1:length(theta_peaks_stat)
    fprintf('  目标 %d: Theta = %.2f 度, Phi = %.2f 度 (谱峰值: %.2f)\n', i, theta_peaks_stat(i), phi_peaks_stat(i), peak_vals_stat(i));
end
fprintf('\n');


% --- 绘制结果图 ---
fig_handle = figure('Name', '不同阵列类型分辨率对比', 'Position', [100, 100, 1500, 450]);

% --- DEFINITIVE FIX: Robust plotting that handles any number of targets ---
% This entire block is rewritten to be robust.

% Plotting helper data
plot_titles = {'静态矩形阵列', '静态圆形阵列', '旋转矩形阵列'};
spectrums = {spectrum_stat, spectrum_circ, spectrum_rot};

for plot_idx = 1:3
    subplot(1, 3, plot_idx);
    surf(search_grid.phi, search_grid.theta, spectrums{plot_idx});
    shading interp; view(2); colorbar; hold on;
    
    % Dynamically plot markers for all existing targets
    markers = {'r+', 'rx', 'ro'};
    for target_idx = 1:numel(targets)
        target_pos = targets{target_idx}.get_position_at(0);
        [phi_true_rad, theta_true_rad, ~] = cart2sph(target_pos(1), target_pos(2), target_pos(3));
        theta_true = 90 - rad2deg(theta_true_rad);
        phi_true = rad2deg(phi_true_rad);
        
        plot(phi_true, theta_true, markers{mod(target_idx-1, numel(markers))+1}, 'MarkerSize', 12, 'LineWidth', 2);
    end
    
    hold off;
    title(plot_titles{plot_idx});
    xlabel('方位角 (phi) [度]');
    if plot_idx == 1
        ylabel('俯仰角 (theta) [度]');
    end
end

sgtitle(sprintf('分辨率对比 (%d 阵元)', num_elements));

% --- 附加：为旋转阵列结果创建三维MUSIC谱图 ---
figure('Name', '旋转矩形阵列三维MUSIC谱图');
surf(search_grid.phi, search_grid.theta, spectrum_rot);
shading interp;
hold on;

max_spectrum_val = max(spectrum_rot(:));
if isempty(max_spectrum_val) || ~isfinite(max_spectrum_val)
    max_spectrum_val = 1; % 防止谱为空或NaN
end

% Dynamically plot markers for all existing targets in 3D
legend_entries = {'MUSIC 谱'};
for target_idx = 1:numel(targets)
    target_pos = targets{target_idx}.get_position_at(0);
    u_true = target_pos / norm(target_pos);
    [~, true_phi_idx] = min(abs(search_grid.phi - atan2d(u_true(2), u_true(1))));
    [~, true_theta_idx] = min(abs(search_grid.theta - acosd(u_true(3))));
    
    marker_style = markers{mod(target_idx-1, numel(markers)) + 1};
    plot3(search_grid.phi(true_phi_idx), search_grid.theta(true_theta_idx), max_spectrum_val, ...
        marker_style, 'MarkerSize', 12, 'LineWidth', 3);
    legend_entries{end+1} = sprintf('真实目标 %d', target_idx);
end

hold off;
title('旋转矩形阵列 (4x2 URA) - 3D MUSIC 谱');
xlabel('方位角 (phi) [度]');
ylabel('俯仰角 (theta) [度]');
zlabel('伪谱值');
legend(legend_entries);
grid on;
view(3); % 确保是三维视角


%% Helper functions

function [tx_pos, rx_pos] = get_theory_tx_rx_pos(elements, state)
    % Helper to calculate theoretical positions for the first Tx/Rx pair
    orientation = state.orientation;
    pos = state.position;
    
    % Rotation matrix
    Rz = [cosd(orientation(3)) -sind(orientation(3)) 0; sind(orientation(3)) cosd(orientation(3)) 0; 0 0 1];
    Ry = [cosd(orientation(2)) 0 sind(orientation(2)); 0 1 0; -sind(orientation(2)) 0 cosd(orientation(2))];
    Rx = [1 0 0; 0 cosd(orientation(1)) -sind(orientation(1)); 0 sind(orientation(1)) cosd(orientation(1))];
    R = Rz * Ry * Rx;
    
    % Assuming first element is Tx and also first element is Rx for the first virtual element
    tx_pos_local = elements(1, :);
    rx_pos_local = elements(1, :);
    
    tx_pos = (R * tx_pos_local')' + pos;
    rx_pos = (R * rx_pos_local')' + pos;
end

function print_comparison(name, theory, actual)
    err = theory - actual;
    fprintf('%-25s | %-20.4e | %-20.4e | %-15.4e\n', name, theory, actual, err);
end

% --- DEFINITIVE FIX for Syntax Error ---
% This is the correct placement for the 'end' that closes the main function
% 'run_rotation_vs_circular_test'.
end
