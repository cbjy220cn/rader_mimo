%% 主瓣宽度优化测试
% 探索不同参数对时间平滑MUSIC主瓣宽度的影响
% 
% 核心问题：时间平滑MUSIC的有效孔径 ≠ 全虚拟孔径
% 原因：为了恢复协方差矩阵的秩，必须使用子阵列
% 折中：子阵列孔径 vs 协方差估计稳定性

clear; clc; close all;
addpath('asset');

fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('              主瓣宽度优化测试                                      \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

%% 基本参数
c = 3e8;
fc = 3e9;
lambda = c / fc;
d = lambda / 2;

% 8元ULA
num_elements = 8;
elements = zeros(num_elements, 3);
for i = 1:num_elements
    elements(i, :) = [(i-1)*d, 0, 0];
end
tx = 1;
rx = 1:num_elements;
array = ArrayPlatform(elements, tx, rx);

% 目标（极坐标转笛卡尔）
target_phi = 30;
target_theta = 90;  % 水平面
target_range = 500;
target_x = target_range * cosd(target_phi) * sind(target_theta);
target_y = target_range * sind(target_phi) * sind(target_theta);
target_z = target_range * cosd(target_theta);
target = {Target([target_x, target_y, target_z], [0, 0, 0], 1)};  % cell数组

% 运动参数
motion_velocity = 5;  % m/s
motion_direction = [1, 0, 0];

% 创建运动轨迹函数
trajectory_func = @(t) struct('position', motion_velocity * t * motion_direction, 'orientation', [0, 0, 0]);

fprintf('【基本参数】\n');
fprintf('  波长: %.2f cm\n', lambda * 100);
fprintf('  阵元数: %d\n', num_elements);
fprintf('  物理孔径: %.1fλ\n', (num_elements-1)*d/lambda);
fprintf('  目标角度: φ=%.0f°\n', target_phi);
fprintf('  运动速度: %.1f m/s\n\n', motion_velocity);

%% 测试1：不同快拍数对主瓣宽度的影响
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('测试1：快拍数对主瓣宽度的影响\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

snapshot_list = [32, 64, 128, 256];
T_obs = 0.5;  % 固定观测时间

results1 = struct();

for idx = 1:length(snapshot_list)
    num_snapshots = snapshot_list(idx);
    t_axis = linspace(0, T_obs, num_snapshots);
    
    % 设置运动
    array.set_trajectory(trajectory_func);
    
    % 计算运动距离和孔径
    motion_dist = motion_velocity * T_obs;
    full_aperture = (num_elements-1)*d + motion_dist;
    
    % 子阵列参数（当前设置）
    subarray_snapshots = min(8, max(2, floor(num_snapshots / 10)));
    num_subarrays = num_snapshots - subarray_snapshots + 1;
    subarray_size = num_elements * subarray_snapshots;
    
    while subarray_size > num_subarrays / 2 && subarray_snapshots > 2
        subarray_snapshots = subarray_snapshots - 1;
        num_subarrays = num_snapshots - subarray_snapshots + 1;
        subarray_size = num_elements * subarray_snapshots;
    end
    
    % 子阵列孔径
    step_size = motion_dist / (num_snapshots - 1);
    subarray_aperture = (num_elements-1)*d + (subarray_snapshots-1)*step_size;
    
    % 理论主瓣宽度 (近似)
    static_beamwidth = 0.886 * lambda / ((num_elements-1)*d) * 180/pi;
    subarray_beamwidth = 0.886 * lambda / subarray_aperture * 180/pi;
    full_beamwidth = 0.886 * lambda / full_aperture * 180/pi;
    
    results1(idx).num_snapshots = num_snapshots;
    results1(idx).subarray_snapshots = subarray_snapshots;
    results1(idx).num_subarrays = num_subarrays;
    results1(idx).subarray_size = subarray_size;
    results1(idx).subarray_aperture = subarray_aperture / lambda;
    results1(idx).full_aperture = full_aperture / lambda;
    results1(idx).static_beamwidth = static_beamwidth;
    results1(idx).subarray_beamwidth = subarray_beamwidth;
    results1(idx).full_beamwidth = full_beamwidth;
    
    fprintf('快拍数: %3d | 子阵列快拍: %2d | 子阵列数: %3d | ', ...
        num_snapshots, subarray_snapshots, num_subarrays);
    fprintf('子阵列孔径: %5.1fλ | 理论主瓣: %5.2f°\n', ...
        subarray_aperture/lambda, subarray_beamwidth);
end

fprintf('\n对比：\n');
fprintf('  静态孔径: %.1fλ → 理论主瓣: %.2f°\n', (num_elements-1)*d/lambda, results1(1).static_beamwidth);
fprintf('  全虚拟孔径: %.1fλ → 理论主瓣: %.2f° (波束形成可达)\n', results1(1).full_aperture, results1(1).full_beamwidth);

%% 测试2：不同子阵列快拍数的影响
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('测试2：子阵列快拍数对主瓣宽度的影响（固定128快拍）\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

num_snapshots = 128;
T_obs = 0.5;
t_axis = linspace(0, T_obs, num_snapshots);
motion_dist = motion_velocity * T_obs;
step_size = motion_dist / (num_snapshots - 1);

subarray_L_list = [4, 8, 16, 24, 32, 48];

fprintf('子阵列快拍L | 子阵列大小 | 子阵列数M | 条件M>N | 子阵列孔径 | 理论主瓣\n');
fprintf('------------|------------|-----------|---------|------------|----------\n');

for L = subarray_L_list
    N = num_elements * L;  % 子阵列大小
    M = num_snapshots - L + 1;  % 子阵列数
    
    subarray_aperture = (num_elements-1)*d + (L-1)*step_size;
    subarray_beamwidth = 0.886 * lambda / subarray_aperture * 180/pi;
    
    condition_ok = M > N;
    condition_str = '✓';
    if ~condition_ok
        condition_str = '✗';
    end
    
    fprintf('     %3d    |    %4d    |    %3d    |    %s    |   %5.1fλ   |  %5.2f°\n', ...
        L, N, M, condition_str, subarray_aperture/lambda, subarray_beamwidth);
end

%% 测试3：实际MUSIC谱测量
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('测试3：实际MUSIC谱主瓣宽度测量\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% 生成信号
num_snapshots = 128;
T_obs = 0.5;
t_axis = linspace(0, T_obs, num_snapshots);
SNR = 20;

array.set_trajectory(trajectory_func);

% 创建radar_params结构体
radar_params = struct();
radar_params.fc = fc;
radar_params.lambda = lambda;

% 使用SignalGeneratorSimple生成信号
sig_gen = SignalGeneratorSimple(radar_params, array, target);
snapshots = sig_gen.generate_snapshots(t_axis, SNR);

% 搜索网格
phi_range = linspace(0, 90, 901);  % 0.1°分辨率

% 测试不同子阵列快拍数
test_L_list = [6, 12, 24];

figure('Position', [100, 100, 1200, 400]);

for test_idx = 1:length(test_L_list)
    L = test_L_list(test_idx);
    
    % 手动时间平滑MUSIC
    N = num_elements * L;
    M = num_snapshots - L + 1;
    
    if M < N
        fprintf('L=%d: 子阵列数(%d) < 子阵列大小(%d)，跳过\n', L, M, N);
        continue;
    end
    
    % 构建子阵列位置
    subarray_positions = zeros(N, 3);
    for k = 1:L
        t_k = t_axis(k);
        pos_k = array.get_mimo_virtual_positions(t_k);
        idx_start = (k-1)*num_elements + 1;
        idx_end = k*num_elements;
        subarray_positions(idx_start:idx_end, :) = pos_k;
    end
    
    % 子阵列孔径
    subarray_aperture = max(subarray_positions(:,1)) - min(subarray_positions(:,1));
    
    % 空间平滑协方差矩阵
    R_smooth = zeros(N, N);
    for i = 1:M
        x_sub = zeros(N, 1);
        for k = 1:L
            snap_idx = i + k - 1;
            idx_start = (k-1)*num_elements + 1;
            idx_end = k*num_elements;
            x_sub(idx_start:idx_end) = snapshots(:, snap_idx);
        end
        R_smooth = R_smooth + (x_sub * x_sub');
    end
    R_smooth = R_smooth / M;
    R_smooth = R_smooth + 1e-8 * trace(R_smooth) / N * eye(N);
    
    % 特征分解
    [V, D] = eig(R_smooth);
    [eigenvalues, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    
    % 噪声子空间
    num_targets = 1;
    Qn = V(:, (num_targets+1):end);
    
    % MUSIC谱
    spectrum = zeros(size(phi_range));
    for phi_idx = 1:length(phi_range)
        phi = phi_range(phi_idx);
        theta = 90;
        
        u = [cosd(phi)*sind(theta), sind(phi)*sind(theta), cosd(theta)];
        phase = 4*pi/lambda * (subarray_positions * u');
        a = exp(-1j * phase);
        
        spectrum(phi_idx) = 1 / real(a' * (Qn * Qn') * a);
    end
    
    % 归一化
    spectrum_db = 10*log10(spectrum / max(spectrum));
    
    % 找主瓣宽度
    [~, peak_idx] = max(spectrum_db);
    
    % 找-3dB点
    left_idx = peak_idx;
    while left_idx > 1 && spectrum_db(left_idx) > -3
        left_idx = left_idx - 1;
    end
    
    right_idx = peak_idx;
    while right_idx < length(spectrum_db) && spectrum_db(right_idx) > -3
        right_idx = right_idx + 1;
    end
    
    beamwidth_measured = phi_range(right_idx) - phi_range(left_idx);
    beamwidth_theory = 0.886 * lambda / subarray_aperture * 180/pi;
    
    fprintf('L=%2d: 子阵列孔径=%.1fλ, 理论主瓣=%.2f°, 实测主瓣=%.2f°, 估计角度=%.1f°\n', ...
        L, subarray_aperture/lambda, beamwidth_theory, beamwidth_measured, phi_range(peak_idx));
    
    % 绘图
    subplot(1, length(test_L_list), test_idx);
    plot(phi_range, spectrum_db, 'LineWidth', 2);
    hold on;
    xline(target_phi, 'r--', 'LineWidth', 1.5);
    yline(-3, 'g--', 'LineWidth', 1);
    xlabel('φ (°)');
    ylabel('功率 (dB)');
    title(sprintf('L=%d, 孔径=%.1fλ, 主瓣=%.2f°', L, subarray_aperture/lambda, beamwidth_measured));
    xlim([target_phi-20, target_phi+20]);
    ylim([-30, 5]);
    grid on;
end

sgtitle('不同子阵列快拍数的MUSIC谱');

%% 测试4：静态 vs 运动对比
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('测试4：静态阵列 vs 运动阵列主瓣宽度对比\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% 静态阵列
array_static = ArrayPlatform(elements, tx, rx);
% 不设置运动

% 生成静态信号
sig_gen_static = SignalGeneratorSimple(radar_params, array_static, target);
snapshots_static = sig_gen_static.generate_snapshots(t_axis, SNR);

% 静态MUSIC
Rxx_static = (snapshots_static * snapshots_static') / num_snapshots;
Rxx_static = Rxx_static + 1e-8 * trace(Rxx_static) / num_elements * eye(num_elements);
[V_static, D_static] = eig(Rxx_static);
[~, idx] = sort(diag(D_static), 'descend');
V_static = V_static(:, idx);
Qn_static = V_static(:, 2:end);

% 静态MUSIC谱
spectrum_static = zeros(size(phi_range));
pos_static = array_static.get_mimo_virtual_positions(0);
for phi_idx = 1:length(phi_range)
    phi = phi_range(phi_idx);
    theta = 90;
    u = [cosd(phi)*sind(theta), sind(phi)*sind(theta), cosd(theta)];
    phase = 4*pi/lambda * (pos_static * u');
    a = exp(-1j * phase);
    spectrum_static(phi_idx) = 1 / real(a' * (Qn_static * Qn_static') * a);
end
spectrum_static_db = 10*log10(spectrum_static / max(spectrum_static));

% 找静态主瓣宽度
[~, peak_idx] = max(spectrum_static_db);
left_idx = peak_idx;
while left_idx > 1 && spectrum_static_db(left_idx) > -3
    left_idx = left_idx - 1;
end
right_idx = peak_idx;
while right_idx < length(spectrum_static_db) && spectrum_static_db(right_idx) > -3
    right_idx = right_idx + 1;
end
static_beamwidth = phi_range(right_idx) - phi_range(left_idx);

% 运动阵列（最佳子阵列参数）
L_best = 24;  % 较大的子阵列快拍数
N = num_elements * L_best;
M = num_snapshots - L_best + 1;

subarray_positions = zeros(N, 3);
for k = 1:L_best
    t_k = t_axis(k);
    pos_k = array.get_mimo_virtual_positions(t_k);
    idx_start = (k-1)*num_elements + 1;
    idx_end = k*num_elements;
    subarray_positions(idx_start:idx_end, :) = pos_k;
end

motion_aperture = max(subarray_positions(:,1)) - min(subarray_positions(:,1));

R_smooth = zeros(N, N);
for i = 1:M
    x_sub = zeros(N, 1);
    for k = 1:L_best
        snap_idx = i + k - 1;
        idx_start = (k-1)*num_elements + 1;
        idx_end = k*num_elements;
        x_sub(idx_start:idx_end) = snapshots(:, snap_idx);
    end
    R_smooth = R_smooth + (x_sub * x_sub');
end
R_smooth = R_smooth / M;
R_smooth = R_smooth + 1e-8 * trace(R_smooth) / N * eye(N);

[V, D] = eig(R_smooth);
[~, idx] = sort(diag(D), 'descend');
V = V(:, idx);
Qn = V(:, 2:end);

spectrum_motion = zeros(size(phi_range));
for phi_idx = 1:length(phi_range)
    phi = phi_range(phi_idx);
    theta = 90;
    u = [cosd(phi)*sind(theta), sind(phi)*sind(theta), cosd(theta)];
    phase = 4*pi/lambda * (subarray_positions * u');
    a = exp(-1j * phase);
    spectrum_motion(phi_idx) = 1 / real(a' * (Qn * Qn') * a);
end
spectrum_motion_db = 10*log10(spectrum_motion / max(spectrum_motion));

[~, peak_idx] = max(spectrum_motion_db);
left_idx = peak_idx;
while left_idx > 1 && spectrum_motion_db(left_idx) > -3
    left_idx = left_idx - 1;
end
right_idx = peak_idx;
while right_idx < length(spectrum_motion_db) && spectrum_motion_db(right_idx) > -3
    right_idx = right_idx + 1;
end
motion_beamwidth = phi_range(right_idx) - phi_range(left_idx);

fprintf('【结果对比】\n');
fprintf('  静态阵列: 孔径=%.1fλ, 主瓣宽度=%.2f°\n', (num_elements-1)*d/lambda, static_beamwidth);
fprintf('  运动阵列: 孔径=%.1fλ, 主瓣宽度=%.2f°\n', motion_aperture/lambda, motion_beamwidth);
fprintf('  改善倍数: %.2f×\n', static_beamwidth / motion_beamwidth);

% 绘图对比
figure('Position', [100, 500, 800, 400]);
plot(phi_range, spectrum_static_db, 'b-', 'LineWidth', 2, 'DisplayName', sprintf('静态 (%.2f°)', static_beamwidth));
hold on;
plot(phi_range, spectrum_motion_db, 'r-', 'LineWidth', 2, 'DisplayName', sprintf('运动 (%.2f°)', motion_beamwidth));
xline(target_phi, 'k--', 'LineWidth', 1.5, 'DisplayName', '真实角度');
yline(-3, 'g--', 'LineWidth', 1, 'DisplayName', '-3dB');
xlabel('φ (°)', 'FontSize', 12);
ylabel('归一化功率 (dB)', 'FontSize', 12);
title('静态 vs 运动阵列 MUSIC谱对比', 'FontSize', 14);
xlim([target_phi-30, target_phi+30]);
ylim([-40, 5]);
legend('Location', 'northeast');
grid on;

%% 结论
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                        结论                                        \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

fprintf('【时间平滑MUSIC的孔径-秩折中】\n\n');
fprintf('1. 全虚拟孔径 = 物理孔径 + 运动距离 = %.1fλ\n', motion_velocity*T_obs/lambda + (num_elements-1)*d/lambda);
fprintf('   → 理论最小主瓣 = %.2f° (波束形成可达)\n\n', 0.886*lambda/(motion_velocity*T_obs + (num_elements-1)*d)*180/pi);

fprintf('2. 时间平滑MUSIC的有效孔径 = 子阵列孔径\n');
fprintf('   当前设置: L=%d → 子阵列孔径 = %.1fλ → 主瓣 = %.2f°\n\n', L_best, motion_aperture/lambda, motion_beamwidth);

fprintf('3. 折中关系:\n');
fprintf('   - 增大L → 子阵列孔径增大 → 主瓣变窄 ✓\n');
fprintf('   - 增大L → 子阵列数M减少 → 协方差估计不稳定 ✗\n');
fprintf('   - 需要满足 M > N (子阵列数 > 子阵列大小)\n\n');

fprintf('4. 优化建议:\n');
fprintf('   - 增加总快拍数 → 可同时增大L和M\n');
fprintf('   - 增加观测时间 → 运动距离增加，子阵列孔径增大\n');
fprintf('   - 对于当前参数，最优L约为%d\n', L_best);

fprintf('\n测试完成！\n');

