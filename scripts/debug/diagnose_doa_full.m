%% 全面诊断DOA估计系统
% 目的：找出为什么运动阵列在高SNR下效果不如静态阵列
% 
% 注意：不要使用 pi 作为变量名！！！

clc; clear; close all;
addpath('../asset');

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║             DOA估计系统全面诊断 (修正版)                     ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

%% 基本参数
fc = 3e9;
c = physconst('LightSpeed');
lambda = c / fc;
d = lambda / 2;  % 阵元间距

radar_params.fc = fc;
radar_params.lambda = lambda;
radar_params.T_chirp = 10e-3;

% 目标参数
target_phi = 30;      % 方位角
target_theta = 90;    % 俯仰角（水平面）
target_R = 500;       % 距离

% 将球坐标转换为笛卡尔坐标
target_pos = target_R * [cosd(target_phi), sind(target_phi), 0];

fprintf('【测试参数】\n');
fprintf('  波长: %.2f cm\n', lambda*100);
fprintf('  目标: φ=%.0f°, θ=%.0f°, R=%.0fm\n', target_phi, target_theta, target_R);
fprintf('  目标位置: [%.1f, %.1f, %.1f] m\n', target_pos(1), target_pos(2), target_pos(3));

%% ═══════════════════════════════════════════════════════════════════
%  测试1: 验证静态阵列的信号生成和导向矢量匹配
%% ═══════════════════════════════════════════════════════════════════
fprintf('\n═══════════════════════════════════════════════════════════════\n');
fprintf('测试1: 静态阵列 - 相对相位匹配检验\n');
fprintf('═══════════════════════════════════════════════════════════════\n');

% 创建8元ULA
num_elements = 8;
element_positions = zeros(num_elements, 3);
for idx = 1:num_elements
    element_positions(idx, :) = [(idx-1)*d, 0, 0];
end

% 计算真实信号（精确球面波）
signals_exact = zeros(num_elements, 1);
for idx = 1:num_elements
    pos = element_positions(idx, :);
    range = 2 * norm(target_pos - pos);  % 双程距离
    phase = 2 * pi * range / lambda;
    signals_exact(idx) = exp(-1j * phase);
end

% 计算导向矢量（平面波近似）
u = [cosd(target_phi); sind(target_phi); 0];  % 方向单位向量
steering_vector = exp(1j * 4 * pi / lambda * (element_positions * u));

% 关键检验：归一化后的相位差
signals_norm = signals_exact / signals_exact(1);  % 以第一个阵元为参考
steering_norm = steering_vector / steering_vector(1);

phase_diff = angle(signals_norm) - angle(steering_norm);

fprintf('阵元 | 信号相位 | 导向相位 | 差异(°)\n');
fprintf('-----|----------|----------|--------\n');
for idx = 1:num_elements
    fprintf('  %d  | %7.2f° | %7.2f° | %+6.2f°\n', ...
        idx, rad2deg(angle(signals_norm(idx))), ...
        rad2deg(angle(steering_norm(idx))), ...
        rad2deg(phase_diff(idx)));
end

fprintf('\n相位差异标准差: %.4f° (应该接近0)\n', rad2deg(std(phase_diff)));
if rad2deg(std(phase_diff)) < 1
    fprintf('✓ 静态阵列: 信号与导向矢量相位匹配良好\n');
else
    fprintf('✗ 静态阵列: 存在相位不匹配问题！\n');
end

%% ═══════════════════════════════════════════════════════════════════
%  测试2: 静态MUSIC是否正常工作
%% ═══════════════════════════════════════════════════════════════════
fprintf('\n═══════════════════════════════════════════════════════════════\n');
fprintf('测试2: 静态MUSIC性能验证\n');
fprintf('═══════════════════════════════════════════════════════════════\n');

% 创建阵列平台（静态）
array = ArrayPlatform(element_positions, 1, 1:num_elements);
array.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

% 创建目标
targets = {Target(target_pos, [0,0,0], 1)};

% 创建信号生成器
sig_gen = SignalGeneratorSimple(radar_params, array, targets);

% 测试不同SNR
snr_list = [0, 10, 20, 40];
num_snapshots = 64;
T_chirp = radar_params.T_chirp;
t_axis = (0:num_snapshots-1) * T_chirp;

fprintf('SNR(dB) | MUSIC估计 | 真实值 | 误差\n');
fprintf('--------|-----------|--------|------\n');

phi_search = 0:0.1:60;

for snr = snr_list
    % 生成信号
    rng(42);  % 固定随机种子
    snapshots = sig_gen.generate_snapshots(t_axis, snr);
    
    % 标准MUSIC
    Rxx = snapshots * snapshots' / num_snapshots;
    [V, D] = eig(Rxx);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    Qn = V(:, 2:end);  % 噪声子空间（假设1个目标）
    
    % MUSIC谱
    spectrum = zeros(size(phi_search));
    for phi_idx = 1:length(phi_search)
        phi = phi_search(phi_idx);
        u_test = [cosd(phi); sind(phi); 0];
        a = exp(1j * 4 * pi / lambda * (element_positions * u_test));
        spectrum(phi_idx) = 1 / real(a' * (Qn * Qn') * a);
    end
    
    [~, peak_idx] = max(spectrum);
    phi_est = phi_search(peak_idx);
    error = phi_est - target_phi;
    
    fprintf('  %3d   |   %6.2f° |  %4.0f° | %+.2f°\n', snr, phi_est, target_phi, error);
end

%% ═══════════════════════════════════════════════════════════════════
%  测试3: 运动阵列CSA-BF测试
%% ═══════════════════════════════════════════════════════════════════
fprintf('\n═══════════════════════════════════════════════════════════════\n');
fprintf('测试3: 运动阵列CSA-BF性能\n');
fprintf('═══════════════════════════════════════════════════════════════\n');

v = 5;  % 运动速度 m/s

% 设置运动轨迹（x方向平移）
array.set_trajectory(@(t) struct('position', [v*t, 0, 0], 'orientation', [0,0,0]));

% 生成运动阵列的信号
sig_gen_motion = SignalGeneratorSimple(radar_params, array, targets);

fprintf('SNR(dB) | CSA-BF估计 | 真实值 | 误差  | 孔径(λ)\n');
fprintf('--------|------------|--------|-------|--------\n');

for snr = snr_list
    rng(42);
    snapshots = sig_gen_motion.generate_snapshots(t_axis, snr);
    
    % 构建虚拟阵列
    num_virtual = num_elements * num_snapshots;
    virtual_positions = zeros(num_virtual, 3);
    virtual_signals = zeros(num_virtual, 1);
    
    for k = 1:num_snapshots
        t_k = t_axis(k);
        positions_k = array.get_mimo_virtual_positions(t_k);
        idx_start = (k-1)*num_elements + 1;
        idx_end = k*num_elements;
        virtual_positions(idx_start:idx_end, :) = positions_k;
        virtual_signals(idx_start:idx_end) = snapshots(:, k);
    end
    
    % 计算合成孔径
    aperture = (max(virtual_positions(:,1)) - min(virtual_positions(:,1))) / lambda;
    
    % 波束形成搜索
    spectrum = zeros(size(phi_search));
    for phi_idx = 1:length(phi_search)
        phi = phi_search(phi_idx);
        u_test = [cosd(phi); sind(phi); 0];
        a = exp(1j * 4 * pi / lambda * (virtual_positions * u_test));
        spectrum(phi_idx) = abs(a' * virtual_signals)^2 / real(a' * a);
    end
    
    [~, peak_idx] = max(spectrum);
    phi_est = phi_search(peak_idx);
    error = phi_est - target_phi;
    
    fprintf('  %3d   |   %6.2f°  |  %4.0f° | %+.2f° | %.1f\n', ...
        snr, phi_est, target_phi, error, aperture);
end

%% ═══════════════════════════════════════════════════════════════════
%  测试4: 无噪声验证
%% ═══════════════════════════════════════════════════════════════════
fprintf('\n═══════════════════════════════════════════════════════════════\n');
fprintf('测试4: 无噪声波束形成验证\n');
fprintf('═══════════════════════════════════════════════════════════════\n');

% 无噪声信号
rng(42);
snapshots_noiseless = sig_gen_motion.generate_snapshots(t_axis, Inf);

% 构建虚拟阵列
virtual_positions = zeros(num_elements * num_snapshots, 3);
virtual_signals = zeros(num_elements * num_snapshots, 1);

for k = 1:num_snapshots
    t_k = t_axis(k);
    positions_k = array.get_mimo_virtual_positions(t_k);
    idx_start = (k-1)*num_elements + 1;
    idx_end = k*num_elements;
    virtual_positions(idx_start:idx_end, :) = positions_k;
    virtual_signals(idx_start:idx_end) = snapshots_noiseless(:, k);
end

% 细搜索
phi_fine = 20:0.02:40;
spectrum_fine = zeros(size(phi_fine));

for phi_idx = 1:length(phi_fine)
    phi = phi_fine(phi_idx);
    u_test = [cosd(phi); sind(phi); 0];
    a = exp(1j * 4 * pi / lambda * (virtual_positions * u_test));
    spectrum_fine(phi_idx) = abs(a' * virtual_signals)^2 / real(a' * a);
end

% 归一化
spectrum_db = 10*log10(spectrum_fine / max(spectrum_fine));

[~, peak_idx] = max(spectrum_db);
phi_peak = phi_fine(peak_idx);

fprintf('真实角度: %.2f°\n', target_phi);
fprintf('峰值角度: %.2f°\n', phi_peak);
fprintf('误差: %.4f°\n', abs(phi_peak - target_phi));

% 计算3dB宽度
idx_3db = find(spectrum_db > -3);
if ~isempty(idx_3db)
    beamwidth = phi_fine(idx_3db(end)) - phi_fine(idx_3db(1));
    fprintf('3dB主瓣宽度: %.4f°\n', beamwidth);
end

%% ═══════════════════════════════════════════════════════════════════
%  测试5: 检验虚拟阵列相位一致性
%% ═══════════════════════════════════════════════════════════════════
fprintf('\n═══════════════════════════════════════════════════════════════\n');
fprintf('测试5: 虚拟阵列相位一致性\n');
fprintf('═══════════════════════════════════════════════════════════════\n');

% 理论导向矢量（真实方向）
u_true = [cosd(target_phi); sind(target_phi); 0];
a_theoretical = exp(1j * 4 * pi / lambda * (virtual_positions * u_true));

% 归一化比较
a_norm = a_theoretical / a_theoretical(1);
s_norm = virtual_signals / virtual_signals(1);

phase_error = angle(s_norm ./ a_norm);
phase_error_deg = rad2deg(phase_error);

fprintf('相位误差统计:\n');
fprintf('  均值: %.2f°\n', mean(phase_error_deg));
fprintf('  标准差: %.2f°\n', std(phase_error_deg));
fprintf('  最大: %.2f°\n', max(abs(phase_error_deg)));

if std(phase_error_deg) > 10
    fprintf('⚠️ 警告: 相位误差较大！\n');
    fprintf('   这会导致合成孔径方法性能下降\n');
else
    fprintf('✓ 相位匹配良好\n');
end

%% 画图
figure('Position', [100, 100, 1200, 400]);

subplot(1,3,1);
plot(phi_fine, spectrum_db, 'b-', 'LineWidth', 1.5);
hold on;
xline(target_phi, 'r--', 'LineWidth', 1.5);
xlabel('方位角 φ (°)');
ylabel('功率 (dB)');
title(sprintf('无噪声波束形成谱\n峰值: %.2f°, 误差: %.2f°', phi_peak, phi_peak-target_phi));
legend('CSA-BF谱', '真实角度', 'Location', 'best');
grid on;
ylim([-40, 0]);

subplot(1,3,2);
plot(1:length(phase_error_deg), phase_error_deg, 'b.');
xlabel('虚拟阵元索引');
ylabel('相位误差 (°)');
title(sprintf('相位误差\n标准差: %.2f°', std(phase_error_deg)));
grid on;

subplot(1,3,3);
% 画虚拟阵列位置
scatter(virtual_positions(:,1), virtual_positions(:,2), 10, phase_error_deg, 'filled');
colorbar;
xlabel('X (m)');
ylabel('Y (m)');
title('相位误差空间分布');
axis equal;
grid on;

sgtitle('DOA系统诊断结果 (修正版)');

fprintf('\n═══════════════════════════════════════════════════════════════\n');
fprintf('诊断完成！\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
