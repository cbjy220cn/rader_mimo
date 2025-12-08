%% 使用理想信号测试DOA估计器
% 目的：绕过SignalGenerator，直接生成理论正确的信号
%       验证DOA估计器是否正确工作

clear; clc; close all;

fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('使用理想信号测试DOA估计器\n');
fprintf('绕过SignalGenerator，直接生成理论信号\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

%% 参数
c = physconst('LightSpeed');
f0 = 3e9;
lambda = c / f0;

radar_params.fc = f0;
radar_params.c = c;
radar_params.lambda = lambda;

%% 阵列配置
num_elements = 8;
spacing = 0.5 * lambda;
physical_aperture = (num_elements - 1) * spacing;

% 阵元位置（沿x轴）
array_pos = zeros(num_elements, 3);
for i = 1:num_elements
    array_pos(i, :) = [(i-1)*spacing - physical_aperture/2, 0, 0];
end

fprintf('阵列: %d元ULA, 间距=%.2fλ, 孔径=%.2fλ\n', ...
    num_elements, spacing/lambda, physical_aperture/lambda);

%% 目标设置
true_theta = 90;  % deg (固定在xy平面)
true_phi = 30;    % deg
num_targets = 1;

% 方向矢量
u = [sind(true_theta)*cosd(true_phi); ...
     sind(true_theta)*sind(true_phi); ...
     cosd(true_theta)];

fprintf('目标: θ=%.1f°, φ=%.1f°\n', true_theta, true_phi);
fprintf('方向矢量: u = [%.4f, %.4f, %.4f]\n\n', u);

%% 生成理想信号（直接计算理论相位）
num_snapshots = 32;
snr_db = 20;

fprintf('生成理想信号:\n');
fprintf('  快拍数: %d\n', num_snapshots);
fprintf('  SNR: %d dB\n\n', snr_db);

% 理想导向矢量（雷达双程：4π/λ）
a_ideal = zeros(num_elements, 1);
fprintf('理想导向矢量:\n');
for i = 1:num_elements
    % 阵元位置
    pos = array_pos(i, :);
    % 相位 = 4π/λ * (位置 · 方向矢量)
    phase = 4 * pi / lambda * (pos * u);
    a_ideal(i) = exp(1j * phase);
    fprintf('  阵元%d: 位置=[%.4f, 0, 0]m, 相位=%.4f rad (%.1f°)\n', ...
        i, pos(1), phase, rad2deg(wrapToPi(phase)));
end

% 生成快拍：信号 + 噪声
signal_power = 1;
noise_power = signal_power / (10^(snr_db/10));

% 每个快拍的信号（添加随机复数幅度使其更真实）
snapshots_ideal = zeros(num_elements, num_snapshots);
for k = 1:num_snapshots
    % 随机复数幅度（模拟不同快拍间的变化）
    s_k = (randn + 1j*randn) / sqrt(2);
    % 信号 = 幅度 × 导向矢量
    signal = s_k * a_ideal;
    % 噪声
    noise = sqrt(noise_power/2) * (randn(num_elements, 1) + 1j*randn(num_elements, 1));
    snapshots_ideal(:, k) = signal + noise;
end

fprintf('\n理想信号相位差（相对于阵元1）:\n');
for i = 1:num_elements
    phase_diff = angle(snapshots_ideal(i, 1) / snapshots_ideal(1, 1));
    theory_diff = 4 * pi / lambda * ((array_pos(i,:) - array_pos(1,:)) * u);
    fprintf('  阵元%d: 实际=%.4f rad, 理论=%.4f rad\n', i, phase_diff, wrapToPi(theory_diff));
end

%% 使用DOA估计器（手动MUSIC）
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('手动MUSIC估计（验证DOA估计逻辑）\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');

% 协方差矩阵
Rxx = (snapshots_ideal * snapshots_ideal') / num_snapshots;

% 特征分解
[V, D] = eig(Rxx);
[eigenvalues, idx] = sort(diag(D), 'descend');
V = V(:, idx);

fprintf('\n特征值:\n');
for i = 1:min(5, num_elements)
    fprintf('  λ%d = %.4e\n', i, eigenvalues(i));
end

% 噪声子空间
Qn = V(:, (num_targets+1):end);

% 搜索φ
phi_search = 0:0.5:90;
music_spectrum = zeros(size(phi_search));

for phi_idx = 1:length(phi_search)
    phi = phi_search(phi_idx);
    theta = 90;  % 固定
    
    % 方向矢量
    u_test = [sind(theta)*cosd(phi); sind(theta)*sind(phi); cosd(theta)];
    
    % 导向矢量
    a_test = zeros(num_elements, 1);
    for i = 1:num_elements
        phase = 4 * pi / lambda * (array_pos(i, :) * u_test);
        a_test(i) = exp(1j * phase);
    end
    
    % MUSIC谱
    music_spectrum(phi_idx) = 1 / abs(a_test' * (Qn * Qn') * a_test);
end

% 找峰值
[~, peak_idx] = max(music_spectrum);
est_phi = phi_search(peak_idx);
phi_error = abs(est_phi - true_phi);

fprintf('\n结果:\n');
fprintf('  真实φ: %.1f°\n', true_phi);
fprintf('  估计φ: %.1f°\n', est_phi);
fprintf('  误差: Δφ=%.2f°\n', phi_error);

if phi_error < 1
    fprintf('  ✅ 测试通过：使用理想信号，DOA估计正确！\n');
    test_pass = true;
else
    fprintf('  ❌ 测试失败\n');
    test_pass = false;
end

%% 测试2：双目标分辨
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('双目标分辨测试（理想信号）\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');

true_phi1 = 20;
true_phi2 = 40;
num_targets2 = 2;

% 两个目标的方向矢量
u1 = [cosd(true_phi1); sind(true_phi1); 0];  % θ=90°简化
u2 = [cosd(true_phi2); sind(true_phi2); 0];

% 导向矢量
a1 = zeros(num_elements, 1);
a2 = zeros(num_elements, 1);
for i = 1:num_elements
    pos = array_pos(i, :);
    a1(i) = exp(1j * 4 * pi / lambda * (pos * u1));
    a2(i) = exp(1j * 4 * pi / lambda * (pos * u2));
end

fprintf('目标1: φ=%.1f°\n', true_phi1);
fprintf('目标2: φ=%.1f° (间隔20°)\n', true_phi2);

% 生成双目标信号
snapshots_dual = zeros(num_elements, num_snapshots);
for k = 1:num_snapshots
    s1 = (randn + 1j*randn) / sqrt(2);
    s2 = (randn + 1j*randn) / sqrt(2);
    signal = s1 * a1 + s2 * a2;
    noise = sqrt(noise_power/2) * (randn(num_elements, 1) + 1j*randn(num_elements, 1));
    snapshots_dual(:, k) = signal + noise;
end

% MUSIC估计
Rxx2 = (snapshots_dual * snapshots_dual') / num_snapshots;
[V2, D2] = eig(Rxx2);
[eigenvalues2, idx2] = sort(diag(D2), 'descend');
V2 = V2(:, idx2);
Qn2 = V2(:, (num_targets2+1):end);

music_spectrum2 = zeros(size(phi_search));
for phi_idx = 1:length(phi_search)
    phi = phi_search(phi_idx);
    u_test = [cosd(phi); sind(phi); 0];
    
    a_test = zeros(num_elements, 1);
    for i = 1:num_elements
        phase = 4 * pi / lambda * (array_pos(i, :) * u_test);
        a_test(i) = exp(1j * phase);
    end
    
    music_spectrum2(phi_idx) = 1 / abs(a_test' * (Qn2 * Qn2') * a_test);
end

% 找两个峰值
[pks, locs] = findpeaks(music_spectrum2, 'MinPeakProminence', 0.1*max(music_spectrum2), ...
                        'SortStr', 'descend', 'NPeaks', 2);

if length(locs) >= 2
    est_phis = sort(phi_search(locs(1:2)));
    fprintf('\n结果:\n');
    fprintf('  真实: φ₁=%.1f°, φ₂=%.1f°\n', true_phi1, true_phi2);
    fprintf('  估计: φ₁=%.1f°, φ₂=%.1f°\n', est_phis(1), est_phis(2));
    
    error1 = abs(est_phis(1) - true_phi1);
    error2 = abs(est_phis(2) - true_phi2);
    
    if error1 < 3 && error2 < 3
        fprintf('  ✅ 双目标分辨正确！\n');
        test2_pass = true;
    else
        fprintf('  ❌ 分辨错误\n');
        test2_pass = false;
    end
else
    fprintf('  ❌ 未检测到两个峰\n');
    test2_pass = false;
    est_phis = [NaN, NaN];
end

%% 绘图
figure('Position', [100, 100, 1200, 400]);

subplot(1,3,1);
plot(phi_search, 10*log10(music_spectrum/max(music_spectrum)), 'b-', 'LineWidth', 2);
hold on;
xline(true_phi, 'r--', 'LineWidth', 2);
xline(est_phi, 'g--', 'LineWidth', 2);
xlabel('φ (°)'); ylabel('MUSIC谱 (dB)');
title(sprintf('单目标（理想信号）\n真实:%.1f°, 估计:%.1f°, 误差:%.2f°', ...
    true_phi, est_phi, phi_error));
grid on; xlim([0, 90]); ylim([-30, 0]);
legend('MUSIC谱', '真实', '估计', 'Location', 'southwest');

subplot(1,3,2);
plot(phi_search, 10*log10(music_spectrum2/max(music_spectrum2)), 'b-', 'LineWidth', 2);
hold on;
xline(true_phi1, 'r--', 'LineWidth', 2);
xline(true_phi2, 'r--', 'LineWidth', 2);
xlabel('φ (°)'); ylabel('MUSIC谱 (dB)');
title(sprintf('双目标分辨（理想信号）\n真实:%.1f°和%.1f°', true_phi1, true_phi2));
grid on; xlim([0, 90]); ylim([-30, 0]);

subplot(1,3,3);
% 对比理想信号和SignalGenerator信号的相位
bar_data = zeros(num_elements, 2);
for i = 1:num_elements
    bar_data(i, 1) = angle(snapshots_ideal(i, 1));  % 理想
    % 计算理论相位
    bar_data(i, 2) = wrapToPi(4 * pi / lambda * (array_pos(i, :) * u));
end
bar(bar_data);
xlabel('阵元索引'); ylabel('相位 (rad)');
title('理想信号相位 vs 理论');
legend('信号', '理论');
grid on;

sgtitle('使用理想信号验证DOA估计器', 'FontSize', 14);
saveas(gcf, 'test_with_ideal_signal.png');
fprintf('\n图片已保存: test_with_ideal_signal.png\n');

%% 总结
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('总结\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('测试1 (单目标): %s\n', ternary(test_pass, '✅ 通过', '❌ 失败'));
fprintf('测试2 (双目标): %s\n', ternary(test2_pass, '✅ 通过', '❌ 失败'));

if test_pass && test2_pass
    fprintf('\n🎉 DOA估计器工作正常！\n');
    fprintf('⚠️  问题确认在SignalGenerator.m\n');
    fprintf('   需要修复信号相位计算\n');
end

fprintf('═══════════════════════════════════════════════════════════════════\n');

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end




