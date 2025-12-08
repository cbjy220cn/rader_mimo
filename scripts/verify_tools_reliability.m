%% ═══════════════════════════════════════════════════════════════════════════
%  工具类可靠性验证实验 (修复版)
%  
%  修复：导向矢量符号问题
%% ═══════════════════════════════════════════════════════════════════════════

clear; clc; close all;

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║       工具类可靠性验证实验 (修复版)                           ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

%% 基础参数
c = physconst('LightSpeed');
f0 = 3e9;
lambda = c / f0;

radar_params.fc = f0;
radar_params.c = c;
radar_params.lambda = lambda;

fprintf('📡 雷达参数: f₀=%.2f GHz, λ=%.2f cm\n\n', f0/1e9, lambda*100);

%% 阵列配置
num_elements = 8;
spacing = 0.5 * lambda;
physical_aperture = (num_elements - 1) * spacing;

% 阵元位置（沿x轴，以中心为原点）
array_pos = zeros(num_elements, 3);
for i = 1:num_elements
    array_pos(i, :) = [(i-1)*spacing - physical_aperture/2, 0, 0];
end

% 静态阵列
array_static = ArrayPlatform(array_pos, 1, 1:num_elements);
array_static = array_static.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

fprintf('阵列: %d元ULA, 间距=%.2fλ, 孔径=%.2fλ\n', ...
    num_elements, spacing/lambda, physical_aperture/lambda);
fprintf('理论分辨率: ~%.1f° (瑞利准则)\n\n', asind(lambda / physical_aperture));

%% ═══════════════════════════════════════════════════════════════════════════
%  测试1：静态阵列单目标
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('测试1：静态阵列 + 单目标 φ方向估计\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');

true_theta = 90;
true_phi = 30;
target_range = 500;

% 目标位置
target_pos = target_range * [cosd(true_phi), sind(true_phi), 0];
target = Target(target_pos, [0,0,0], 1);

fprintf('目标: 距离=%.0fm, θ=90°(固定), φ=%.1f°\n', target_range, true_phi);

% 信号生成
num_snapshots = 32;
snr_db = 20;
t_axis = (0:num_snapshots-1) * 0.01;

sig_gen = SignalGeneratorSimple(radar_params, array_static, {target});
snapshots = sig_gen.generate_snapshots(t_axis, snr_db);

fprintf('快拍: %d个, SNR=%ddB\n', num_snapshots, snr_db);

% 验证信号相位
fprintf('\n信号相位验证:\n');
virtual_pos = array_static.get_mimo_virtual_positions(0);
u = [cosd(true_phi); sind(true_phi); 0];

fprintf('  阵元   实际相位差   理论相位差   误差\n');
phase_errors = zeros(num_elements, 1);
for i = 1:num_elements
    actual_diff = angle(snapshots(i, 1) / snapshots(1, 1));
    
    % 理论相位差推导：
    % 阵元i到目标的距离 ≈ R - pos_i · u（远场近似）
    % 双程相位 = -4π/λ * (R - pos_i · u) = -4πR/λ + 4π/λ * pos_i · u
    % 相对于阵元1的相位差 = 4π/λ * (pos_i - pos_1) · u
    delta_pos = virtual_pos(i, :) - virtual_pos(1, :);
    
    % 信号生成用的是exp(-j*phase)，所以相位差是负的投影
    % 但实际上越远的阵元相位越滞后（负），所以：
    % 理论相位差 = -4π/λ * delta_distance = -4π/λ * (-delta_pos · u) = 4π/λ * delta_pos · u
    % 错了！让我重新推导...
    
    % SignalGeneratorSimple中：
    % phase = 2π * 2 * |target - pos| / λ
    % signal = exp(-j * phase)
    % 
    % 对于远场：|target - pos_i| ≈ R - pos_i · u_target
    % phase_i = 4π * (R - pos_i · u) / λ
    % signal_i = exp(-j * 4π * (R - pos_i · u) / λ)
    %          = exp(-j * 4πR/λ) * exp(j * 4π * pos_i · u / λ)
    %
    % 相对于阵元1：
    % signal_i / signal_1 = exp(j * 4π * (pos_i - pos_1) · u / λ)
    
    theory_diff = 4 * pi / lambda * (delta_pos * u);
    theory_diff = wrapToPi(theory_diff);
    
    error = abs(wrapToPi(actual_diff - theory_diff));
    phase_errors(i) = error;
    fprintf('   %d     %+.4f rad   %+.4f rad   %.4f rad\n', i, actual_diff, theory_diff, error);
end

% 验证相位一致性
if max(phase_errors) < 0.1
    fprintf('  ✅ 信号相位与理论一致\n\n');
else
    fprintf('  ⚠️ 信号相位有误差，但继续测试...\n\n');
end

% DOA估计
phi_search = 0:0.5:90;
spectrum = zeros(size(phi_search));

Rxx = (snapshots * snapshots') / num_snapshots;
[V, D] = eig(Rxx);
[~, idx] = sort(diag(D), 'descend');
V = V(:, idx);
Qn = V(:, 2:end);

for phi_idx = 1:length(phi_search)
    phi = phi_search(phi_idx);
    u_test = [cosd(phi); sind(phi); 0];
    
    % 导向矢量：与信号相位一致
    % signal_i / signal_1 = exp(j * 4π/λ * (pos_i - pos_1) · u)
    % 所以 a_i = exp(j * 4π/λ * pos_i · u)（假设pos_1在原点）
    a = zeros(num_elements, 1);
    for i = 1:num_elements
        % 使用 +j，与信号生成一致
        phase = 4 * pi / lambda * (virtual_pos(i, :) * u_test);
        a(i) = exp(1j * phase);
    end
    
    spectrum(phi_idx) = 1 / abs(a' * (Qn * Qn') * a);
end

[~, peak_idx] = max(spectrum);
est_phi = phi_search(peak_idx);
phi_error = abs(est_phi - true_phi);

fprintf('结果:\n');
fprintf('  真实φ: %.1f°\n', true_phi);
fprintf('  估计φ: %.1f°\n', est_phi);
fprintf('  误差: Δφ=%.2f°\n', phi_error);

if phi_error < 2
    fprintf('  ✅ 测试1通过\n\n');
    test1_pass = true;
else
    fprintf('  ❌ 测试1失败\n\n');
    test1_pass = false;
end

test1_spectrum = spectrum;

%% ═══════════════════════════════════════════════════════════════════════════
%  测试2：静态阵列双目标
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('测试2：静态阵列 + 双目标分辨（间隔20°）\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');

true_phi1 = 20;
true_phi2 = 40;

target1 = Target(target_range * [cosd(true_phi1), sind(true_phi1), 0], [0,0,0], 1);
target2 = Target(target_range * [cosd(true_phi2), sind(true_phi2), 0], [0,0,0], 1);

fprintf('目标1: φ=%.1f°\n', true_phi1);
fprintf('目标2: φ=%.1f°\n', true_phi2);

sig_gen2 = SignalGeneratorSimple(radar_params, array_static, {target1, target2});
snapshots2 = sig_gen2.generate_snapshots(t_axis, snr_db);

Rxx2 = (snapshots2 * snapshots2') / num_snapshots;
[V2, D2] = eig(Rxx2);
[~, idx2] = sort(diag(D2), 'descend');
V2 = V2(:, idx2);
Qn2 = V2(:, 3:end);

spectrum2 = zeros(size(phi_search));
for phi_idx = 1:length(phi_search)
    phi = phi_search(phi_idx);
    u_test = [cosd(phi); sind(phi); 0];
    
    a = zeros(num_elements, 1);
    for i = 1:num_elements
        phase = 4 * pi / lambda * (virtual_pos(i, :) * u_test);
        a(i) = exp(1j * phase);
    end
    
    spectrum2(phi_idx) = 1 / abs(a' * (Qn2 * Qn2') * a);
end

[pks, locs] = findpeaks(spectrum2, 'MinPeakProminence', 0.05*max(spectrum2), ...
                        'SortStr', 'descend', 'NPeaks', 2);

if length(locs) >= 2
    est_phis = sort(phi_search(locs(1:2)));
    error1 = abs(est_phis(1) - true_phi1);
    error2 = abs(est_phis(2) - true_phi2);
    
    fprintf('\n结果:\n');
    fprintf('  真实: φ₁=%.1f°, φ₂=%.1f°\n', true_phi1, true_phi2);
    fprintf('  估计: φ₁=%.1f°, φ₂=%.1f°\n', est_phis(1), est_phis(2));
    
    if error1 < 5 && error2 < 5
        fprintf('  ✅ 测试2通过\n\n');
        test2_pass = true;
    else
        fprintf('  ❌ 测试2失败\n\n');
        test2_pass = false;
    end
else
    fprintf('  ❌ 未检测到两个峰值\n\n');
    test2_pass = false;
    est_phis = [NaN, NaN];
end

test2_spectrum = spectrum2;

%% ═══════════════════════════════════════════════════════════════════════════
%  测试3：运动阵列
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('测试3：运动阵列 + 单目标\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');

v_platform = 5;
array_moving = ArrayPlatform(array_pos, 1, 1:num_elements);
array_moving = array_moving.set_trajectory(@(t) struct('position', [v_platform*t, 0, 0], 'orientation', [0,0,0]));

total_displacement = v_platform * t_axis(end);
synthetic_aperture = physical_aperture + total_displacement;

fprintf('运动: v=%.1fm/s, 平移=%.2fm (%.1fλ)\n', v_platform, total_displacement, total_displacement/lambda);
fprintf('合成孔径: %.2fλ (扩展%.1f倍)\n', synthetic_aperture/lambda, synthetic_aperture/physical_aperture);

target3 = Target(target_pos, [0,0,0], 1);
fprintf('目标: φ=%.1f°\n', true_phi);

sig_gen3 = SignalGeneratorSimple(radar_params, array_moving, {target3});
snapshots3 = sig_gen3.generate_snapshots(t_axis, snr_db);

% 非相干分段MUSIC
num_segments = 4;
snapshots_per_seg = floor(num_snapshots / num_segments);
fprintf('非相干: %d段×%d快拍\n', num_segments, snapshots_per_seg);

spectrum3 = zeros(size(phi_search));

for seg = 1:num_segments
    idx_start = (seg-1)*snapshots_per_seg + 1;
    idx_end = seg * snapshots_per_seg;
    
    snapshots_seg = snapshots3(:, idx_start:idx_end);
    t_center = mean(t_axis(idx_start:idx_end));
    positions_seg = array_moving.get_mimo_virtual_positions(t_center);
    
    Rxx_seg = (snapshots_seg * snapshots_seg') / snapshots_per_seg;
    [V_seg, D_seg] = eig(Rxx_seg);
    [~, idx_seg] = sort(diag(D_seg), 'descend');
    V_seg = V_seg(:, idx_seg);
    Qn_seg = V_seg(:, 2:end);
    
    for phi_idx = 1:length(phi_search)
        phi = phi_search(phi_idx);
        u_test = [cosd(phi); sind(phi); 0];
        
        a = zeros(num_elements, 1);
        for i = 1:num_elements
            phase = 4 * pi / lambda * (positions_seg(i, :) * u_test);
            a(i) = exp(1j * phase);
        end
        
        spectrum3(phi_idx) = spectrum3(phi_idx) + 1 / abs(a' * (Qn_seg * Qn_seg') * a);
    end
end

spectrum3 = spectrum3 / num_segments;

[~, peak_idx3] = max(spectrum3);
est_phi3 = phi_search(peak_idx3);
phi_error3 = abs(est_phi3 - true_phi);

fprintf('\n结果:\n');
fprintf('  真实φ: %.1f°\n', true_phi);
fprintf('  估计φ: %.1f°\n', est_phi3);
fprintf('  误差: Δφ=%.2f°\n', phi_error3);

if phi_error3 < 3
    fprintf('  ✅ 测试3通过\n\n');
    test3_pass = true;
else
    fprintf('  ❌ 测试3失败\n\n');
    test3_pass = false;
end

%% 绘图
figure('Position', [100, 100, 1400, 400]);

subplot(1,3,1);
plot(phi_search, 10*log10(test1_spectrum/max(test1_spectrum)), 'b-', 'LineWidth', 2);
hold on;
xline(true_phi, 'r--', 'LineWidth', 2, 'Label', sprintf('真实%.0f°', true_phi));
xline(est_phi, 'g--', 'LineWidth', 2, 'Label', sprintf('估计%.1f°', est_phi));
xlabel('φ (°)'); ylabel('MUSIC谱 (dB)');
title(sprintf('测试1: 静态单目标\n误差:%.2f° %s', phi_error, ternary(test1_pass, '✓', '✗')));
grid on; xlim([0, 90]); ylim([-30, 0]);

subplot(1,3,2);
plot(phi_search, 10*log10(test2_spectrum/max(test2_spectrum)), 'b-', 'LineWidth', 2);
hold on;
xline(true_phi1, 'r--', 'LineWidth', 2, 'Label', sprintf('%.0f°', true_phi1));
xline(true_phi2, 'r--', 'LineWidth', 2, 'Label', sprintf('%.0f°', true_phi2));
xlabel('φ (°)'); ylabel('MUSIC谱 (dB)');
title(sprintf('测试2: 双目标 %s', ternary(test2_pass, '✓', '✗')));
grid on; xlim([0, 90]); ylim([-30, 0]);

subplot(1,3,3);
plot(phi_search, 10*log10(spectrum3/max(spectrum3)), 'b-', 'LineWidth', 2);
hold on;
xline(true_phi, 'r--', 'LineWidth', 2, 'Label', sprintf('真实%.0f°', true_phi));
xline(est_phi3, 'g--', 'LineWidth', 2, 'Label', sprintf('估计%.1f°', est_phi3));
xlabel('φ (°)'); ylabel('MUSIC谱 (dB)');
title(sprintf('测试3: 运动阵列\n孔径%.1fλ, 误差%.2f° %s', synthetic_aperture/lambda, phi_error3, ternary(test3_pass, '✓', '✗')));
grid on; xlim([0, 90]); ylim([-30, 0]);

sgtitle(sprintf('工具类可靠性验证\n%d元ULA, SNR=%ddB', num_elements, snr_db), 'FontSize', 14);

saveas(gcf, 'verify_tools_reliability.png');
fprintf('图片已保存: verify_tools_reliability.png\n\n');

%% 总结
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('                        验证结果总结                              \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

fprintf('测试1 - 静态单目标:  %s (误差%.2f°)\n', ternary(test1_pass, '✅ 通过', '❌ 失败'), phi_error);
fprintf('测试2 - 静态双目标:  %s\n', ternary(test2_pass, '✅ 通过', '❌ 失败'));
fprintf('测试3 - 运动单目标:  %s (误差%.2f°)\n', ternary(test3_pass, '✅ 通过', '❌ 失败'), phi_error3);

fprintf('\n');
if test1_pass && test2_pass && test3_pass
    fprintf('🎉 所有测试通过！\n');
else
    fprintf('⚠️  部分测试失败\n');
end

fprintf('═══════════════════════════════════════════════════════════════════\n');

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
