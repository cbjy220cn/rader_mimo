%% 诊断脚本：逐步检查信号生成和DOA估计管道
% 目的：找出为什么φ=30°的目标被估计为φ=62.5°

clear; clc; close all;

fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('诊断：信号生成 → DOA估计 管道检查\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

%% 基础参数
c = physconst('LightSpeed');
f0 = 3e9;
lambda = c / f0;

radar_params.fc = f0;
radar_params.c = c;
radar_params.lambda = lambda;
radar_params.fs = 36100;
radar_params.T_chirp = 10e-3;
radar_params.slope = 5e12;
radar_params.BW = 50e6;
radar_params.num_samples = 361;
radar_params.range_res = c / (2 * radar_params.BW);

%% Step 1: 检查阵列几何
fprintf('【Step 1】阵列几何检查\n');
fprintf('─────────────────────────────────────────────────────────────────\n');

num_elements = 8;
spacing = 0.5 * lambda;

array_pos = zeros(num_elements, 3);
for i = 1:num_elements
    array_pos(i, :) = [(i-1)*spacing - (num_elements-1)*spacing/2, 0, 0];
end

fprintf('阵元位置 (单位: λ):\n');
for i = 1:num_elements
    fprintf('  阵元%d: [%.2f, %.2f, %.2f]\n', i, array_pos(i,:)/lambda);
end

array_static = ArrayPlatform(array_pos, 1, 1:num_elements);
array_static = array_static.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

virtual_pos = array_static.get_mimo_virtual_positions(0);
fprintf('\n虚拟阵元位置 (SIMO 1发8收):\n');
for i = 1:size(virtual_pos, 1)
    fprintf('  虚拟阵元%d: [%.4f, %.4f, %.4f] m\n', i, virtual_pos(i,:));
end

%% Step 2: 检查目标方向矢量
fprintf('\n【Step 2】目标方向矢量检查\n');
fprintf('─────────────────────────────────────────────────────────────────\n');

true_theta = 90;  % deg
true_phi = 30;    % deg
target_range = 500; % m

% 球坐标 → 笛卡尔坐标的方向矢量
u_target = [sind(true_theta)*cosd(true_phi); ...
            sind(true_theta)*sind(true_phi); ...
            cosd(true_theta)];
        
fprintf('目标角度: θ=%.1f°, φ=%.1f°\n', true_theta, true_phi);
fprintf('方向矢量 u = [%.4f, %.4f, %.4f]\n', u_target(1), u_target(2), u_target(3));

% 目标位置
target_pos = target_range * u_target';
fprintf('目标位置: [%.2f, %.2f, %.2f] m\n', target_pos);

target = Target(target_pos, [0,0,0], 1);

%% Step 3: 检查理论相位 vs SignalGenerator相位
fprintf('\n【Step 3】相位计算检查\n');
fprintf('─────────────────────────────────────────────────────────────────\n');

% 理论相位（雷达双程：4π/λ）
fprintf('理论相位 (4π/λ * r·u):\n');
for i = 1:num_elements
    % 虚拟阵元位置（对于SIMO，虚拟位置 = (Tx + Rx)/2）
    virt_pos = virtual_pos(i, :);
    phase_theory = 4 * pi / lambda * (virt_pos * u_target);
    fprintf('  阵元%d: 相位 = %.4f rad (%.2f°)\n', i, phase_theory, rad2deg(phase_theory));
end

% 用SignalGenerator生成信号
num_snapshots = 32;
snr_db = 50;  % 高SNR
t_axis = (0:num_snapshots-1) * radar_params.T_chirp;

sig_gen = SignalGenerator(radar_params, array_static, {target});
snapshots = sig_gen.generate_snapshots(t_axis, snr_db);

fprintf('\n实际信号相位 (第1个快拍):\n');
for i = 1:num_elements
    phase_actual = angle(snapshots(i, 1));
    fprintf('  阵元%d: 相位 = %.4f rad (%.2f°)\n', i, phase_actual, rad2deg(phase_actual));
end

% 计算相位差（相对于第1个阵元）
fprintf('\n相位差（相对于阵元1）:\n');
fprintf('         理论        实际        差异\n');
for i = 1:num_elements
    virt_pos = virtual_pos(i, :);
    virt_pos_ref = virtual_pos(1, :);
    phase_diff_theory = 4 * pi / lambda * ((virt_pos - virt_pos_ref) * u_target);
    phase_diff_actual = angle(snapshots(i, 1) / snapshots(1, 1));
    fprintf('  阵元%d: %+.4f rad  %+.4f rad  %.4f rad\n', i, ...
        phase_diff_theory, phase_diff_actual, phase_diff_actual - phase_diff_theory);
end

%% Step 4: 检查DoaEstimator的导向矢量
fprintf('\n【Step 4】DoaEstimator导向矢量检查\n');
fprintf('─────────────────────────────────────────────────────────────────\n');

estimator = DoaEstimator(array_static, radar_params);

% 检查DoaEstimator在真实方向上的导向矢量
search_grid_point.theta = true_theta;
search_grid_point.phi = true_phi;

% 调用estimate_gmusic获取导向矢量（通过u_debug参数）
u_debug = [sind(true_theta)*cosd(true_phi); ...
           sind(true_theta)*sind(true_phi); ...
           cosd(true_theta)];
       
[~, A_u] = estimator.estimate_gmusic(snapshots, t_axis, 1, search_grid_point, u_debug, virtual_pos);

fprintf('DoaEstimator导向矢量 (θ=%.1f°, φ=%.1f°):\n', true_theta, true_phi);
if ~isempty(A_u)
    for i = 1:num_elements
        fprintf('  阵元%d: %.4f + %.4fi (相位=%.4f rad)\n', i, ...
            real(A_u(i,1)), imag(A_u(i,1)), angle(A_u(i,1)));
    end
else
    fprintf('  无法获取导向矢量\n');
end

%% Step 5: 手动计算MUSIC谱
fprintf('\n【Step 5】手动计算MUSIC谱\n');
fprintf('─────────────────────────────────────────────────────────────────\n');

% 计算协方差矩阵
Rxx = (snapshots * snapshots') / num_snapshots;

% 特征分解
[V, D] = eig(Rxx);
[eigenvalues, idx] = sort(diag(D), 'descend');
V = V(:, idx);

fprintf('特征值:\n');
for i = 1:num_elements
    fprintf('  λ%d = %.4e\n', i, eigenvalues(i));
end

% 信号子空间和噪声子空间
num_targets = 1;
Qs = V(:, 1:num_targets);
Qn = V(:, (num_targets+1):end);

fprintf('\n手动搜索φ从0°到90°:\n');
phi_search = 0:5:90;
music_spectrum = zeros(size(phi_search));

for phi_idx = 1:length(phi_search)
    phi = phi_search(phi_idx);
    theta = 90;  % 固定
    
    % 方向矢量
    u = [sind(theta)*cosd(phi); sind(theta)*sind(phi); cosd(theta)];
    
    % 导向矢量（手动计算）
    a = zeros(num_elements, 1);
    for i = 1:num_elements
        phase = 4 * pi / lambda * (virtual_pos(i, :) * u);
        a(i) = exp(1j * phase);
    end
    
    % MUSIC谱
    music_spectrum(phi_idx) = 1 / abs(a' * (Qn * Qn') * a);
end

% 归一化
music_spectrum_db = 10*log10(music_spectrum / max(music_spectrum));

fprintf('  φ(°)   谱值(dB)\n');
for phi_idx = 1:length(phi_search)
    marker = '';
    if phi_search(phi_idx) == true_phi
        marker = ' <-- 真实目标';
    end
    if music_spectrum_db(phi_idx) == 0
        marker = [marker ' <-- 峰值'];
    end
    fprintf('  %5.1f  %+7.2f dB%s\n', phi_search(phi_idx), music_spectrum_db(phi_idx), marker);
end

[~, peak_idx] = max(music_spectrum);
est_phi_manual = phi_search(peak_idx);
fprintf('\n手动计算峰值: φ=%.1f°\n', est_phi_manual);
fprintf('误差: %.1f°\n', abs(est_phi_manual - true_phi));

%% Step 6: 可视化诊断
figure('Position', [100, 100, 1200, 500]);

% 子图1：信号相位
subplot(1,3,1);
phase_actual = angle(snapshots(:, 1));
phase_theory = zeros(num_elements, 1);
for i = 1:num_elements
    phase_theory(i) = 4 * pi / lambda * (virtual_pos(i, :) * u_target);
end
phase_theory = wrapToPi(phase_theory);  % 包装到[-π, π]

bar([phase_actual, phase_theory]);
xlabel('阵元索引');
ylabel('相位 (rad)');
legend('实际', '理论');
title('信号相位对比');
grid on;

% 子图2：手动MUSIC谱
subplot(1,3,2);
plot(phi_search, music_spectrum_db, 'b-o', 'LineWidth', 2);
hold on;
xline(true_phi, 'r--', 'LineWidth', 2, 'Label', sprintf('真实φ=%.0f°', true_phi));
xline(est_phi_manual, 'g--', 'LineWidth', 2, 'Label', sprintf('峰值φ=%.0f°', est_phi_manual));
xlabel('φ (°)');
ylabel('MUSIC谱 (dB)');
title('手动计算MUSIC谱');
grid on;
xlim([0, 90]);

% 子图3：检查信号一致性
subplot(1,3,3);
% 计算信号的空间频率
signal_vec = snapshots(:, 1);
signal_phase_diff = diff(angle(signal_vec));
signal_phase_diff = wrapToPi(signal_phase_diff);

% 理论空间频率
d = spacing;  % 阵元间距
spatial_freq_theory = 4 * pi * d / lambda * cosd(true_phi);
spatial_freq_actual = mean(signal_phase_diff);

fprintf('\n【诊断】空间频率检查:\n');
fprintf('  理论空间频率（相邻阵元相位差）: %.4f rad\n', spatial_freq_theory);
fprintf('  实际空间频率（相邻阵元相位差）: %.4f rad\n', spatial_freq_actual);

% 反推实际φ
phi_inferred = acosd(spatial_freq_actual * lambda / (4 * pi * d));
fprintf('  从实际相位反推的φ: %.1f°\n', phi_inferred);

bar([spatial_freq_theory, spatial_freq_actual]);
set(gca, 'XTickLabel', {'理论', '实际'});
ylabel('空间频率 (rad/阵元)');
title(sprintf('空间频率对比\n反推φ=%.1f°', phi_inferred));
grid on;

sgtitle('信号生成与DOA估计诊断', 'FontSize', 14);
saveas(gcf, 'diagnose_pipeline.png');
fprintf('\n图片已保存: diagnose_pipeline.png\n');

%% 总结
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('诊断总结\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('目标: θ=%.1f°, φ=%.1f°\n', true_theta, true_phi);
fprintf('手动MUSIC峰值: φ=%.1f°\n', est_phi_manual);
fprintf('从信号相位反推: φ=%.1f°\n', phi_inferred);

if abs(phi_inferred - true_phi) > 5
    fprintf('\n⚠️  问题在于【信号生成】：实际相位与理论不符！\n');
    fprintf('   需要检查 SignalGenerator.m 中的相位计算\n');
else
    fprintf('\n⚠️  问题在于【DOA估计】：信号正确但估计错误\n');
    fprintf('   需要检查 DoaEstimator.m 中的导向矢量计算\n');
end




