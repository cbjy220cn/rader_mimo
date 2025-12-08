%% 最小化DOA测试：绕过SignalGenerator，直接生成理想平面波

clear; clc; close all;

c = 3e8;
f0 = 3e9;
lambda = c / f0;

fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║     最小化DOA测试（理想平面波）                        ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

%% 配置
num_elements = 16;
spacing = 0.5 * lambda;
target_theta = 90;  % 正对ULA
target_phi = 0;
snr_db = 20;

% ULA阵列
array_pos = zeros(num_elements, 3);
for i = 1:num_elements
    array_pos(i, 1) = (i - (num_elements+1)/2) * spacing;
end

fprintf('配置:\n');
fprintf('  阵元: %d元ULA, 间距%.2fλ\n', num_elements, spacing/lambda);
fprintf('  目标: θ=%.0f°, φ=%.0f°\n', target_theta, target_phi);
fprintf('  SNR: %.0f dB\n\n', snr_db);

%% 1. 手动生成理想平面波（绕过SignalGenerator）
fprintf('[步骤1] 生成理想平面波信号（多快拍）...\n');

num_snapshots = 16;  % ⚠️ 多快拍避免秩缺陷

u = [sind(target_theta)*cosd(target_phi);
     sind(target_theta)*sind(target_phi);
     cosd(target_theta)];

% 使用4π相位（雷达双程）
phase = 4 * pi / lambda * (array_pos * u);
signal_ideal = exp(1j * phase);

% 复制为多快拍（每个快拍独立噪声）
snapshots = zeros(num_elements, num_snapshots);
signal_power = mean(abs(signal_ideal).^2);
noise_power = signal_power / (10^(snr_db/10));

for k = 1:num_snapshots
    noise_k = (randn(size(signal_ideal)) + 1j*randn(size(signal_ideal))) * sqrt(noise_power/2);
    snapshots(:, k) = signal_ideal + noise_k;
end

fprintf('  信号维度: %d × %d\n', size(snapshots, 1), size(snapshots, 2));
fprintf('  信号功率: %.2e\n', signal_power);
fprintf('  噪声功率: %.2e\n', noise_power);
fprintf('  实际SNR: %.1f dB\n\n', 10*log10(signal_power / noise_power));

%% 2. MUSIC算法（使用DoaEstimator_DEBUG）
fprintf('[步骤2] 运行MUSIC算法...\n');

% 创建虚拟platform（只用于传递参数）
radar_params.c = c;
radar_params.fc = f0;
radar_params.lambda = lambda;

platform = ArrayPlatform(zeros(1,3), 1, 1);  % 虚拟的，不会被用到
platform = platform.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

estimator = DoaEstimator_DEBUG(platform, radar_params);

% 搜索网格（局部）
search_grid.theta = (target_theta-10):0.2:(target_theta+10);
search_grid.phi = (target_phi-10):0.2:(target_phi+10);

t_axis = zeros(1, num_snapshots);  % 多快拍（静态阵列，时间无关）

% ⚠️ 关键：直接传入阵列位置，绕过ArrayPlatform的虚拟阵元计算
spectrum = estimator.estimate_gmusic(snapshots, t_axis, 1, search_grid, array_pos);

fprintf('\n');

%% 3. 找峰值
[max_val, max_idx] = max(spectrum(:));
[i_max, j_max] = ind2sub(size(spectrum), max_idx);
theta_est = search_grid.theta(i_max);
phi_est = search_grid.phi(j_max);

fprintf('═══════════════════════════════════════════════════════\n');
fprintf('结果:\n');
fprintf('─────────────────────────────────────────────────────\n');
fprintf('  真实角度: θ=%.1f°, φ=%.1f°\n', target_theta, target_phi);
fprintf('  估计角度: θ=%.1f°, φ=%.1f°\n', theta_est, phi_est);
fprintf('  角度误差: Δθ=%.1f°, Δφ=%.1f°\n', ...
    abs(theta_est - target_theta), abs(phi_est - target_phi));
fprintf('  谱动态范围: %.2f\n', max_val / min(spectrum(:)));
fprintf('═══════════════════════════════════════════════════════\n\n');

%% 4. 可视化
figure('Position', [100, 100, 1400, 400]);

subplot(1,3,1);
imagesc(search_grid.phi, search_grid.theta, 10*log10(spectrum));
axis xy; colorbar;
xlabel('Phi (°)'); ylabel('Theta (°)');
title('MUSIC谱 (dB)');
hold on;
plot(target_phi, target_theta, 'r+', 'MarkerSize', 20, 'LineWidth', 3);
plot(phi_est, theta_est, 'wo', 'MarkerSize', 15, 'LineWidth', 2);

subplot(1,3,2);
[~, phi_idx] = min(abs(search_grid.phi - target_phi));
plot(search_grid.theta, 10*log10(spectrum(:, phi_idx)), 'b-', 'LineWidth', 2);
xlabel('Theta (°)'); ylabel('谱值 (dB)');
title(sprintf('Theta切片 (φ=%.0f°)', target_phi));
grid on;
hold on;
plot([target_theta, target_theta], ylim, 'r--', 'LineWidth', 2);

subplot(1,3,3);
plot(1:num_elements, angle(signal_ideal)*180/pi, 'ro-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '理想信号');
hold on;
plot(1:num_elements, angle(snapshots(:,1))*180/pi, 'bx--', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', '快拍1（含噪声）');
xlabel('阵元索引'); ylabel('相位 (°)');
title(sprintf('信号相位（%d快拍）', num_snapshots));
legend; grid on;

sgtitle('最小化DOA测试：理想平面波', 'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, 'test_minimal_doa.png');
fprintf('✓ 图片已保存: test_minimal_doa.png\n\n');

%% 5. 结论
fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║  测试结论                                              ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

theta_error = abs(theta_est - target_theta);
dynamic_range = max_val / min(spectrum(:));

if theta_error < 1.0 && dynamic_range > 100
    fprintf('✅ MUSIC算法完美工作！\n');
    fprintf('   - 角度估计精度: %.1f° ✓\n', theta_error);
    fprintf('   - 谱动态范围: %.0f ✓\n', dynamic_range);
    fprintf('\n💡 关键结论:\n');
    fprintf('   ✅ DoaEstimator_DEBUG实现正确（4π相位，归一化）\n');
    fprintf('   ✅ 多快拍（16个）避免了秩缺陷\n');
    fprintf('   ❌ 问题确定在SignalGenerator的信号生成！\n\n');
    fprintf('   SignalGenerator的问题:\n');
    fprintf('   - 随机baseband_signals破坏了相位结构\n');
    fprintf('   - 复杂的几何/RVP相位处理可能有误\n');
    fprintf('   - 需要修复或简化信号生成逻辑\n\n');
elseif theta_error < 1.0
    fprintf('⚠️  角度估计正确，但动态范围偏小（%.1f）\n', dynamic_range);
    fprintf('   可能原因: SNR太低或快拍数不足\n\n');
else
    fprintf('❌ 角度估计误差过大（%.1f°）\n', theta_error);
    fprintf('   DoaEstimator可能仍有问题\n\n');
end

