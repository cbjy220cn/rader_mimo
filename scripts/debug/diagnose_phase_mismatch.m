%% 诊断：SignalGenerator vs DoaEstimator 相位一致性

clear; clc;

c = 3e8;
f0 = 3e9;
lambda = c / f0;

% 目标（正对ULA）
target_theta = 90;
target_phi = 0;
target_range = 1000;

% 简单ULA：4元，0.5λ间距
num_elements = 4;
spacing = 0.5 * lambda;
array_pos = [(-1.5:1:1.5)' * spacing, zeros(4,2)];

fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║        相位一致性诊断                                  ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

fprintf('阵列配置: %d元ULA, 间距%.2fλ\n', num_elements, spacing/lambda);
fprintf('目标: θ=%.0f°, φ=%.0f°, R=%.0fm\n\n', target_theta, target_phi, target_range);

%% 1. 生成雷达信号（单快拍）
radar_params.c = c;
radar_params.fc = f0;
radar_params.lambda = lambda;
radar_params.BW = 100e6;
radar_params.range_res = c / (2 * 100e6);
radar_params.fs = 36100;
radar_params.T_chirp = 10e-3;
radar_params.slope = 5e12;
radar_params.num_samples = 361;

platform = ArrayPlatform(array_pos, 1, 1:num_elements);
platform = platform.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

target_pos = [target_range * sind(target_theta) * cosd(target_phi), ...
             target_range * sind(target_theta) * sind(target_phi), ...
             target_range * cosd(target_theta)];
targets = {Target(target_pos, [0,0,0], 1)};

sig_gen = SignalGenerator(radar_params, platform, targets);
t_axis = 0;  % 单快拍
snapshots = sig_gen.generate_snapshots(t_axis, inf);  % 无噪声

fprintf('[步骤1] SignalGenerator生成的信号:\n');
fprintf('  维度: %d × %d\n', size(snapshots, 1), size(snapshots, 2));
fprintf('  信号向量:\n');
for i = 1:num_elements
    fprintf('    元素%d: 幅度=%.4f, 相位=%.2f°\n', i, abs(snapshots(i)), angle(snapshots(i))*180/pi);
end
fprintf('\n');

%% 2. 计算理论导向矢量
u_true = [sind(target_theta)*cosd(target_phi);
          sind(target_theta)*sind(target_phi);
          cosd(target_theta)];

fprintf('[步骤2] 目标方向矢量: u=[%.3f, %.3f, %.3f]\n\n', u_true);

% 方法A: 使用DoaEstimator的build_steering_matrix
estimator = DoaEstimator(platform, radar_params);
A_method_A = estimator.build_steering_matrix(t_axis, u_true);

fprintf('[步骤3] 方法A (DoaEstimator.build_steering_matrix):\n');
fprintf('  相位公式: 4π/λ * (positions · u)\n');
fprintf('  导向矢量:\n');
for i = 1:num_elements
    fprintf('    元素%d: 幅度=%.4f, 相位=%.2f°\n', i, abs(A_method_A(i)), angle(A_method_A(i))*180/pi);
end
fprintf('\n');

% 方法B: 手动计算（双程4π）
positions = platform.get_mimo_virtual_positions(0);
phase_B = 4 * pi / lambda * (positions * u_true);
A_method_B = exp(1j * phase_B);

fprintf('[步骤4] 方法B (手动计算, 4π):\n');
fprintf('  相位公式: 4π/λ * (positions · u)\n');
fprintf('  导向矢量:\n');
for i = 1:num_elements
    fprintf('    元素%d: 幅度=%.4f, 相位=%.2f°\n', i, abs(A_method_B(i)), angle(A_method_B(i))*180/pi);
end
fprintf('\n');

% 方法C: 单程2π（错误，用于对比）
phase_C = 2 * pi / lambda * (positions * u_true);
A_method_C = exp(1j * phase_C);

fprintf('[步骤5] 方法C (错误示例, 2π):\n');
fprintf('  相位公式: 2π/λ * (positions · u)\n');
fprintf('  导向矢量:\n');
for i = 1:num_elements
    fprintf('    元素%d: 幅度=%.4f, 相位=%.2f°\n', i, abs(A_method_C(i)), angle(A_method_C(i))*180/pi);
end
fprintf('\n');

%% 3. 计算相关性
% 确保都是列向量
signal_vec = snapshots(:);
A_A_vec = A_method_A(:);
A_B_vec = A_method_B(:);
A_C_vec = A_method_C(:);

signal_normalized = signal_vec / norm(signal_vec);
A_A_norm = A_A_vec / norm(A_A_vec);
A_B_norm = A_B_vec / norm(A_B_vec);
A_C_norm = A_C_vec / norm(A_C_vec);

corr_A = abs(signal_normalized' * A_A_norm);
corr_B = abs(signal_normalized' * A_B_norm);
corr_C = abs(signal_normalized' * A_C_norm);

fprintf('═══════════════════════════════════════════════════════\n');
fprintf('相关性对比（期望≈1.0）:\n');
fprintf('─────────────────────────────────────────────────────\n');
fprintf('  信号 vs 方法A (DoaEstimator): %.6f %s\n', corr_A, ternary(corr_A>0.99, '✓', '❌'));
fprintf('  信号 vs 方法B (4π手动):      %.6f %s\n', corr_B, ternary(corr_B>0.99, '✓', '❌'));
fprintf('  信号 vs 方法C (2π错误):      %.6f %s\n', corr_C, ternary(corr_C>0.99, '✓', '❌'));
fprintf('═══════════════════════════════════════════════════════\n\n');

%% 4. 相位差分析
% 确保形状一致
signal_vec = snapshots(:);
A_A_vec = A_method_A(:);
A_B_vec = A_method_B(:);
A_C_vec = A_method_C(:);

phase_diff_A = angle(signal_vec ./ A_A_vec) * 180/pi;
phase_diff_B = angle(signal_vec ./ A_B_vec) * 180/pi;
phase_diff_C = angle(signal_vec ./ A_C_vec) * 180/pi;

fprintf('相位差（所有元素应相同）:\n');
fprintf('─────────────────────────────────────────────────────\n');
fprintf('元素 | 方法A | 方法B | 方法C\n');
for i = 1:num_elements
    fprintf(' %d   | %6.1f° | %6.1f° | %6.1f°\n', i, phase_diff_A(i), phase_diff_B(i), phase_diff_C(i));
end
fprintf('标准差| %6.2f° | %6.2f° | %6.2f°\n', std(phase_diff_A), std(phase_diff_B), std(phase_diff_C));
fprintf('═══════════════════════════════════════════════════════\n\n');

%% 5. 可视化
figure('Position', [100, 100, 1400, 500]);

subplot(1,3,1);
signal_vec = snapshots(:);
A_A_vec = A_method_A(:);
A_B_vec = A_method_B(:);
A_C_vec = A_method_C(:);

plot(1:num_elements, angle(signal_vec)*180/pi, 'ko-', 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', '实际信号');
hold on;
plot(1:num_elements, angle(A_A_vec)*180/pi, 'rx--', 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', '方法A (DoaEst)');
plot(1:num_elements, angle(A_B_vec)*180/pi, 'bs--', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '方法B (4π)');
plot(1:num_elements, angle(A_C_vec)*180/pi, 'g^--', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '方法C (2π)');
xlabel('阵元索引');
ylabel('相位 (°)');
title('相位对比');
legend('Location', 'best');
grid on;

subplot(1,3,2);
bar([corr_A, corr_B, corr_C]);
set(gca, 'XTickLabel', {'方法A', '方法B', '方法C'});
ylabel('相关系数');
title('相关性对比（期望=1.0）');
ylim([0, 1.1]);
grid on;
hold on;
plot(xlim, [0.99, 0.99], 'r--', 'LineWidth', 2);
text(1.5, 0.99, '阈值=0.99', 'VerticalAlignment', 'bottom');

subplot(1,3,3);
boxplot([phase_diff_A, phase_diff_B, phase_diff_C], 'Labels', {'方法A', '方法B', '方法C'});
ylabel('相位差 (°)');
title('相位差分布（应接近常数）');
grid on;

sgtitle('SignalGenerator vs DoaEstimator 相位一致性诊断', 'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, 'diagnose_phase_mismatch.png');
fprintf('✓ 图片已保存: diagnose_phase_mismatch.png\n\n');

%% 结论
fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║  诊断结论                                              ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

if corr_A > 0.99
    fprintf('✅ DoaEstimator的导向矢量与信号完全匹配！\n');
    fprintf('   问题不在相位公式。\n\n');
elseif corr_B > 0.99
    fprintf('⚠️  手动4π公式匹配，但DoaEstimator不匹配！\n');
    fprintf('   需要检查DoaEstimator.build_steering_matrix实现。\n\n');
elseif corr_C > 0.99
    fprintf('❌ 只有2π公式匹配！\n');
    fprintf('   SignalGenerator可能使用了单程相位，需要修正。\n\n');
else
    fprintf('❌ 所有方法都不匹配！\n');
    fprintf('   存在更深层的问题，需要进一步调查。\n\n');
    fprintf('可能原因:\n');
    fprintf('  1. 虚拟阵元位置计算错误\n');
    fprintf('  2. SignalGenerator中有额外的相位偏移\n');
    fprintf('  3. 目标方向矢量计算错误\n\n');
end

function out = ternary(cond, true_val, false_val)
    if cond
        out = true_val;
    else
        out = false_val;
    end
end

