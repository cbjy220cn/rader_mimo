%% 测试CA-CFAR峰值检测 vs 传统方法
% 验证CA-CFAR在多目标场景下的性能提升

clear; clc; close all;
fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║      CA-CFAR峰值检测 vs 传统方法对比测试              ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

%% 雷达参数
c = physconst('LightSpeed');
f0 = 3e9;
lambda = c/f0;

radar_params.fc = f0;
radar_params.c = c;
radar_params.lambda = lambda;
radar_params.fs = 36100;
radar_params.T_chirp = 10e-3;
radar_params.slope = 5e12;
radar_params.BW = 50e6;
radar_params.num_samples = 361;
radar_params.range_res = c / (2 * radar_params.BW);

fprintf('📡 雷达参数: f₀=%.2f GHz\n\n', f0/1e9);

%% 测试场景：近距离双目标
target_range = 600;
theta_true = 30;
phi1_true = 60;
phi2_true = 62;  % 2度间隔

fprintf('🎯 测试场景: 近距离双目标\n');
fprintf('   目标1: θ=%.1f°, φ=%.1f°\n', theta_true, phi1_true);
fprintf('   目标2: θ=%.1f°, φ=%.1f°\n', theta_true, phi2_true);
fprintf('   间隔: %.1f°\n\n', phi2_true - phi1_true);

target1_pos = [target_range * sind(theta_true) * cosd(phi1_true), ...
               target_range * sind(theta_true) * sind(phi1_true), ...
               target_range * cosd(theta_true)];
target2_pos = [target_range * sind(theta_true) * cosd(phi2_true), ...
               target_range * sind(theta_true) * sind(phi2_true), ...
               target_range * cosd(theta_true)];
targets = {Target(target1_pos, [0,0,0], 1), Target(target2_pos, [0,0,0], 1)};

%% 创建旋转阵列
num_elements = 8;
R_rx = 0.05;
theta_rx = linspace(0, 2*pi, num_elements+1); 
theta_rx(end) = [];
rx_elements = zeros(num_elements, 3);
for i = 1:num_elements
    rx_elements(i,:) = [R_rx*cos(theta_rx(i)), R_rx*sin(theta_rx(i)), 0];
end

num_snapshots = 64;
t_axis = (0:num_snapshots-1) * radar_params.T_chirp;
omega_dps = 360 / t_axis(end);

array_rotating = ArrayPlatform(rx_elements, 1, 1:num_elements);
array_rotating = array_rotating.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]));

%% 生成信号并计算MUSIC谱
fprintf('生成信号和计算MUSIC谱...\n');
sig_gen = SignalGenerator(radar_params, array_rotating, targets);
snapshots = sig_gen.generate_snapshots(t_axis, inf);

estimator = DoaEstimatorIncoherent(array_rotating, radar_params);

% 使用智能搜索获取完整谱
smart_grid.coarse_res = 3.0;
smart_grid.fine_res = 0.2;
smart_grid.roi_margin = 12.0;
smart_grid.theta_range = [0, 90];
smart_grid.phi_range = [0, 180];

[spectrum, search_grid] = smart_doa_search(estimator, snapshots, t_axis, 2, smart_grid, ...
    struct('verbose', false, 'weighting', 'uniform'));

fprintf('✓ MUSIC谱计算完成\n\n');

%% 方法1: 传统峰值检测
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('方法1: 传统峰值检测（全局最大值）\n');
fprintf('═══════════════════════════════════════════════════════\n');

[theta_trad, phi_trad, vals_trad] = DoaEstimatorIncoherent.find_peaks(spectrum, search_grid, 2);

fprintf('找到 %d 个峰值:\n', length(theta_trad));
for i = 1:length(theta_trad)
    error1 = abs(theta_trad(i) - theta_true);
    error2 = min(abs(phi_trad(i) - phi1_true), abs(phi_trad(i) - phi2_true));
    fprintf('  峰值%d: θ=%.2f° (误差%.2f°), φ=%.2f° (误差%.2f°), 幅度=%.2e\n', ...
        i, theta_trad(i), error1, phi_trad(i), error2, vals_trad(i));
end

% 计算峰值间隔
if length(phi_trad) >= 2
    actual_sep_trad = abs(phi_trad(1) - phi_trad(2));
    fprintf('  实际间隔: %.2f° (理论: %.1f°)\n', actual_sep_trad, phi2_true - phi1_true);
else
    fprintf('  ⚠️ 只检测到1个峰值，未能分辨双目标\n');
end
fprintf('\n');

%% 方法2: CA-CFAR峰值检测
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('方法2: CA-CFAR峰值检测（自适应阈值）\n');
fprintf('═══════════════════════════════════════════════════════\n');

cfar_options.numGuard = 3;          % 保护单元（考虑峰值宽度）
cfar_options.numTrain = 6;          % 训练单元
cfar_options.P_fa = 1e-4;           % 虚警概率
cfar_options.SNR_offset_dB = -15;   % SNR偏移（宽松一点）
cfar_options.min_separation = 1.5;  % 最小峰值间隔（度）

[theta_cfar, phi_cfar, vals_cfar, cfar_mask] = find_peaks_cfar(spectrum, search_grid, 2, cfar_options);

fprintf('找到 %d 个峰值:\n', length(theta_cfar));
for i = 1:length(theta_cfar)
    error1 = abs(theta_cfar(i) - theta_true);
    error2 = min(abs(phi_cfar(i) - phi1_true), abs(phi_cfar(i) - phi2_true));
    fprintf('  峰值%d: θ=%.2f° (误差%.2f°), φ=%.2f° (误差%.2f°), 幅度=%.2e\n', ...
        i, theta_cfar(i), error1, phi_cfar(i), error2, vals_cfar(i));
end

% 计算峰值间隔
if length(phi_cfar) >= 2
    actual_sep_cfar = abs(phi_cfar(1) - phi_cfar(2));
    fprintf('  实际间隔: %.2f° (理论: %.1f°)\n', actual_sep_cfar, phi2_true - phi1_true);
else
    fprintf('  ⚠️ 只检测到1个峰值，未能分辨双目标\n');
end
fprintf('\n');

%% 对比分析
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('对比分析\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

fprintf('📊 峰值检测数量:\n');
fprintf('   传统方法: %d个\n', length(theta_trad));
fprintf('   CA-CFAR:  %d个\n\n', length(theta_cfar));

if length(phi_trad) >= 2 && length(phi_cfar) >= 2
    fprintf('📏 峰值间隔精度:\n');
    fprintf('   真实间隔: %.1f°\n', phi2_true - phi1_true);
    fprintf('   传统方法: %.2f° (误差: %.2f°)\n', actual_sep_trad, abs(actual_sep_trad - (phi2_true - phi1_true)));
    fprintf('   CA-CFAR:  %.2f° (误差: %.2f°)\n\n', actual_sep_cfar, abs(actual_sep_cfar - (phi2_true - phi1_true)));
end

fprintf('🎯 角度估计精度:\n');
fprintf('   传统方法平均误差: φ=%.2f°\n', mean(abs([phi_trad(1) - phi1_true, phi_trad(end) - phi2_true])));
fprintf('   CA-CFAR平均误差:  φ=%.2f°\n\n', mean(abs([phi_cfar(1) - phi1_true, phi_cfar(end) - phi2_true])));

%% 可视化对比
figure('Position', [50, 50, 1600, 900]);

% 图1: 原始MUSIC谱
subplot(2,3,1);
surf(search_grid.phi, search_grid.theta, spectrum / max(spectrum(:)));
shading interp; view(2); colorbar;
caxis([0, 1]);
hold on;
plot(phi1_true, theta_true, 'r+', 'MarkerSize', 20, 'LineWidth', 3);
plot(phi2_true, theta_true, 'r+', 'MarkerSize', 20, 'LineWidth', 3);
xlabel('Phi (°)');
ylabel('Theta (°)');
title('原始MUSIC谱（归一化）');
xlim([50, 70]);
ylim([20, 40]);

% 图2: 传统方法检测结果
subplot(2,3,2);
surf(search_grid.phi, search_grid.theta, spectrum / max(spectrum(:)));
shading interp; view(2); colorbar;
caxis([0, 1]);
hold on;
plot(phi1_true, theta_true, 'r+', 'MarkerSize', 20, 'LineWidth', 3);
plot(phi2_true, theta_true, 'r+', 'MarkerSize', 20, 'LineWidth', 3);
for i = 1:length(theta_trad)
    plot(phi_trad(i), theta_trad(i), 'go', 'MarkerSize', 15, 'LineWidth', 2);
end
xlabel('Phi (°)');
ylabel('Theta (°)');
title(sprintf('传统方法（检测%d个峰）', length(theta_trad)));
xlim([50, 70]);
ylim([20, 40]);

% 图3: CA-CFAR检测结果
subplot(2,3,3);
surf(search_grid.phi, search_grid.theta, spectrum / max(spectrum(:)));
shading interp; view(2); colorbar;
caxis([0, 1]);
hold on;
plot(phi1_true, theta_true, 'r+', 'MarkerSize', 20, 'LineWidth', 3);
plot(phi2_true, theta_true, 'r+', 'MarkerSize', 20, 'LineWidth', 3);
for i = 1:length(theta_cfar)
    plot(phi_cfar(i), theta_cfar(i), 'mo', 'MarkerSize', 15, 'LineWidth', 2);
end
xlabel('Phi (°)');
ylabel('Theta (°)');
title(sprintf('CA-CFAR方法（检测%d个峰）', length(theta_cfar)));
xlim([50, 70]);
ylim([20, 40]);

% 图4: CA-CFAR检测掩码
subplot(2,3,4);
imagesc(search_grid.phi, search_grid.theta, cfar_mask);
colorbar;
colormap(gca, 'gray');
hold on;
plot(phi1_true, theta_true, 'r+', 'MarkerSize', 20, 'LineWidth', 3);
plot(phi2_true, theta_true, 'r+', 'MarkerSize', 20, 'LineWidth', 3);
xlabel('Phi (°)');
ylabel('Theta (°)');
title('CA-CFAR检测掩码');
xlim([50, 70]);
ylim([20, 40]);

% 图5: 1D切片对比
subplot(2,3,[5,6]);
[~, theta_idx] = min(abs(search_grid.theta - theta_true));
slice = spectrum(theta_idx, :);
slice_db = 10*log10(slice / max(slice));

plot(search_grid.phi, slice_db, 'k-', 'LineWidth', 2, 'DisplayName', 'MUSIC谱'); hold on;
xline(phi1_true, 'r--', 'LineWidth', 1.5, 'DisplayName', '真实目标');
xline(phi2_true, 'r--', 'LineWidth', 1.5, 'HandleVisibility', 'off');

% 标记传统方法检测的峰值
for i = 1:length(phi_trad)
    if abs(theta_trad(i) - theta_true) < 5
        plot(phi_trad(i), slice_db(find(abs(search_grid.phi - phi_trad(i)) < 0.3, 1)), ...
            'go', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', sprintf('传统峰%d', i));
    end
end

% 标记CA-CFAR检测的峰值
for i = 1:length(phi_cfar)
    if abs(theta_cfar(i) - theta_true) < 5
        plot(phi_cfar(i), slice_db(find(abs(search_grid.phi - phi_cfar(i)) < 0.3, 1)), ...
            'ms', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', sprintf('CFAR峰%d', i));
    end
end

xlim([50, 70]);
ylim([-40, 5]);
grid on;
xlabel('Phi (°)', 'FontSize', 12);
ylabel('归一化幅度 (dB)', 'FontSize', 12);
title(sprintf('1D切片对比 (theta=%.1f°)', theta_true), 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 10);

sgtitle('CA-CFAR峰值检测对比分析', 'FontSize', 16, 'FontWeight', 'bold');

fprintf('✓ 图表生成完成\n\n');

%% 结论
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('✅ 测试结论\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

if length(theta_cfar) > length(theta_trad)
    fprintf('🎉 CA-CFAR方法表现更好！\n');
    fprintf('   检测到更多峰值，更准确地分辨了近距离多目标\n\n');
elseif length(theta_cfar) == length(theta_trad)
    fprintf('✓ CA-CFAR方法与传统方法检测数量相同\n');
    if length(phi_cfar) >= 2 && abs(actual_sep_cfar - (phi2_true - phi1_true)) < abs(actual_sep_trad - (phi2_true - phi1_true))
        fprintf('   但CA-CFAR的峰值间隔更准确\n\n');
    else
        fprintf('   两种方法性能接近\n\n');
    end
else
    fprintf('⚠️ CA-CFAR检测峰值较少\n');
    fprintf('   可能需要调整CFAR参数（降低P_fa或SNR_offset）\n\n');
end

fprintf('建议:\n');
fprintf('  • 对于近距离多目标（<5°间隔），推荐使用CA-CFAR\n');
fprintf('  • 对于单目标或远距离多目标，传统方法已足够\n');
fprintf('  • CA-CFAR参数需要根据场景调优\n\n');

