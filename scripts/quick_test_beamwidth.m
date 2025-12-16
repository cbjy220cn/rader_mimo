%% 快速测试：主瓣宽度验证
% 检查孔径扩展后主瓣是否真的变窄
clear; clc; close all;
addpath('asset');

fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('              主瓣宽度验证测试                                    \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

%% 参数
c = physconst('LightSpeed');
fc = 3e9; lambda = c / fc; d = lambda / 2;
radar_params.fc = fc; radar_params.lambda = lambda; radar_params.c = c;
radar_params.T_chirp = 10e-3; radar_params.fs = 100e6;

T_obs = 0.5; num_snapshots = 64;
t_axis = linspace(0, T_obs, num_snapshots);
v_linear = 5;

% 目标
target_phi = 30; target_theta = 90; target_range = 500;
snr_db = 20;

% 高分辨率搜索网格（关键！）
phi_search = 20:0.02:40;  % 0.02°步进，足够细

%% 4元阵列测试
num_elements = 4;
x_pos = ((0:num_elements-1) - (num_elements-1)/2) * d;
elements = [x_pos', zeros(num_elements, 1), zeros(num_elements, 1)];
array = ArrayPlatform(elements, 1, 1:num_elements);

target_pos = target_range * [cosd(target_phi), sind(target_phi), 0];
target = Target(target_pos, [0,0,0], 1);

fprintf('【4元阵列测试】\n');
fprintf('  静态孔径: %.1f λ\n', (num_elements-1)*d/lambda);
fprintf('  合成孔径: %.1f λ\n', v_linear*T_obs/lambda);

%% 测试1: 静态
fprintf('\n【静态阵列】\n');
array.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));
sig_gen = SignalGeneratorSimple(radar_params, array, {target});
rng(0); snapshots = sig_gen.generate_snapshots(t_axis, snr_db);
positions = array.get_mimo_virtual_positions(0);

% 标准MUSIC
Rxx = (snapshots * snapshots') / num_snapshots;
[V, D] = eig(Rxx);
[~, idx] = sort(diag(D), 'descend');
V = V(:, idx);
Qn = V(:, 2:end);

spectrum_static = zeros(size(phi_search));
for i = 1:length(phi_search)
    phi = phi_search(i);
    u = [cosd(phi); sind(phi); 0];
    a = exp(-1j * 4 * pi / lambda * (positions * u));
    spectrum_static(i) = 1 / abs(a' * (Qn * Qn') * a);
end
spectrum_static_db = 10*log10(spectrum_static / max(spectrum_static));

[~, peak_idx] = max(spectrum_static_db);
beamwidth_static = calc_3db_width(spectrum_static_db, phi_search);
fprintf('  峰值位置: %.2f°\n', phi_search(peak_idx));
fprintf('  主瓣宽度: %.2f°\n', beamwidth_static);

%% 测试2: y平移
fprintf('\n【运动阵列 (y平移)】\n');
array.set_trajectory(@(t) struct('position', [0, v_linear*t, 0], 'orientation', [0,0,0]));
sig_gen = SignalGeneratorSimple(radar_params, array, {target});
rng(0); snapshots = sig_gen.generate_snapshots(t_axis, snr_db);

estimator = DoaEstimatorSynthetic(array, radar_params);
est_options.search_mode = '1d';
[spectrum_motion, peaks, info] = estimator.estimate(snapshots, t_axis, struct('phi', phi_search), 1, est_options);
spectrum_motion_db = 10*log10(spectrum_motion / max(spectrum_motion));

[~, peak_idx] = max(spectrum_motion_db);
beamwidth_motion = calc_3db_width(spectrum_motion_db, phi_search);
fprintf('  虚拟阵元数: %d\n', info.num_virtual);
fprintf('  峰值位置: %.2f°\n', phi_search(peak_idx));
fprintf('  主瓣宽度: %.2f°\n', beamwidth_motion);

%% 结果对比
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                        结果对比                                   \n');
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('              | 静态  | 运动  | 改善  \n');
fprintf('  ------------|-------|-------|--------\n');
fprintf('  孔径(λ)     | %.1f  | %.1f  | %.1fx\n', ...
    (num_elements-1)*d/lambda, info.synthetic_aperture.total_lambda, ...
    info.synthetic_aperture.total_lambda / ((num_elements-1)*d/lambda));
fprintf('  主瓣宽度    | %.2f° | %.2f° | %.1fx\n', ...
    beamwidth_static, beamwidth_motion, beamwidth_static / max(beamwidth_motion, 0.01));

if beamwidth_motion < beamwidth_static * 0.5
    fprintf('\n✅ 主瓣宽度显著变窄！合成孔径有效！\n');
else
    fprintf('\n⚠️ 主瓣宽度改善不明显，检查算法\n');
end

%% 绘图
figure('Position', [100, 100, 1000, 400], 'Color', 'white');

subplot(1, 2, 1);
plot(phi_search, spectrum_static_db, 'k-', 'LineWidth', 2);
hold on;
xline(target_phi, 'r--', 'LineWidth', 2);
yline(-3, 'g--', '-3dB', 'LineWidth', 1);
hold off;
xlabel('方位角 φ (°)');
ylabel('归一化功率 (dB)');
title(sprintf('静态MUSIC (孔径=%.1fλ, 主瓣=%.2f°)', (num_elements-1)*d/lambda, beamwidth_static));
grid on; ylim([-40, 5]); xlim([20, 40]);

subplot(1, 2, 2);
plot(phi_search, spectrum_motion_db, 'b-', 'LineWidth', 2);
hold on;
xline(target_phi, 'r--', 'LineWidth', 2);
yline(-3, 'g--', '-3dB', 'LineWidth', 1);
hold off;
xlabel('方位角 φ (°)');
ylabel('归一化功率 (dB)');
title(sprintf('运动MUSIC (孔径=%.1fλ, 主瓣=%.2f°)', info.synthetic_aperture.total_lambda, beamwidth_motion));
grid on; ylim([-40, 5]); xlim([20, 40]);

sgtitle('主瓣宽度对比验证', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('\n图已生成！\n');

%% 辅助函数
function width = calc_3db_width(spectrum_db, angle_axis)
    [~, peak_idx] = max(spectrum_db);
    
    % 左边-3dB点
    left_idx = peak_idx;
    while left_idx > 1 && spectrum_db(left_idx) > -3
        left_idx = left_idx - 1;
    end
    
    % 右边-3dB点
    right_idx = peak_idx;
    while right_idx < length(spectrum_db) && spectrum_db(right_idx) > -3
        right_idx = right_idx + 1;
    end
    
    width = angle_axis(right_idx) - angle_axis(left_idx);
    if width <= 0
        width = 0.1;  % 默认最小值
    end
end

