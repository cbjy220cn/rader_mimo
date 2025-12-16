%% 快速测试：验证合成孔径MUSIC是否有效
% 测试内容：对比静态MUSIC和运动MUSIC的主瓣宽度
% 预期：运动阵列主瓣应该更窄（孔径大16倍）
clear; clc; close all;

addpath('asset');

fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('              合成孔径MUSIC快速验证测试                           \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

%% 参数设置
c = physconst('LightSpeed');
fc = 3e9;
lambda = c / fc;
d = lambda / 2;

radar_params.fc = fc;
radar_params.lambda = lambda;
radar_params.c = c;
radar_params.T_chirp = 10e-3;
radar_params.fs = 100e6;

% 观测参数
T_obs = 0.5;
num_snapshots = 64;
t_axis = linspace(0, T_obs, num_snapshots);
v_linear = 5;  % 5 m/s

% 目标
target_phi = 30;
target_theta = 90;  % 水平面
target_range = 500;
snr_db = 20;

fprintf('【参数】\n');
fprintf('  波长: %.2f cm\n', lambda * 100);
fprintf('  观测时间: %.1f s, 快拍数: %d\n', T_obs, num_snapshots);
fprintf('  平移速度: %.1f m/s → 平移距离: %.2f m\n', v_linear, v_linear * T_obs);
fprintf('  目标: φ=%.0f°, SNR=%ddB\n\n', target_phi, snr_db);

%% 创建4元小阵列
num_elements = 4;
x_pos = ((0:num_elements-1) - (num_elements-1)/2) * d;
elements = [x_pos', zeros(num_elements, 1), zeros(num_elements, 1)];
array = ArrayPlatform(elements, 1, 1:num_elements);

% 静态孔径
static_aperture = (num_elements - 1) * d;
fprintf('【阵列】\n');
fprintf('  阵元数: %d\n', num_elements);
fprintf('  静态孔径: %.2f cm (%.1f λ)\n', static_aperture * 100, static_aperture / lambda);
fprintf('  合成孔径: %.2f cm (%.1f λ)\n', (v_linear * T_obs) * 100, (v_linear * T_obs) / lambda);

%% 创建目标
target_pos = target_range * [cosd(target_phi)*sind(target_theta), ...
                              sind(target_phi)*sind(target_theta), ...
                              cosd(target_theta)];
target = Target(target_pos, [0,0,0], 1);

%% 搜索网格
phi_search = 0:0.1:60;

%% 测试1: 静态阵列 MUSIC
fprintf('\n【测试1: 静态阵列 MUSIC】\n');
array.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));
sig_gen = SignalGeneratorSimple(radar_params, array, {target});

rng(0);
snapshots_static = sig_gen.generate_snapshots(t_axis, snr_db);

% 静态MUSIC
positions_static = array.get_mimo_virtual_positions(0);
spectrum_static = music_1d(snapshots_static, positions_static, phi_search, lambda, 1);
spectrum_static_db = 10*log10(spectrum_static / max(spectrum_static));

% 找峰值和主瓣宽度
[~, peak_idx] = max(spectrum_static_db);
est_phi_static = phi_search(peak_idx);
beamwidth_static = calc_beamwidth(spectrum_static_db, phi_search);

fprintf('  估计角度: %.2f° (真值: %.0f°, 误差: %.2f°)\n', est_phi_static, target_phi, abs(est_phi_static - target_phi));
fprintf('  主瓣宽度: %.2f°\n', beamwidth_static);

%% 测试2: 运动阵列 时间平滑MUSIC
fprintf('\n【测试2: 运动阵列 时间平滑MUSIC】\n');
array.set_trajectory(@(t) struct('position', [0, v_linear*t, 0], 'orientation', [0,0,0]));
sig_gen = SignalGeneratorSimple(radar_params, array, {target});

rng(0);
snapshots_motion = sig_gen.generate_snapshots(t_axis, snr_db);

% 合成孔径 时间平滑MUSIC
estimator = DoaEstimatorSynthetic(array, radar_params);
est_options.search_mode = '1d';
[spectrum_motion, peaks_motion, info] = estimator.estimate(snapshots_motion, t_axis, struct('phi', phi_search), 1, est_options);
spectrum_motion_db = 10*log10(spectrum_motion / max(spectrum_motion));

est_phi_motion = peaks_motion.phi(1);
beamwidth_motion = calc_beamwidth(spectrum_motion_db, phi_search);

fprintf('  虚拟阵元数: %d\n', info.num_virtual);
fprintf('  合成孔径: %.1f λ\n', info.synthetic_aperture.total_lambda);
fprintf('  估计角度: %.2f° (真值: %.0f°, 误差: %.2f°)\n', est_phi_motion, target_phi, abs(est_phi_motion - target_phi));
fprintf('  主瓣宽度: %.2f°\n', beamwidth_motion);

%% 结果对比
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                        结果对比                                   \n');
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('              | 静态MUSIC | 运动MUSIC | 改善  \n');
fprintf('  ------------|-----------|-----------|--------\n');
fprintf('  孔径        | %.1f λ    | %.1f λ    | %.1fx\n', ...
    static_aperture/lambda, info.synthetic_aperture.total_lambda, ...
    info.synthetic_aperture.total_lambda / (static_aperture/lambda));
fprintf('  主瓣宽度    | %.2f°     | %.2f°     | %.1fx\n', ...
    beamwidth_static, beamwidth_motion, beamwidth_static / max(beamwidth_motion, 0.01));
fprintf('  估计误差    | %.2f°     | %.2f°     | -\n', ...
    abs(est_phi_static - target_phi), abs(est_phi_motion - target_phi));

%% 判断测试是否通过
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
if beamwidth_motion < beamwidth_static
    fprintf('✅ 测试通过！运动阵列MUSIC主瓣更窄 (%.2f° → %.2f°)\n', beamwidth_static, beamwidth_motion);
    fprintf('   时间平滑MUSIC有效！合成孔径发挥作用！\n');
else
    fprintf('⚠️ 主瓣宽度未改善，但检查估计精度和谱形状\n');
    fprintf('   静态: %.2f°, 运动: %.2f°\n', beamwidth_static, beamwidth_motion);
end
fprintf('═══════════════════════════════════════════════════════════════════\n');

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
title(sprintf('静态MUSIC (孔径=%.1fλ, 主瓣=%.2f°)', static_aperture/lambda, beamwidth_static));
grid on;
ylim([-30, 5]);
xlim([10, 50]);
legend({'MUSIC谱', '真实角度'}, 'Location', 'south');

subplot(1, 2, 2);
plot(phi_search, spectrum_motion_db, 'b-', 'LineWidth', 2);
hold on;
xline(target_phi, 'r--', 'LineWidth', 2);
yline(-3, 'g--', '-3dB', 'LineWidth', 1);
hold off;
xlabel('方位角 φ (°)');
ylabel('归一化功率 (dB)');
title(sprintf('运动MUSIC (孔径=%.1fλ, 主瓣=%.2f°)', info.synthetic_aperture.total_lambda, beamwidth_motion));
grid on;
ylim([-30, 5]);
xlim([10, 50]);
legend({'时间平滑MUSIC谱', '真实角度'}, 'Location', 'south');

sgtitle('合成孔径MUSIC验证：主瓣宽度对比', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('\n图已生成，请查看主瓣对比！\n');

%% 辅助函数
function spectrum = music_1d(snapshots, positions, phi_search, lambda, num_targets)
    num_elements = size(snapshots, 1);
    num_snapshots = size(snapshots, 2);
    
    Rxx = (snapshots * snapshots') / num_snapshots;
    [V, D] = eig(Rxx);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    Qn = V(:, (num_targets+1):end);
    
    spectrum = zeros(size(phi_search));
    for phi_idx = 1:length(phi_search)
        phi = phi_search(phi_idx);
        u = [cosd(phi); sind(phi); 0];
        
        a = zeros(num_elements, 1);
        for i = 1:num_elements
            phase = 4 * pi / lambda * (positions(i, :) * u);
            a(i) = exp(-1j * phase);  % 与信号模型一致
        end
        
        spectrum(phi_idx) = 1 / abs(a' * (Qn * Qn') * a);
    end
end

function beamwidth = calc_beamwidth(spectrum_db, angle_axis)
    [~, peak_idx] = max(spectrum_db);
    
    left_3db = find(spectrum_db(1:peak_idx) < -3, 1, 'last');
    right_3db = peak_idx + find(spectrum_db(peak_idx:end) < -3, 1, 'first') - 1;
    
    if isempty(left_3db), left_3db = 1; end
    if isempty(right_3db), right_3db = length(angle_axis); end
    
    beamwidth = angle_axis(right_3db) - angle_axis(left_3db);
    if beamwidth <= 0
        beamwidth = angle_axis(2) - angle_axis(1);
    end
end
