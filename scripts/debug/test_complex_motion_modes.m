%% ═══════════════════════════════════════════════════════════════════════════
%  复杂运动模式测试：绕边缘旋转、旋转+平移
%  验证合成虚拟阵列方法对不同运动的适用性
%% ═══════════════════════════════════════════════════════════════════════════
clear; clc; close all;

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║     复杂运动模式 DOA 性能测试                                  ║\n');
fprintf('║  绕边缘旋转 / 旋转+平移 / 纯平移 对比                         ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

%% 基本参数
fc = 3e9;
c = physconst('LightSpeed');
lambda = c / fc;
radar_params = struct('fc', fc, 'lambda', lambda);

num_elements = 8;
d = lambda / 2;
physical_aperture = (num_elements - 1) * d;

% 目标参数
target_phi = 30;  % 目标方位角
target_range = 500;
snr = 20;

% 观测时间和快拍数
T_obs = 0.5;  % 观测时间
num_snapshots = 64;
t_axis = linspace(0, T_obs, num_snapshots);

% 运动参数
v_linear = 5;  % 平移速度 m/s
omega_deg = 90;  % 总旋转角度（度）

fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('基本配置\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('阵列: %d元ULA, 间距=%.2fλ, 物理孔径=%.2fλ\n', num_elements, d/lambda, physical_aperture/lambda);
fprintf('目标: φ=%.1f°, 距离=%.0fm, SNR=%ddB\n', target_phi, target_range, snr);
fprintf('观测: T=%.2fs, %d快拍\n', T_obs, num_snapshots);
fprintf('运动: 平移速度=%.1fm/s, 旋转角度=%.1f°\n', v_linear, omega_deg);
fprintf('\n');

%% 定义四种运动模式
motion_modes = struct();

% 模式1: 纯平移 (沿y轴)
motion_modes(1).name = '纯平移';
motion_modes(1).trajectory = @(t) struct(...
    'position', [0, v_linear * t, 0], ...
    'orientation', [0, 0, 0]);
motion_modes(1).description = sprintf('沿y平移 %.2fm', v_linear * T_obs);

% 模式2: 绕中心旋转
motion_modes(2).name = '绕中心旋转';
motion_modes(2).trajectory = @(t) struct(...
    'position', [0, 0, 0], ...
    'orientation', [0, 0, omega_deg * t / T_obs]);
motion_modes(2).description = sprintf('绕中心旋转 %.1f°', omega_deg);

% 模式3: 绕边缘旋转 (旋转中心在阵列左端)
edge_offset = physical_aperture / 2;  % 旋转中心到阵列中心的距离
motion_modes(3).name = '绕边缘旋转';
motion_modes(3).trajectory = @(t) create_edge_rotation_trajectory(t, T_obs, omega_deg, edge_offset);
motion_modes(3).description = sprintf('绕左端点旋转 %.1f°', omega_deg);

% 模式4: 旋转 + 平移
motion_modes(4).name = '旋转+平移';
motion_modes(4).trajectory = @(t) struct(...
    'position', [0, v_linear * t, 0], ...
    'orientation', [0, 0, omega_deg * t / T_obs]);
motion_modes(4).description = sprintf('平移%.2fm + 旋转%.1f°', v_linear*T_obs, omega_deg);

%% 对每种运动模式进行测试
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('运动模式测试\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

results = struct();
phi_search = 0:0.5:90;
search_grid = struct('phi', phi_search);

for mode_idx = 1:length(motion_modes)
    mode = motion_modes(mode_idx);
    fprintf('【模式%d】%s: %s\n', mode_idx, mode.name, mode.description);
    
    % 创建阵列
    array = ArrayPlatform.create_ula(num_elements, d);
    array.set_trajectory(mode.trajectory);
    
    % 计算合成孔径（预览）
    [aperture_x, aperture_y, all_positions] = calc_synthetic_aperture(array, t_axis);
    total_aperture = sqrt(aperture_x^2 + aperture_y^2);
    fprintf('   合成孔径: x=%.2fλ, y=%.2fλ, 总=%.2fλ (扩展%.1f倍)\n', ...
        aperture_x/lambda, aperture_y/lambda, total_aperture/lambda, total_aperture/physical_aperture);
    
    % 生成信号 - 注意参数顺序: (radar_params, array_platform, targets)
    % Target需要笛卡尔坐标: (initial_position, velocity, rcs)
    % 将球坐标转换为笛卡尔坐标 (theta=90°表示水平面)
    target_pos = target_range * [cosd(target_phi), sind(target_phi), 0];
    target = Target(target_pos, [0, 0, 0], 1);  % 静止目标，RCS=1
    sig_gen = SignalGeneratorSimple(radar_params, array, {target});
    snapshots = sig_gen.generate_snapshots(t_axis, snr);
    
    % 使用新的DoaEstimatorSynthetic类进行DOA估计
    estimator = DoaEstimatorSynthetic(array, radar_params);
    [spectrum, peaks, info] = estimator.estimate(snapshots, t_axis, search_grid, 1);
    
    % 计算主瓣宽度
    beamwidth = estimator.estimate_beamwidth(spectrum, phi_search);
    
    % 存储结果
    results(mode_idx).name = mode.name;
    results(mode_idx).spectrum = spectrum;
    results(mode_idx).est_phi = peaks.phi(1);
    results(mode_idx).error = abs(peaks.phi(1) - target_phi);
    results(mode_idx).beamwidth = beamwidth;
    results(mode_idx).aperture = info.synthetic_aperture;
    results(mode_idx).num_virtual = info.num_virtual;
    results(mode_idx).all_positions = all_positions;
    
    fprintf('   虚拟阵元: %d个\n', info.num_virtual);
    fprintf('   估计: φ=%.1f° (误差%.1f°), 主瓣宽度=%.1f°\n\n', ...
        peaks.phi(1), results(mode_idx).error, beamwidth);
end

%% 静态阵列作为对照
fprintf('【对照】静态阵列\n');
array_static = ArrayPlatform.create_ula(num_elements, d);
target_pos_static = target_range * [cosd(target_phi), sind(target_phi), 0];
target_static = Target(target_pos_static, [0, 0, 0], 1);
sig_gen_static = SignalGeneratorSimple(radar_params, array_static, {target_static});
snapshots_static = sig_gen_static.generate_snapshots(t_axis, snr);

% 静态阵列使用标准MUSIC
positions_static = array_static.get_mimo_virtual_positions(0);
spectrum_static = music_standard(snapshots_static, positions_static, phi_search, lambda, 1);
beamwidth_static = calc_beamwidth(spectrum_static, phi_search);
[~, peak_idx] = max(spectrum_static);
est_phi_static = phi_search(peak_idx);

fprintf('   估计: φ=%.1f° (误差%.1f°), 主瓣宽度=%.1f°\n\n', ...
    est_phi_static, abs(est_phi_static - target_phi), beamwidth_static);

%% SNR扫描实验 - 展示运动阵列在低SNR下的优势
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('SNR扫描实验 - 估计精度对比\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

snr_range = -20:5:20;  % 更宽的SNR范围，包含更低的SNR
num_trials = 50;       % 更多蒙特卡洛次数

% 使用更细的搜索网格
phi_search_fine = 0:0.1:90;
search_grid_fine = struct('phi', phi_search_fine);

% 只测试有代表性的模式：纯平移、绕中心旋转、静态
test_modes = [1, 2];  % 纯平移、绕中心旋转
mode_names = {'纯平移', '绕中心旋转', '静态'};

rmse_results = zeros(length(test_modes)+1, length(snr_range));

fprintf('进度: ');
for snr_idx = 1:length(snr_range)
    snr_test = snr_range(snr_idx);
    fprintf('%ddB ', snr_test);
    
    errors_modes = zeros(length(test_modes)+1, num_trials);
    
    for trial = 1:num_trials
        % 每次试验使用略微不同的目标角度（避免落在格点上）
        rng(trial * 1000 + snr_idx);
        target_phi_trial = target_phi + (rand() - 0.5) * 1;  % 30° ± 0.5°
        
        % 测试各运动模式
        for mode_idx = 1:length(test_modes)
            mode = motion_modes(test_modes(mode_idx));
            
            array_test = ArrayPlatform.create_ula(num_elements, d);
            array_test.set_trajectory(mode.trajectory);
            
            target_pos_test = target_range * [cosd(target_phi_trial), sind(target_phi_trial), 0];
            target_test = Target(target_pos_test, [0, 0, 0], 1);
            sig_gen_test = SignalGeneratorSimple(radar_params, array_test, {target_test});
            
            snapshots_test = sig_gen_test.generate_snapshots(t_axis, snr_test);
            
            estimator_test = DoaEstimatorSynthetic(array_test, radar_params);
            [~, peaks_test, ~] = estimator_test.estimate(snapshots_test, t_axis, search_grid_fine, 1);
            
            errors_modes(mode_idx, trial) = peaks_test.phi(1) - target_phi_trial;
        end
        
        % 静态阵列
        array_static_test = ArrayPlatform.create_ula(num_elements, d);
        target_pos_test = target_range * [cosd(target_phi_trial), sind(target_phi_trial), 0];
        target_test = Target(target_pos_test, [0, 0, 0], 1);
        sig_gen_static_test = SignalGeneratorSimple(radar_params, array_static_test, {target_test});
        
        snapshots_static_test = sig_gen_static_test.generate_snapshots(t_axis, snr_test);
        
        positions_static_test = array_static_test.get_mimo_virtual_positions(0);
        spectrum_static_test = music_standard(snapshots_static_test, positions_static_test, phi_search_fine, lambda, 1);
        [~, peak_idx_test] = max(spectrum_static_test);
        
        errors_modes(end, trial) = phi_search_fine(peak_idx_test) - target_phi_trial;
    end
    
    % 计算RMSE
    for mode_idx = 1:size(errors_modes, 1)
        rmse_results(mode_idx, snr_idx) = sqrt(mean(errors_modes(mode_idx, :).^2));
    end
end
fprintf('\n\n');

% 打印RMSE结果
fprintf('SNR(dB) | 纯平移RMSE | 绕中心RMSE | 静态RMSE | 平移优势\n');
fprintf('--------|------------|------------|----------|----------\n');
for snr_idx = 1:length(snr_range)
    advantage = rmse_results(end, snr_idx) / max(rmse_results(1, snr_idx), 0.01);
    fprintf('%6d  | %9.2f° | %9.2f° | %8.2f° | %.1fx\n', ...
        snr_range(snr_idx), rmse_results(1, snr_idx), ...
        rmse_results(2, snr_idx), rmse_results(end, snr_idx), advantage);
end

%% 绘图
figure('Position', [100, 100, 1600, 900]);

% 1. SNR vs RMSE 曲线 (最重要的对比图)
subplot(2,3,1);
hold on;
plot(snr_range, rmse_results(1, :), 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', '纯平移(合成孔径)');
plot(snr_range, rmse_results(2, :), 'r-s', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', '绕中心旋转');
plot(snr_range, rmse_results(end, :), 'k--^', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', '静态阵列');
xlabel('SNR (dB)');
ylabel('RMSE (度)');
title('估计精度 vs SNR');
legend('Location', 'northeast');
grid on;
set(gca, 'YScale', 'log');
ylim([0.01, 100]);

% 2. MUSIC谱对比 (高SNR)
subplot(2,3,2);
hold on;
colors = {'b', 'r', 'g', 'm'};
for mode_idx = 1:length(motion_modes)
    spec_db = 10*log10(results(mode_idx).spectrum / max(results(mode_idx).spectrum));
    plot(phi_search, spec_db, colors{mode_idx}, 'LineWidth', 1.5, ...
        'DisplayName', results(mode_idx).name);
end
spec_static_db = 10*log10(spectrum_static / max(spectrum_static));
plot(phi_search, spec_static_db, 'k--', 'LineWidth', 2, 'DisplayName', '静态');
xline(target_phi, 'r:', 'LineWidth', 1.5, 'DisplayName', '真实');
xlabel('φ (度)');
ylabel('MUSIC谱 (dB)');
title(sprintf('MUSIC谱对比 (SNR=%ddB)', snr));
legend('Location', 'northeast', 'FontSize', 7);
grid on;
ylim([-30, 5]);
xlim([0, 90]);

% 3. 主瓣宽度和孔径对比
subplot(2,3,3);
names_short = {'平移', '中心转', '边缘转', '转+移', '静态'};
beamwidths = [results.beamwidth, beamwidth_static];
apertures = zeros(1, length(results)+1);
for i = 1:length(results)
    apertures(i) = results(i).aperture.total_lambda;
end
apertures(end) = physical_aperture/lambda;

yyaxis left;
bar(1:5, beamwidths, 0.4, 'FaceColor', [0.2 0.6 0.8]);
ylabel('主瓣宽度 (度)');
ylim([0, max(beamwidths)*1.3]);

yyaxis right;
hold on;
plot(1:5, apertures, 'ro-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
ylabel('合成孔径 (λ)');

set(gca, 'XTick', 1:5, 'XTickLabel', names_short);
title('主瓣宽度 vs 孔径');
grid on;

% 4. 纯平移虚拟阵元分布
subplot(2,3,4);
pos = results(1).all_positions;
scatter(pos(:,1)/lambda, pos(:,2)/lambda, 8, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('x (λ)');
ylabel('y (λ)');
title(sprintf('纯平移 (孔径%.1fλ)', results(1).aperture.total_lambda));
axis equal;
grid on;

% 5. 绕中心旋转虚拟阵元分布
subplot(2,3,5);
pos = results(2).all_positions;
scatter(pos(:,1)/lambda, pos(:,2)/lambda, 8, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('x (λ)');
ylabel('y (λ)');
title(sprintf('绕中心旋转 (孔径%.1fλ)', results(2).aperture.total_lambda));
axis equal;
grid on;

% 6. 低SNR时RMSE改善倍数
subplot(2,3,6);
% 计算相对于静态的改善倍数（避免除零）
eps_val = 0.01;  % 最小值防止除零
improvement_translation = rmse_results(end, :) ./ max(rmse_results(1, :), eps_val);
improvement_rotation = rmse_results(end, :) ./ max(rmse_results(2, :), eps_val);
% 限制最大值
improvement_translation = min(improvement_translation, 50);
improvement_rotation = min(improvement_rotation, 50);

hold on;
plot(snr_range, improvement_translation, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, ...
    'MarkerFaceColor', 'b', 'DisplayName', '纯平移');
plot(snr_range, improvement_rotation, 'r-s', 'LineWidth', 2, 'MarkerSize', 6, ...
    'MarkerFaceColor', 'r', 'DisplayName', '绕中心旋转');
yline(1, 'k--', 'LineWidth', 1.5, 'DisplayName', '无改善');
xlabel('SNR (dB)');
ylabel('RMSE改善倍数');
title('相对静态阵列的RMSE改善');
legend('Location', 'northeast');
grid on;
ylim([0, max([improvement_translation, improvement_rotation, 2])*1.2]);

sgtitle('复杂运动模式 DOA 性能对比', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, 'test_complex_motion_modes.png');
fprintf('图片已保存: test_complex_motion_modes.png\n');

%% 结果总结
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                        数据总结                                   \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% 直接输出完整数据表
fprintf('\n【完整RMSE数据表】\n');
fprintf('SNR(dB) | 纯平移    | 绕中心旋转 | 静态阵列  | 平移/静态 | 旋转/静态\n');
fprintf('--------|-----------|-----------|-----------|-----------|----------\n');
for snr_idx = 1:length(snr_range)
    ratio1 = rmse_results(end, snr_idx) / max(rmse_results(1, snr_idx), 0.001);
    ratio2 = rmse_results(end, snr_idx) / max(rmse_results(2, snr_idx), 0.001);
    fprintf('%6d  | %8.2f° | %8.2f° | %8.2f° | %8.1fx | %8.1fx\n', ...
        snr_range(snr_idx), rmse_results(1, snr_idx), ...
        rmse_results(2, snr_idx), rmse_results(end, snr_idx), ...
        min(ratio1, 999), min(ratio2, 999));
end

fprintf('\n【孔径与分辨率】\n');
fprintf('模式          | 虚拟阵元 | 合成孔径 | 主瓣宽度\n');
fprintf('--------------|----------|----------|----------\n');
for mode_idx = 1:length(motion_modes)
    fprintf('%-12s  | %6d   | %6.1fλ  | %6.1f°\n', ...
        results(mode_idx).name, results(mode_idx).num_virtual, ...
        results(mode_idx).aperture.total_lambda, results(mode_idx).beamwidth);
end
fprintf('%-12s  | %6d   | %6.1fλ  | %6.1f°\n', ...
    '静态', num_elements, physical_aperture/lambda, beamwidth_static);

fprintf('\n═══════════════════════════════════════════════════════════════════\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  辅助函数
%% ═══════════════════════════════════════════════════════════════════════════

function state = create_edge_rotation_trajectory(t, T_obs, omega_deg, edge_offset)
    % 绕边缘旋转的轨迹
    % 旋转中心在阵列左端点 (x = -edge_offset)
    % 阵列绕该点旋转，同时阵列自身也旋转
    
    angle = omega_deg * t / T_obs;  % 当前旋转角度（度）
    
    % 阵列中心绕左端点旋转
    % 初始: 阵列中心在 (0, 0)，左端点在 (-edge_offset, 0)
    % 旋转后: 阵列中心位置 = 旋转中心 + 旋转后的偏移
    rotation_center_x = -edge_offset;  % 旋转中心 (左端点初始位置)
    
    % 阵列中心相对于旋转中心的初始偏移是 (edge_offset, 0)
    % 旋转后的位置
    center_x = rotation_center_x + edge_offset * cosd(angle);
    center_y = edge_offset * sind(angle);
    
    state = struct(...
        'position', [center_x, center_y, 0], ...
        'orientation', [0, 0, angle]);
end

function [aperture_x, aperture_y, all_positions] = calc_synthetic_aperture(array, t_axis)
    % 计算合成孔径
    num_elements = array.get_num_virtual_elements();
    num_snapshots = length(t_axis);
    all_positions = zeros(num_elements * num_snapshots, 3);
    
    for k = 1:num_snapshots
        pos_k = array.get_mimo_virtual_positions(t_axis(k));
        idx_start = (k-1)*num_elements + 1;
        idx_end = k*num_elements;
        all_positions(idx_start:idx_end, :) = pos_k;
    end
    
    aperture_x = max(all_positions(:,1)) - min(all_positions(:,1));
    aperture_y = max(all_positions(:,2)) - min(all_positions(:,2));
end

function spectrum = music_standard(snapshots, positions, phi_search, lambda, num_targets)
    % 标准MUSIC算法（静态阵列）
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
            a(i) = exp(1j * phase);
        end
        
        spectrum(phi_idx) = 1 / abs(a' * (Qn * Qn') * a);
    end
end

function beamwidth = calc_beamwidth(spectrum, phi_search)
    spec_db = 10*log10(spectrum / max(spectrum));
    [~, peak_idx] = max(spec_db);
    
    left_idx = find(spec_db(1:peak_idx) < -3, 1, 'last');
    if isempty(left_idx), left_idx = 1; end
    
    right_idx = peak_idx + find(spec_db(peak_idx:end) < -3, 1, 'first') - 1;
    if isempty(right_idx), right_idx = length(phi_search); end
    
    beamwidth = phi_search(right_idx) - phi_search(left_idx);
    if beamwidth <= 0
        beamwidth = 1;  % 最小宽度
    end
end
