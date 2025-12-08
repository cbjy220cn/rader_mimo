%% ═══════════════════════════════════════════════════════════════════════════
%  运动阵列 vs 静态阵列 对比实验 (正确方法)
%  
%  核心思想：
%    运动阵列通过时间采样获得虚拟阵元位置，形成合成孔径
%    将所有时刻的阵元位置视为一个大的虚拟阵列
%    关键是构建正确的空间协方差矩阵
%% ═══════════════════════════════════════════════════════════════════════════

clear; clc; close all;

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║     运动阵列 vs 静态阵列 DOA性能对比 (正确方法)               ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

%% 基础参数
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

array_pos = zeros(num_elements, 3);
for i = 1:num_elements
    array_pos(i, :) = [(i-1)*spacing - physical_aperture/2, 0, 0];
end

fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('配置\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('阵元数: %d\n', num_elements);
fprintf('物理孔径: %.2fλ\n', physical_aperture/lambda);

%% 实验参数
target_range = 500;
true_phi = 30;
snr_db = 15;

% 运动配置 - 使用更多快拍来获得更大的合成孔径
v_platform = 5;  % m/s
T_chirp = 0.01;  % 10ms chirp周期
num_snapshots = 128;  % 增加快拍数
t_axis = (0:num_snapshots-1) * T_chirp;

total_displacement = v_platform * t_axis(end);
synthetic_aperture = physical_aperture + total_displacement;

fprintf('\n运动阵列:\n');
fprintf('  速度: %.1f m/s\n', v_platform);
fprintf('  快拍数: %d\n', num_snapshots);
fprintf('  观测时间: %.2f s\n', t_axis(end));
fprintf('  平移: %.2f m (%.1fλ)\n', total_displacement, total_displacement/lambda);
fprintf('  合成孔径: %.1fλ (扩展%.1f倍)\n\n', ...
    synthetic_aperture/lambda, synthetic_aperture/physical_aperture);

%% ═══════════════════════════════════════════════════════════════════════════
%  方法说明
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('方法说明：虚拟阵元合成法\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('  运动阵列：将不同时刻的阵元位置视为虚拟阵元\n');
fprintf('  每个快拍对应不同的阵元位置，形成合成孔径\n');
fprintf('  通过构建空间协方差矩阵来估计DOA\n\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  实验1：MUSIC谱对比
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('实验1：MUSIC谱对比（单目标）\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');

target = Target(target_range * [cosd(true_phi), sind(true_phi), 0], [0,0,0], 1);

phi_search = 0:0.1:90;

% --- 静态阵列 ---
array_static = ArrayPlatform(array_pos, 1, 1:num_elements);
array_static = array_static.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

sig_gen_s = SignalGeneratorSimple(radar_params, array_static, {target});
snapshots_s = sig_gen_s.generate_snapshots(t_axis, snr_db);

positions_s = array_static.get_mimo_virtual_positions(0);
spectrum_s = music_standard(snapshots_s, positions_s, phi_search, lambda, 1);

% --- 运动阵列（虚拟阵元合成法）---
array_moving = ArrayPlatform(array_pos, 1, 1:num_elements);
array_moving = array_moving.set_trajectory(@(t) struct('position', [v_platform*t, 0, 0], 'orientation', [0,0,0]));

sig_gen_m = SignalGeneratorSimple(radar_params, array_moving, {target});
snapshots_m = sig_gen_m.generate_snapshots(t_axis, snr_db);

% 使用空间平滑的思想：利用运动产生的虚拟阵元
% 将每个快拍的数据与对应的阵元位置关联
spectrum_m = music_synthetic_aperture(snapshots_m, array_moving, t_axis, phi_search, lambda, 1);

% 计算主瓣宽度
beamwidth_s = calc_beamwidth(spectrum_s, phi_search);
beamwidth_m = calc_beamwidth(spectrum_m, phi_search);

fprintf('\n主瓣宽度:\n');
fprintf('  静态阵列: %.1f°\n', beamwidth_s);
fprintf('  运动阵列: %.1f°\n', beamwidth_m);
if beamwidth_m < beamwidth_s
    fprintf('  改善: %.1f倍 ✓\n', beamwidth_s / beamwidth_m);
else
    fprintf('  改善: %.1f倍\n', beamwidth_s / beamwidth_m);
end

%% ═══════════════════════════════════════════════════════════════════════════
%  实验2：角度分辨率对比（双目标）
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('实验2：角度分辨率对比（双目标）\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');

% 测试不同的目标间隔
separations = [15, 10, 5, 3];  % 角度间隔
phi_center = 30;

resolution_results = struct();

for sep_idx = 1:length(separations)
    sep = separations(sep_idx);
    phi1 = phi_center - sep/2;
    phi2 = phi_center + sep/2;
    
    fprintf('\n间隔 %.0f°:\n', sep);
    
    target1 = Target(target_range * [cosd(phi1), sind(phi1), 0], [0,0,0], 1);
    target2 = Target(target_range * [cosd(phi2), sind(phi2), 0], [0,0,0], 1);
    
    phi_search_res = (phi_center-20):0.1:(phi_center+20);
    
    % 静态
    sig_s = SignalGeneratorSimple(radar_params, array_static, {target1, target2});
    snap_s = sig_s.generate_snapshots(t_axis, snr_db);
    spec_s = music_standard(snap_s, positions_s, phi_search_res, lambda, 2);
    resolved_s = check_resolution(spec_s, phi_search_res, [phi1, phi2]);
    
    % 运动
    sig_m = SignalGeneratorSimple(radar_params, array_moving, {target1, target2});
    snap_m = sig_m.generate_snapshots(t_axis, snr_db);
    spec_m = music_synthetic_aperture(snap_m, array_moving, t_axis, phi_search_res, lambda, 2);
    resolved_m = check_resolution(spec_m, phi_search_res, [phi1, phi2]);
    
    fprintf('  静态: %s\n', ternary(resolved_s, '✓ 可分辨', '✗ 无法分辨'));
    fprintf('  运动: %s\n', ternary(resolved_m, '✓ 可分辨', '✗ 无法分辨'));
    
    resolution_results(sep_idx).sep = sep;
    resolution_results(sep_idx).resolved_s = resolved_s;
    resolution_results(sep_idx).resolved_m = resolved_m;
    resolution_results(sep_idx).spec_s = spec_s;
    resolution_results(sep_idx).spec_m = spec_m;
    resolution_results(sep_idx).phi_search = phi_search_res;
    resolution_results(sep_idx).phi_true = [phi1, phi2];
end

%% ═══════════════════════════════════════════════════════════════════════════
%  实验3：RMSE vs SNR
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('实验3：估计精度 vs SNR\n');
fprintf('═══════════════════════════════════════════════════════════════════\n');

snr_range = -5:5:20;
num_trials = 20;
phi_search_fine = (true_phi-10):0.1:(true_phi+10);

rmse_static = zeros(length(snr_range), 1);
rmse_moving = zeros(length(snr_range), 1);

fprintf('进度: ');
for snr_idx = 1:length(snr_range)
    snr = snr_range(snr_idx);
    fprintf('%ddB ', snr);
    
    errors_s = zeros(num_trials, 1);
    errors_m = zeros(num_trials, 1);
    
    for trial = 1:num_trials
        rng(trial + snr_idx*100);
        
        % 静态
        sig_s = SignalGeneratorSimple(radar_params, array_static, {target});
        snap_s = sig_s.generate_snapshots(t_axis, snr);
        spec_s = music_standard(snap_s, positions_s, phi_search_fine, lambda, 1);
        [~, pk] = max(spec_s);
        errors_s(trial) = (phi_search_fine(pk) - true_phi)^2;
        
        % 运动
        sig_m = SignalGeneratorSimple(radar_params, array_moving, {target});
        snap_m = sig_m.generate_snapshots(t_axis, snr);
        spec_m = music_synthetic_aperture(snap_m, array_moving, t_axis, phi_search_fine, lambda, 1);
        [~, pk] = max(spec_m);
        errors_m(trial) = (phi_search_fine(pk) - true_phi)^2;
    end
    
    rmse_static(snr_idx) = sqrt(mean(errors_s));
    rmse_moving(snr_idx) = sqrt(mean(errors_m));
end
fprintf('\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  绘图
%% ═══════════════════════════════════════════════════════════════════════════
figure('Position', [50, 50, 1500, 800]);

% 单目标MUSIC谱对比
subplot(2,3,1);
spec_s_db = 10*log10(spectrum_s/max(spectrum_s));
spec_m_db = 10*log10(spectrum_m/max(spectrum_m));
plot(phi_search, spec_s_db, 'b-', 'LineWidth', 2);
hold on;
plot(phi_search, spec_m_db, 'r-', 'LineWidth', 2);
yline(-3, 'k--', '3dB');
xline(true_phi, 'g--', 'LineWidth', 1.5);
xlabel('φ (°)');
ylabel('MUSIC谱 (dB)');
title(sprintf('单目标MUSIC谱\n静态:%.1f° vs 运动:%.1f°', beamwidth_s, beamwidth_m));
legend(sprintf('静态(%.1fλ)', physical_aperture/lambda), ...
       sprintf('运动(%.1fλ)', synthetic_aperture/lambda), 'Location', 'northeast');
grid on;
xlim([true_phi-20, true_phi+20]);
ylim([-30, 0]);

% 双目标分辨（选最难的情况）
subplot(2,3,2);
best_idx = find([resolution_results.resolved_m] & ~[resolution_results.resolved_s], 1);
if isempty(best_idx)
    best_idx = 1;  % 使用第一个
end
res = resolution_results(best_idx);
spec_s_db = 10*log10(res.spec_s/max(res.spec_s));
spec_m_db = 10*log10(res.spec_m/max(res.spec_m));
plot(res.phi_search, spec_s_db, 'b-', 'LineWidth', 2);
hold on;
plot(res.phi_search, spec_m_db, 'r-', 'LineWidth', 2);
for phi_t = res.phi_true
    xline(phi_t, 'k--');
end
xlabel('φ (°)');
ylabel('MUSIC谱 (dB)');
title(sprintf('双目标分辨(间隔%.0f°)\n静态:%s 运动:%s', ...
    res.sep, ternary(res.resolved_s, '✓', '✗'), ternary(res.resolved_m, '✓', '✗')));
legend('静态', '运动');
grid on;
ylim([-30, 0]);

% 分辨率统计
subplot(2,3,3);
seps = [resolution_results.sep];
resolved_s_vec = [resolution_results.resolved_s];
resolved_m_vec = [resolution_results.resolved_m];
bar_data = [resolved_s_vec; resolved_m_vec]';
bar(seps, bar_data);
xlabel('目标间隔 (°)');
ylabel('可分辨 (1=是, 0=否)');
title('角度分辨能力');
legend('静态', '运动');
grid on;
set(gca, 'XDir', 'reverse');

% RMSE vs SNR
subplot(2,3,4);
semilogy(snr_range, rmse_static, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
semilogy(snr_range, rmse_moving, 'r^-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('SNR (dB)');
ylabel('RMSE (°)');
title('估计精度 vs SNR');
legend('静态', '运动', 'Location', 'northeast');
grid on;

% 改善统计
subplot(2,3,5);
static_resolved = sum(resolved_s_vec);
moving_resolved = sum(resolved_m_vec);
bar([static_resolved, moving_resolved]);
set(gca, 'XTickLabel', {'静态', '运动'});
ylabel('可分辨的目标间隔数');
title(sprintf('分辨能力对比\n静态:%d/4 运动:%d/4', static_resolved, moving_resolved));
grid on;

% 配置总结
subplot(2,3,6);
text(0.1, 0.9, '配置参数:', 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.75, sprintf('阵元数: %d', num_elements), 'FontSize', 11);
text(0.1, 0.6, sprintf('物理孔径: %.1fλ', physical_aperture/lambda), 'FontSize', 11);
text(0.1, 0.45, sprintf('合成孔径: %.1fλ', synthetic_aperture/lambda), 'FontSize', 11);
text(0.1, 0.3, sprintf('孔径扩展: %.1f倍', synthetic_aperture/physical_aperture), 'FontSize', 11);
text(0.1, 0.15, sprintf('主瓣宽度改善: %.1f倍', beamwidth_s/beamwidth_m), 'FontSize', 11);
axis off;

sgtitle(sprintf('运动阵列 vs 静态阵列 性能对比\n合成孔径法 (孔径扩展%.1fx)', ...
    synthetic_aperture/physical_aperture), 'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, 'motion_vs_static_comparison.png');
fprintf('\n图片已保存: motion_vs_static_comparison.png\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  结论
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                        实验结论                                   \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

fprintf('📊 主瓣宽度:\n');
fprintf('   静态: %.1f°, 运动: %.1f°\n', beamwidth_s, beamwidth_m);
fprintf('   改善: %.1f倍\n', beamwidth_s/beamwidth_m);

fprintf('\n📊 角度分辨率:\n');
fprintf('   静态阵列可分辨: %d/%d 种间隔\n', static_resolved, length(separations));
fprintf('   运动阵列可分辨: %d/%d 种间隔\n', moving_resolved, length(separations));

fprintf('\n🎯 核心结论:\n');
if beamwidth_m < beamwidth_s
    fprintf('   ✅ 运动阵列主瓣更窄，角度分辨能力更强\n');
end
if moving_resolved > static_resolved
    fprintf('   ✅ 运动阵列能分辨更小的目标间隔\n');
end
fprintf('   合成孔径扩展 %.1f 倍\n', synthetic_aperture/physical_aperture);

fprintf('\n═══════════════════════════════════════════════════════════════════\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  辅助函数
%% ═══════════════════════════════════════════════════════════════════════════

function spectrum = music_standard(snapshots, positions, phi_search, lambda, num_targets)
    % 标准MUSIC算法
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

function spectrum = music_synthetic_aperture(snapshots, array_platform, t_axis, phi_search, lambda, num_targets)
    % 合成孔径MUSIC算法
    % 利用运动产生的虚拟阵元位置进行DOA估计
    
    [num_elements, num_snapshots] = size(snapshots);
    
    % 方法：将时间维度展开为空间维度
    % 每个时刻的每个阵元作为虚拟阵元
    
    % 收集所有虚拟阵元位置
    all_positions = zeros(num_elements * num_snapshots, 3);
    all_signals = zeros(num_elements * num_snapshots, 1);
    
    for k = 1:num_snapshots
        positions_k = array_platform.get_mimo_virtual_positions(t_axis(k));
        idx_start = (k-1)*num_elements + 1;
        idx_end = k*num_elements;
        all_positions(idx_start:idx_end, :) = positions_k;
        all_signals(idx_start:idx_end) = snapshots(:, k);
    end
    
    % 由于虚拟阵元太多，使用子采样或空间平滑
    % 这里使用选择性子采样：选取分布均匀的虚拟阵元
    subsample_factor = max(1, floor(num_snapshots / 16));  % 控制虚拟阵元数量
    selected_snapshots = 1:subsample_factor:num_snapshots;
    num_selected = length(selected_snapshots);
    
    selected_positions = zeros(num_elements * num_selected, 3);
    selected_signals = zeros(num_elements * num_selected, 1);
    
    for k = 1:num_selected
        t_k = t_axis(selected_snapshots(k));
        positions_k = array_platform.get_mimo_virtual_positions(t_k);
        idx_start = (k-1)*num_elements + 1;
        idx_end = k*num_elements;
        selected_positions(idx_start:idx_end, :) = positions_k;
        selected_signals(idx_start:idx_end) = snapshots(:, selected_snapshots(k));
    end
    
    num_virtual = size(selected_positions, 1);
    
    % 构建协方差矩阵（虚拟阵元）
    Rxx = selected_signals * selected_signals';
    
    % 特征分解
    [V, D] = eig(Rxx);
    [eigenvalues, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    
    % 确定噪声子空间
    noise_dim = num_virtual - num_targets;
    Qn = V(:, (num_targets+1):end);
    
    % 计算MUSIC谱
    spectrum = zeros(size(phi_search));
    for phi_idx = 1:length(phi_search)
        phi = phi_search(phi_idx);
        u = [cosd(phi); sind(phi); 0];
        
        % 虚拟阵列的导向矢量
        a = zeros(num_virtual, 1);
        for i = 1:num_virtual
            phase = 4 * pi / lambda * (selected_positions(i, :) * u);
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
end

function resolved = check_resolution(spectrum, phi_search, phi_true)
    [pks, locs] = findpeaks(spectrum, 'MinPeakProminence', 0.1*max(spectrum), ...
                            'MinPeakDistance', 3, 'SortStr', 'descend', 'NPeaks', 2);
    
    if length(locs) >= 2
        peaks = sort(phi_search(locs(1:2)));
        sep_true = abs(diff(phi_true));
        sep_est = abs(diff(peaks));
        
        % 判断是否正确分辨
        error1 = min(abs(peaks - phi_true(1)));
        error2 = min(abs(peaks - phi_true(2)));
        
        resolved = sep_est > sep_true/2 && error1 < sep_true/2 && error2 < sep_true/2;
    else
        resolved = false;
    end
end

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
