%% ═══════════════════════════════════════════════════════════════════════════
%  实验：CSA-BF vs ISA-MUSIC 方法对比
%  对比两种合成孔径DOA估计方法的性能
%% ═══════════════════════════════════════════════════════════════════════════
clear; clc; close all;

addpath('asset');

% 创建输出文件夹
script_name = 'experiment_method_comparison';
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
output_folder = fullfile('validation_results', [script_name '_' timestamp]);
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

log_file = fullfile(output_folder, 'experiment_log.txt');
diary(log_file);

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║       CSA-BF vs ISA-MUSIC 方法对比实验                         ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');
fprintf('输出目录: %s\n\n', output_folder);

%% ═════════════════════════════════════════════════════════════════════════════
%  实验说明
%% ═════════════════════════════════════════════════════════════════════════════
fprintf('┌─────────────────────────────────────────────────────────────────┐\n');
fprintf('│                        实验说明                                 │\n');
fprintf('├─────────────────────────────────────────────────────────────────┤\n');
fprintf('│ 【实验目的】                                                    │\n');
fprintf('│   对比两种合成孔径DOA估计方法的精度和特性                       │\n');
fprintf('│                                                                 │\n');
fprintf('│ 【方法1: CSA-BF (相干合成孔径波束形成)】                        │\n');
fprintf('│   原理：M×K虚拟阵列 + 匹配滤波                                 │\n');
fprintf('│   公式：P(θ) = |a(θ)''x|² / |a(θ)|²                            │\n');
fprintf('│   优势：完全相干利用孔径，分辨率高                              │\n');
fprintf('│   劣势：对相位误差敏感，无超分辨能力                            │\n');
fprintf('│                                                                 │\n');
fprintf('│ 【方法2: ISA-MUSIC (非相干合成孔径MUSIC)】                      │\n');
fprintf('│   原理：分段MUSIC + 谱累加                                     │\n');
fprintf('│   公式：P(θ) = Σ P_seg(θ)                                      │\n');
fprintf('│   优势：对相位误差鲁棒，有超分辨能力                            │\n');
fprintf('│   劣势：未完全相干利用孔径                                      │\n');
fprintf('│                                                                 │\n');
fprintf('│ 【方法3: 标准MUSIC (静态基准)】                                 │\n');
fprintf('│   原理：标准MUSIC，不利用运动                                  │\n');
fprintf('│   用于对比运动带来的增益                                        │\n');
fprintf('│                                                                 │\n');
fprintf('│ 【评估指标】                                                    │\n');
fprintf('│   ① RMSE: 均方根误差                                           │\n');
fprintf('│   ② 3dB主瓣宽度: 角度分辨率                                    │\n');
fprintf('│   ③ 计算时间: 算法效率                                         │\n');
fprintf('└─────────────────────────────────────────────────────────────────┘\n\n');

%% 参数设置
c = physconst('LightSpeed');
fc = 3e9;
lambda = c / fc;
d = lambda / 2;

radar_params = struct('fc', fc, 'lambda', lambda);

% 目标参数
target_phi = 30;
target_range = 500;

% 阵列参数
num_elements = 8;
x_pos = ((0:num_elements-1) - (num_elements-1)/2) * d;
elements = [x_pos', zeros(num_elements,1), zeros(num_elements,1)];

% 运动参数
v = 5;  % m/s
T_obs = 0.5;
num_snapshots = 64;
t_axis = linspace(0, T_obs, num_snapshots);

% 搜索参数 - 高精度
phi_coarse = 0:1:90;      % 粗搜索 1°
phi_fine = 0:0.02:90;     % 细搜索 0.02°

% SNR范围
snr_range = -10:5:20;
num_trials = 50;

fprintf('【系统参数】\n');
fprintf('  载频: %.2f GHz (λ = %.2f cm)\n', fc/1e9, lambda*100);
fprintf('  阵元数: %d, 间距: %.2f cm (%.2fλ)\n', num_elements, d*100, d/lambda);
fprintf('  运动速度: %.1f m/s\n', v);
fprintf('  观测时间: %.2f s, 快拍数: %d\n', T_obs, num_snapshots);
fprintf('  理论孔径扩展: %.1fλ → %.1fλ\n', (num_elements-1)*d/lambda, ...
    (num_elements-1)*d/lambda + v*T_obs/lambda);
fprintf('  搜索精度: 粗%.1f° → 细%.2f°\n\n', phi_coarse(2)-phi_coarse(1), phi_fine(2)-phi_fine(1));

%% 创建阵列
array_static = ArrayPlatform(elements, 1, 1:num_elements);
array_static.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

array_motion = ArrayPlatform(elements, 1, 1:num_elements);
array_motion.set_trajectory(@(t) struct('position', [0, v*t, 0], 'orientation', [0,0,0]));

%% 初始化结果
methods = {'静态MUSIC', 'CSA-BF', 'ISA-MUSIC', 'CSA-MUSIC'};
num_methods = length(methods);
num_snr = length(snr_range);

results = struct();
results.rmse = zeros(num_methods, num_snr);
results.bias = zeros(num_methods, num_snr);
results.std = zeros(num_methods, num_snr);
results.beamwidth = zeros(num_methods, 1);
results.compute_time = zeros(num_methods, 1);

%% 计算各方法的主瓣宽度（使用高SNR信号）
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('计算主瓣宽度和典型计算时间\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

target_pos = target_range * [cosd(target_phi), sind(target_phi), 0];
target = Target(target_pos, [0,0,0], 1);

search_grid_fine.phi = phi_fine;

% 生成高SNR测试信号
rng(1);
sig_gen_static = SignalGeneratorSimple(radar_params, array_static, {target});
snapshots_static = sig_gen_static.generate_snapshots(t_axis, 20);

sig_gen_motion = SignalGeneratorSimple(radar_params, array_motion, {target});
snapshots_motion = sig_gen_motion.generate_snapshots(t_axis, 20);

% 方法1: 静态MUSIC
fprintf('测试 静态MUSIC ...\n');
tic;
positions_static = array_static.get_mimo_virtual_positions(0);
Rxx = (snapshots_static * snapshots_static') / num_snapshots;
[V, D] = eig(Rxx);
[~, idx] = sort(diag(D), 'descend');
V = V(:, idx);
Qn = V(:, 2:end);
spectrum_static = music_1d(positions_static, Qn, phi_fine, lambda);
results.compute_time(1) = toc;
results.beamwidth(1) = calc_beamwidth(spectrum_static, phi_fine);
fprintf('  主瓣宽度: %.2f°, 计算时间: %.1f ms\n', results.beamwidth(1), results.compute_time(1)*1000);

% 方法2: CSA-BF
fprintf('测试 CSA-BF ...\n');
tic;
estimator_csa = DoaEstimatorSynthetic(array_motion, radar_params);
options_csa.method = 'csa-bf';
[spectrum_csa, ~, info_csa] = estimator_csa.estimate(snapshots_motion, t_axis, search_grid_fine, 1, options_csa);
results.compute_time(2) = toc;
results.beamwidth(2) = calc_beamwidth(spectrum_csa, phi_fine);
fprintf('  主瓣宽度: %.2f°, 计算时间: %.1f ms\n', results.beamwidth(2), results.compute_time(2)*1000);
fprintf('  合成孔径: %.1fλ\n', info_csa.synthetic_aperture.total_lambda);

% 方法3: ISA-MUSIC
fprintf('测试 ISA-MUSIC ...\n');
tic;
estimator_isa = DoaEstimatorSynthetic(array_motion, radar_params);
options_isa.method = 'isa-music';
[spectrum_isa, ~, info_isa] = estimator_isa.estimate(snapshots_motion, t_axis, search_grid_fine, 1, options_isa);
results.compute_time(3) = toc;
results.beamwidth(3) = calc_beamwidth(spectrum_isa, phi_fine);
fprintf('  主瓣宽度: %.2f°, 计算时间: %.1f ms\n', results.beamwidth(3), results.compute_time(3)*1000);
fprintf('  分段数: %d\n', info_isa.num_segments);

% 方法4: CSA-MUSIC（时间平滑）
fprintf('测试 CSA-MUSIC ...\n');
tic;
estimator_csam = DoaEstimatorSynthetic(array_motion, radar_params);
options_csam.method = 'csa-music';
[spectrum_csam, ~, info_csam] = estimator_csam.estimate(snapshots_motion, t_axis, search_grid_fine, 1, options_csam);
results.compute_time(4) = toc;
results.beamwidth(4) = calc_beamwidth(spectrum_csam, phi_fine);
fprintf('  主瓣宽度: %.2f°, 计算时间: %.1f ms\n', results.beamwidth(4), results.compute_time(4)*1000);
fprintf('  虚拟阵元: %d, 子阵列数: %d\n\n', info_csam.num_virtual, info_csam.num_subarrays);

%% 绘制MUSIC谱对比
fig1 = figure('Position', [100, 100, 1400, 400], 'Color', 'white');
set(gcf, 'DefaultAxesFontName', 'SimHei');

subplot(1, 4, 1);
spectrum_db = 10*log10(spectrum_static / max(spectrum_static));
plot(phi_fine, spectrum_db, 'b-', 'LineWidth', 1.5);
hold on;
xline(target_phi, 'r--', 'LineWidth', 1.5);
hold off;
xlabel('方位角 φ (°)');
ylabel('归一化功率 (dB)');
title(sprintf('(a) 静态MUSIC\n主瓣%.2f°', results.beamwidth(1)));
xlim([target_phi-15, target_phi+15]);
ylim([-40, 0]);
grid on;

subplot(1, 4, 2);
spectrum_db = 10*log10(spectrum_csa / max(spectrum_csa));
plot(phi_fine, spectrum_db, 'b-', 'LineWidth', 1.5);
hold on;
xline(target_phi, 'r--', 'LineWidth', 1.5);
hold off;
xlabel('方位角 φ (°)');
ylabel('归一化功率 (dB)');
title(sprintf('(b) CSA-BF\n主瓣%.2f°', results.beamwidth(2)));
xlim([target_phi-15, target_phi+15]);
ylim([-40, 0]);
grid on;

subplot(1, 4, 3);
spectrum_db = 10*log10(spectrum_isa / max(spectrum_isa));
plot(phi_fine, spectrum_db, 'b-', 'LineWidth', 1.5);
hold on;
xline(target_phi, 'r--', 'LineWidth', 1.5);
hold off;
xlabel('方位角 φ (°)');
ylabel('归一化功率 (dB)');
title(sprintf('(c) ISA-MUSIC\n主瓣%.2f°', results.beamwidth(3)));
xlim([target_phi-15, target_phi+15]);
ylim([-40, 0]);
grid on;

subplot(1, 4, 4);
spectrum_db = 10*log10(spectrum_csam / max(spectrum_csam));
plot(phi_fine, spectrum_db, 'b-', 'LineWidth', 1.5);
hold on;
xline(target_phi, 'r--', 'LineWidth', 1.5);
hold off;
xlabel('方位角 φ (°)');
ylabel('归一化功率 (dB)');
title(sprintf('(d) CSA-MUSIC\n主瓣%.2f°', results.beamwidth(4)));
xlim([target_phi-15, target_phi+15]);
ylim([-40, 0]);
grid on;

sgtitle('四种方法角度谱对比 (SNR=20dB)', 'FontSize', 14, 'FontWeight', 'bold');
saveas(fig1, fullfile(output_folder, 'fig1_谱对比.png'));
saveas(fig1, fullfile(output_folder, 'fig1_谱对比.eps'), 'epsc');

%% SNR扫描 - 蒙特卡洛实验
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('SNR扫描 - 蒙特卡洛实验 (%d SNR × %d trials)\n', num_snr, num_trials);
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% 使用粗搜索+细搜索加速
search_grid_coarse.phi = phi_coarse;
search_grid_fine.phi = phi_fine;

fprintf('SNR(dB) | 静态MUSIC | CSA-BF   | ISA-MUSIC | CSA-MUSIC\n');
fprintf('--------|-----------|----------|-----------|----------\n');

for snr_idx = 1:num_snr
    snr_db = snr_range(snr_idx);
    
    errors_static = zeros(1, num_trials);
    errors_csa = zeros(1, num_trials);
    errors_isa = zeros(1, num_trials);
    errors_csam = zeros(1, num_trials);
    
    for trial = 1:num_trials
        rng(trial * 1000 + snr_idx);
        
        % 目标角度加小随机偏移
        target_phi_trial = target_phi + (rand() - 0.5) * 0.5;
        target_pos_trial = target_range * [cosd(target_phi_trial), sind(target_phi_trial), 0];
        target_trial = Target(target_pos_trial, [0,0,0], 1);
        
        % 生成信号
        sig_gen_s = SignalGeneratorSimple(radar_params, array_static, {target_trial});
        snapshots_s = sig_gen_s.generate_snapshots(t_axis, snr_db);
        
        sig_gen_m = SignalGeneratorSimple(radar_params, array_motion, {target_trial});
        snapshots_m = sig_gen_m.generate_snapshots(t_axis, snr_db);
        
        % 方法1: 静态MUSIC（两阶段搜索）
        est_phi_static = two_stage_music(snapshots_s, array_static, phi_coarse, phi_fine, lambda, 5);
        errors_static(trial) = est_phi_static - target_phi_trial;
        
        % 方法2: CSA-BF（两阶段搜索）
        est_phi_csa = two_stage_synthetic(snapshots_m, t_axis, array_motion, radar_params, ...
            phi_coarse, phi_fine, 'csa-bf', 5);
        errors_csa(trial) = est_phi_csa - target_phi_trial;
        
        % 方法3: ISA-MUSIC（两阶段搜索）
        est_phi_isa = two_stage_synthetic(snapshots_m, t_axis, array_motion, radar_params, ...
            phi_coarse, phi_fine, 'isa-music', 5);
        errors_isa(trial) = est_phi_isa - target_phi_trial;
        
        % 方法4: CSA-MUSIC（两阶段搜索）
        est_phi_csam = two_stage_synthetic(snapshots_m, t_axis, array_motion, radar_params, ...
            phi_coarse, phi_fine, 'csa-music', 5);
        errors_csam(trial) = est_phi_csam - target_phi_trial;
    end
    
    results.rmse(1, snr_idx) = sqrt(mean(errors_static.^2));
    results.rmse(2, snr_idx) = sqrt(mean(errors_csa.^2));
    results.rmse(3, snr_idx) = sqrt(mean(errors_isa.^2));
    results.rmse(4, snr_idx) = sqrt(mean(errors_csam.^2));
    
    results.bias(1, snr_idx) = mean(errors_static);
    results.bias(2, snr_idx) = mean(errors_csa);
    results.bias(3, snr_idx) = mean(errors_isa);
    results.bias(4, snr_idx) = mean(errors_csam);
    
    results.std(1, snr_idx) = std(errors_static);
    results.std(2, snr_idx) = std(errors_csa);
    results.std(3, snr_idx) = std(errors_isa);
    results.std(4, snr_idx) = std(errors_csam);
    
    fprintf('  %3d   |   %.3f°  |  %.3f°  |  %.3f°   |  %.3f°\n', snr_db, ...
        results.rmse(1, snr_idx), results.rmse(2, snr_idx), results.rmse(3, snr_idx), results.rmse(4, snr_idx));
end

%% 绘制RMSE对比曲线
fig2 = figure('Position', [100, 100, 900, 500], 'Color', 'white');
set(gcf, 'DefaultAxesFontName', 'SimHei');

colors = {'k', 'b', 'r', [0 0.7 0]};
markers = {'o', 's', '^', 'd'};

for i = 1:num_methods
    semilogy(snr_range, results.rmse(i, :), ['-' markers{i}], 'Color', colors{i}, ...
        'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colors{i});
    hold on;
end
hold off;

xlabel('信噪比 (dB)', 'FontWeight', 'bold');
ylabel('RMSE (°)', 'FontWeight', 'bold');
title('四种方法RMSE对比', 'FontSize', 14, 'FontWeight', 'bold');
legend(methods, 'Location', 'northeast', 'FontSize', 11);
grid on;
xlim([min(snr_range)-1, max(snr_range)+1]);

% 添加孔径信息标注
text_y = max(results.rmse(:, 1)) * 0.7;
text(-8, text_y, sprintf('静态孔径: %.1fλ', (num_elements-1)*d/lambda), 'FontSize', 10);
text(-8, text_y*0.5, sprintf('合成孔径: %.1fλ', info_csa.synthetic_aperture.total_lambda), 'FontSize', 10);

saveas(fig2, fullfile(output_folder, 'fig2_RMSE对比.png'));
saveas(fig2, fullfile(output_folder, 'fig2_RMSE对比.eps'), 'epsc');

%% 统计汇总
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                        统计汇总                                   \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

fprintf('【主瓣宽度】\n');
for i = 1:num_methods
    fprintf('  %s: %.2f°\n', methods{i}, results.beamwidth(i));
end

fprintf('\n【高SNR (20dB) RMSE】\n');
for i = 1:num_methods
    fprintf('  %s: %.4f°\n', methods{i}, results.rmse(i, end));
end

fprintf('\n【低SNR (-10dB) RMSE】\n');
for i = 1:num_methods
    fprintf('  %s: %.3f°\n', methods{i}, results.rmse(i, 1));
end

fprintf('\n【改善倍数 (vs 静态MUSIC)】\n');
fprintf('  CSA-BF 高SNR改善: %.1f倍\n', results.rmse(1, end) / results.rmse(2, end));
fprintf('  ISA-MUSIC 高SNR改善: %.1f倍\n', results.rmse(1, end) / results.rmse(3, end));
fprintf('  CSA-MUSIC 高SNR改善: %.1f倍\n', results.rmse(1, end) / results.rmse(4, end));
fprintf('  CSA-BF 低SNR改善: %.1f倍\n', results.rmse(1, 1) / results.rmse(2, 1));
fprintf('  ISA-MUSIC 低SNR改善: %.1f倍\n', results.rmse(1, 1) / results.rmse(3, 1));
fprintf('  CSA-MUSIC 低SNR改善: %.1f倍\n', results.rmse(1, 1) / results.rmse(4, 1));

fprintf('\n【计算时间】\n');
for i = 1:num_methods
    fprintf('  %s: %.1f ms\n', methods{i}, results.compute_time(i)*1000);
end

%% 绘制偏差和标准差
fig3 = figure('Position', [100, 100, 1000, 400], 'Color', 'white');
set(gcf, 'DefaultAxesFontName', 'SimHei');

subplot(1, 2, 1);
for i = 1:num_methods
    plot(snr_range, results.bias(i, :), ['-' markers{i}], 'Color', colors{i}, ...
        'LineWidth', 1.5, 'MarkerSize', 6);
    hold on;
end
yline(0, 'k--');
hold off;
xlabel('信噪比 (dB)');
ylabel('偏差 (°)');
title('(a) 估计偏差');
legend(methods, 'Location', 'best');
grid on;

subplot(1, 2, 2);
for i = 1:num_methods
    semilogy(snr_range, results.std(i, :), ['-' markers{i}], 'Color', colors{i}, ...
        'LineWidth', 1.5, 'MarkerSize', 6);
    hold on;
end
hold off;
xlabel('信噪比 (dB)');
ylabel('标准差 (°)');
title('(b) 估计标准差');
legend(methods, 'Location', 'best');
grid on;

sgtitle('偏差与标准差分析', 'FontSize', 14, 'FontWeight', 'bold');
saveas(fig3, fullfile(output_folder, 'fig3_偏差标准差.png'));
saveas(fig3, fullfile(output_folder, 'fig3_偏差标准差.eps'), 'epsc');

%% 保存结果
results.snr_range = snr_range;
results.methods = methods;
results.info_csa = info_csa;
results.info_isa = info_isa;
save(fullfile(output_folder, 'experiment_results.mat'), 'results');

fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('实验完成！\n');
fprintf('所有结果保存在: %s\n', output_folder);
fprintf('═══════════════════════════════════════════════════════════════════\n');

diary off;

%% ═══════════════════════════════════════════════════════════════════════════
%  辅助函数
%% ═══════════════════════════════════════════════════════════════════════════

function spectrum = music_1d(positions, Qn, phi_search, lambda)
    % 标准1D MUSIC谱（平面波模型）
    % 相位 = 4π/λ × (位置·方向)，与SignalGeneratorSimple一致
    
    num_phi = length(phi_search);
    spectrum = zeros(1, num_phi);
    Qn_proj = Qn * Qn';
    
    for phi_idx = 1:num_phi
        phi = phi_search(phi_idx);
        u = [cosd(phi); sind(phi); 0];
        % 平面波导向矢量
        phase = 4 * pi / lambda * (positions * u);
        a = exp(1j * phase);
        
        denom = real(a' * Qn_proj * a);
        spectrum(phi_idx) = 1 / max(denom, 1e-12);
    end
end

function beamwidth = calc_beamwidth(spectrum, phi_search)
    % 计算3dB主瓣宽度
    spec_db = 10*log10(spectrum / max(spectrum));
    [~, peak_idx] = max(spec_db);
    
    left_idx = find(spec_db(1:peak_idx) < -3, 1, 'last');
    if isempty(left_idx), left_idx = 1; end
    
    right_idx = peak_idx + find(spec_db(peak_idx:end) < -3, 1, 'first') - 1;
    if isempty(right_idx), right_idx = length(phi_search); end
    
    beamwidth = phi_search(right_idx) - phi_search(left_idx);
    if beamwidth <= 0
        beamwidth = phi_search(2) - phi_search(1);
    end
end

function est_phi = two_stage_music(snapshots, array, phi_coarse, phi_fine, lambda, roi_margin)
    % 两阶段MUSIC搜索
    num_snapshots = size(snapshots, 2);
    positions = array.get_mimo_virtual_positions(0);
    
    Rxx = (snapshots * snapshots') / num_snapshots;
    [V, D] = eig(Rxx);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    Qn = V(:, 2:end);
    
    % 粗搜索
    spectrum_coarse = music_1d(positions, Qn, phi_coarse, lambda);
    [~, peak_idx] = max(spectrum_coarse);
    phi_peak = phi_coarse(peak_idx);
    
    % 细搜索
    phi_roi = phi_fine(phi_fine >= phi_peak - roi_margin & phi_fine <= phi_peak + roi_margin);
    if isempty(phi_roi)
        phi_roi = phi_fine;
    end
    spectrum_fine = music_1d(positions, Qn, phi_roi, lambda);
    [~, peak_idx_fine] = max(spectrum_fine);
    est_phi = phi_roi(peak_idx_fine);
end

function est_phi = two_stage_synthetic(snapshots, t_axis, array, radar_params, ...
    phi_coarse, phi_fine, method, roi_margin)
    % 两阶段合成孔径搜索
    
    estimator = DoaEstimatorSynthetic(array, radar_params);
    
    % 粗搜索
    search_grid_coarse.phi = phi_coarse;
    options.method = method;
    [~, peaks_coarse, ~] = estimator.estimate(snapshots, t_axis, search_grid_coarse, 1, options);
    phi_peak = peaks_coarse.phi(1);
    
    % 细搜索
    phi_roi = phi_fine(phi_fine >= phi_peak - roi_margin & phi_fine <= phi_peak + roi_margin);
    if isempty(phi_roi)
        phi_roi = phi_fine;
    end
    search_grid_fine.phi = phi_roi;
    [~, peaks_fine, ~] = estimator.estimate(snapshots, t_axis, search_grid_fine, 1, options);
    est_phi = peaks_fine.phi(1);
end

