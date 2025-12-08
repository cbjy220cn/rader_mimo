%% 调试脚本：单个配置测试
% 目的：查看为什么RMSE和SNR关系不对

clear; clc; close all;

%% 参数
c = 3e8;
f0 = 3e9;
lambda = c / f0;

% 目标（改为正对ULA方向，验证算法）
target_theta = 90;  % deg (正在xy平面上)
target_phi = 0;     % deg (正对x轴，ULA的最佳方向)
target_range = 1000; % m
% 此时方向矢量 u = [1, 0, 0]
% 完美匹配ULA的x轴方向，应该得到最佳分辨率

% 阵列：简单ULA（⚠️ 必须≤0.5λ避免空间混叠！）
num_elements = 16;  % 增加到16元以获得足够分辨率
spacing = 0.5 * lambda;  % 0.5λ（奈奎斯特极限，避免栅瓣）
array_pos = generate_ula(num_elements, spacing);
total_length = (num_elements-1)*spacing;
fprintf('阵列配置:\n');
fprintf('  阵元数: %d\n', num_elements);
fprintf('  间距: %.2fλ (奈奎斯特: ≤0.5λ) ✓\n', spacing/lambda);
fprintf('  总长度: %.1fλ (%.2f m)\n', total_length/lambda, total_length);
fprintf('  理论分辨率 (1000m处): ~%.1f°\n\n', atand(lambda/total_length));

% 雷达参数
radar_params.c = c;
radar_params.fc = f0;
radar_params.f0 = f0;
radar_params.lambda = lambda;
radar_params.bandwidth = 100e6;
radar_params.BW = 100e6;
radar_params.range_res = c / (2 * 100e6);
radar_params.fs = 36100;
radar_params.T_chirp = 10e-3;
radar_params.slope = 5e12;
radar_params.num_samples = 361;

% 搜索网格
search_grid.theta = 0:1:90;
search_grid.phi = 0:1:180;

% 测试不同SNR
snr_range = [0, 5, 10, 15, 20];
num_snapshots = 16;  % ⚠️ 关键：快拍数≈阵元数，避免噪声子空间过大

fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║           单配置调试测试                               ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');
fprintf('真实目标: θ=%.0f°, φ=%.0f°, R=%.0f m\n\n', target_theta, target_phi, target_range);

%% 测试静态阵列（不同SNR）
fprintf('【测试1】静态阵列 + 不同SNR\n');
fprintf('─────────────────────────────────────────────────\n');

for snr_db = snr_range
    % 创建平台
    platform = ArrayPlatform(array_pos, 1, 1:num_elements);
    platform = platform.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));
    
    % 创建目标
    target_pos = [target_range * sind(target_theta) * cosd(target_phi), ...
                 target_range * sind(target_theta) * sind(target_phi), ...
                 target_range * cosd(target_theta)];
    targets = {Target(target_pos, [0,0,0], 1)};
    
    % 生成信号
    t_axis = (0:num_snapshots-1) * radar_params.T_chirp;
    sig_gen = SignalGenerator(radar_params, platform, targets);
    snapshots = sig_gen.generate_snapshots(t_axis, snr_db);
    
    % DOA估计
    estimator = DoaEstimator(platform, radar_params);
    spectrum = estimator.estimate_gmusic(snapshots, t_axis, 1, search_grid);
    
    % 峰值检测
    [max_val, max_idx] = max(spectrum(:));
    [theta_idx, phi_idx] = ind2sub(size(spectrum), max_idx);
    est_theta = search_grid.theta(theta_idx);
    est_phi = search_grid.phi(phi_idx);
    
    % 计算误差
    error = sqrt((est_theta - target_theta)^2 + (est_phi - target_phi)^2);
    
    % 输出
    fprintf('SNR=%+3d dB: 估计=(%.1f°, %.1f°) | 误差=%.2f° | 峰值=%.2e\n', ...
        snr_db, est_theta, est_phi, error, max_val);
    
    % 如果误差大，检查谱
    if error > 10
        fprintf('  ⚠️  误差过大！检查谱的前5个最大值位置:\n');
        [sorted_vals, sorted_idx] = sort(spectrum(:), 'descend');
        for k = 1:5
            [t_idx, p_idx] = ind2sub(size(spectrum), sorted_idx(k));
            fprintf('      #%d: (%.0f°, %.0f°) = %.2e\n', k, ...
                search_grid.theta(t_idx), search_grid.phi(p_idx), sorted_vals(k));
        end
    end
end

fprintf('\n');

%% 可视化静态阵列的谱（诊断用）
fprintf('【诊断】生成静态阵列MUSIC谱的2D图...\n');
fprintf('─────────────────────────────────────────────────\n');

% 使用SNR=10dB的情况
snr_db = 10;
platform = ArrayPlatform(array_pos, 1, 1:num_elements);
platform = platform.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

target_pos = [target_range * sind(target_theta) * cosd(target_phi), ...
             target_range * sind(target_theta) * sind(target_phi), ...
             target_range * cosd(target_theta)];
targets = {Target(target_pos, [0,0,0], 1)};

t_axis = (0:num_snapshots-1) * radar_params.T_chirp;
sig_gen = SignalGenerator(radar_params, platform, targets);
snapshots = sig_gen.generate_snapshots(t_axis, snr_db);

fprintf('快拍维度: %d × %d\n', size(snapshots, 1), size(snapshots, 2));
fprintf('快拍功率: %.2e\n', mean(abs(snapshots(:)).^2));

% 检查协方差矩阵
Rxx = (snapshots * snapshots') / num_snapshots;
fprintf('协方差矩阵维度: %d × %d\n', size(Rxx, 1), size(Rxx, 2));
fprintf('协方差矩阵迹: %.2e\n', trace(Rxx));

% 特征值
[~, eigenvalues] = eig(Rxx);
eigvals = sort(diag(eigenvalues), 'descend');
fprintf('特征值（前5个）: ');
for i = 1:min(5, length(eigvals))
    fprintf('%.2e ', eigvals(i));
end
fprintf('\n');
fprintf('信号特征值/噪声特征值: %.2f\n', eigvals(1) / eigvals(end));

% MUSIC估计
estimator = DoaEstimator(platform, radar_params);
spectrum = estimator.estimate_gmusic(snapshots, t_axis, 1, search_grid);

fprintf('谱维度: %d × %d\n', size(spectrum, 1), size(spectrum, 2));
fprintf('谱的范围: [%.2e, %.2e]\n', min(spectrum(:)), max(spectrum(:)));
fprintf('谱的均值: %.2e\n', mean(spectrum(:)));
fprintf('谱的标准差: %.2e\n', std(spectrum(:)));

% 画图
figure('Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
imagesc(search_grid.phi, search_grid.theta, spectrum);
axis xy; colorbar;
xlabel('Phi (°)'); ylabel('Theta (°)');
title('MUSIC谱（静态，SNR=10dB）');
hold on;
plot(target_phi, target_theta, 'r+', 'MarkerSize', 20, 'LineWidth', 3);

subplot(1, 3, 2);
imagesc(search_grid.phi, search_grid.theta, 10*log10(spectrum));
axis xy; colorbar;
caxis([-40, 0]);
xlabel('Phi (°)'); ylabel('Theta (°)');
title('MUSIC谱 (dB)');
hold on;
plot(target_phi, target_theta, 'r+', 'MarkerSize', 20, 'LineWidth', 3);

subplot(1, 3, 3);
% Theta切片
[~, phi_idx] = min(abs(search_grid.phi - target_phi));
plot(search_grid.theta, 10*log10(spectrum(:, phi_idx)), 'b-', 'LineWidth', 2);
hold on;
plot(target_theta, 10*log10(spectrum(find(search_grid.theta==target_theta, 1), phi_idx)), 'ro', 'MarkerSize', 10);
grid on;
xlabel('Theta (°)'); ylabel('谱值 (dB)');
title(sprintf('Theta切片 (φ=%.0f°)', target_phi));
legend('MUSIC谱', '真实目标');

saveas(gcf, 'debug_static_spectrum.png');
fprintf('✓ 谱图已保存: debug_static_spectrum.png\n\n');

%% 测试运动阵列
fprintf('【测试2】平移运动 (v=10 m/s) + SNR=10 dB\n');
fprintf('─────────────────────────────────────────────────\n');

v = 10;  % m/s
snr_db = 10;

% 创建平台
platform = ArrayPlatform(array_pos, 1, 1:num_elements);
platform = platform.set_trajectory(@(t) struct('position', [v*t,0,0], 'orientation', [0,0,0]));

% 创建目标
target_pos = [target_range * sind(target_theta) * cosd(target_phi), ...
             target_range * sind(target_theta) * sind(target_phi), ...
             target_range * cosd(target_theta)];
targets = {Target(target_pos, [0,0,0], 1)};

% 生成信号
t_axis = (0:num_snapshots-1) * radar_params.T_chirp;
sig_gen = SignalGenerator(radar_params, platform, targets);
snapshots = sig_gen.generate_snapshots(t_axis, snr_db);

% DOA估计
estimator = DoaEstimatorIncoherent_FIXED(platform, radar_params);
options.verbose = false;
options.weighting = 'uniform';
options.num_segments = 4;
spectrum = estimator.estimate_incoherent_music(snapshots, t_axis, 1, search_grid, options);

% 峰值检测
[max_val, max_idx] = max(spectrum(:));
[theta_idx, phi_idx] = ind2sub(size(spectrum), max_idx);
est_theta = search_grid.theta(theta_idx);
est_phi = search_grid.phi(phi_idx);

% 计算误差
error = sqrt((est_theta - target_theta)^2 + (est_phi - target_phi)^2);

% 输出
fprintf('SNR=%+3d dB: 估计=(%.1f°, %.1f°) | 误差=%.2f° | 峰值=%.2e\n', ...
    snr_db, est_theta, est_phi, error, max_val);
fprintf('飞行距离: %.1f m\n', v * t_axis(end));

fprintf('\n╔════════════════════════════════════════════════════════╗\n');
fprintf('║  调试完成！检查上面的输出                              ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n');

%% 辅助函数
function pos = generate_ula(N, spacing)
    x = ((0:N-1) * spacing - (N-1)*spacing/2)';
    pos = [x, zeros(N,1), zeros(N,1)];
end

