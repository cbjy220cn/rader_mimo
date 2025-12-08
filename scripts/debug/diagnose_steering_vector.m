%% 诊断脚本：检查导向矢量与信号的匹配性
% 如果导向矢量正确，那么 A_true 应该与信号高度相关

clear; clc;

%% 参数
c = 3e8;
f0 = 3e9;
lambda = c / f0;

% 目标（正对ULA，验证算法）
target_theta = 90;  % deg (xy平面)
target_phi = 0;     % deg (x轴正方向)
target_range = 1000; % m

% 阵列
num_elements = 16;
spacing = 0.5 * lambda;
array_pos = generate_ula(num_elements, spacing);

% 雷达参数
radar_params.c = c;
radar_params.fc = f0;
radar_params.lambda = lambda;
radar_params.BW = 100e6;
radar_params.range_res = c / (2 * 100e6);
radar_params.fs = 36100;
radar_params.T_chirp = 10e-3;
radar_params.slope = 5e12;
radar_params.num_samples = 361;

num_snapshots = 128;
t_axis = (0:num_snapshots-1) * radar_params.T_chirp;

fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║        导向矢量诊断测试                                ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

%% 1. 生成信号
fprintf('[步骤1] 生成雷达信号...\n');

platform = ArrayPlatform(array_pos, 1, 1:num_elements);
platform = platform.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

target_pos = [target_range * sind(target_theta) * cosd(target_phi), ...
             target_range * sind(target_theta) * sind(target_phi), ...
             target_range * cosd(target_theta)];
targets = {Target(target_pos, [0,0,0], 1)};

sig_gen = SignalGenerator(radar_params, platform, targets);
snapshots = sig_gen.generate_snapshots(t_axis, 20);  % SNR=20dB

fprintf('  快拍维度: %d × %d\n', size(snapshots, 1), size(snapshots, 2));
fprintf('  快拍功率: %.2e\n\n', mean(abs(snapshots(:)).^2));

%% 2. 计算理论导向矢量
fprintf('[步骤2] 计算理论导向矢量...\n');

u_true = [sind(target_theta)*cosd(target_phi);
          sind(target_theta)*sind(target_phi);
          cosd(target_theta)];

fprintf('  真实方向矢量 u: [%.3f, %.3f, %.3f]\n', u_true);

% 使用DoaEstimator计算导向矢量
estimator = DoaEstimator(platform, radar_params);
A_true = estimator.build_steering_matrix(t_axis, u_true);

fprintf('  导向矢量维度: %d × %d\n', size(A_true, 1), size(A_true, 2));
fprintf('  导向矢量范数: %.2e\n\n', norm(A_true, 'fro'));

%% 3. 检查导向矢量与信号的相关性
fprintf('[步骤3] 检查导向矢量与信号的相关性...\n');

% 归一化
A_norm = A_true / norm(A_true, 'fro');
X_norm = snapshots / norm(snapshots, 'fro');

% 计算相关性（应该接近1如果匹配）
correlation = abs(trace(A_norm' * X_norm * X_norm' * A_norm)) / num_snapshots;
fprintf('  相关系数: %.4f (期望≈1.0)\n', correlation);

% 计算投影比例
signal_power = norm(snapshots, 'fro')^2;
projection_power = norm(A_true' * snapshots, 'fro')^2;
projection_ratio = projection_power / signal_power;
fprintf('  投影比例: %.2f%% (期望>90%%)\n', projection_ratio * 100);

if correlation < 0.5
    fprintf('  ❌ 相关性太低！导向矢量与信号不匹配！\n');
elseif correlation < 0.9
    fprintf('  ⚠️  相关性偏低，可能有问题\n');
else
    fprintf('  ✓ 相关性良好\n');
end
fprintf('\n');

%% 4. 检查协方差矩阵的主特征向量
fprintf('[步骤4] 检查协方差矩阵的主特征向量...\n');

Rxx = (snapshots * snapshots') / num_snapshots;
[V, D] = eig(Rxx);
[eigenvalues, idx] = sort(diag(D), 'descend');
V_sorted = V(:, idx);

fprintf('  特征值（前3个）: %.2e, %.2e, %.2e\n', eigenvalues(1:3));
fprintf('  信号/噪声比: %.2f\n', eigenvalues(1) / eigenvalues(end));

% 主特征向量（信号子空间）
v1 = V_sorted(:, 1);

% 与导向矢量（任一快拍）的相关性
a1 = A_true(:, 1);  % 第一个快拍的导向矢量
a1_norm = a1 / norm(a1);
v1_norm = v1 / norm(v1);

corr_av = abs(a1_norm' * v1_norm);
fprintf('  主特征向量与导向矢量相关性: %.4f (期望≈1.0)\n', corr_av);

if corr_av < 0.5
    fprintf('  ❌ 主特征向量与导向矢量不一致！\n');
    fprintf('     这说明实际信号的到达方向与理论不符\n');
elseif corr_av < 0.9
    fprintf('  ⚠️  相关性偏低\n');
else
    fprintf('  ✓ 主特征向量与导向矢量一致\n');
end
fprintf('\n');

%% 5. 测试调试版DoaEstimator
fprintf('[步骤5] 测试调试版DoaEstimator (标准MUSIC)...\n');

estimator_debug = DoaEstimator_DEBUG(platform, radar_params);
search_grid_local.theta = (target_theta-5):0.2:(target_theta+5);
search_grid_local.phi = (target_phi-5):0.2:(target_phi+5);

spec_debug = estimator_debug.estimate_gmusic(snapshots, t_axis, 1, search_grid_local);

[max_val_debug, max_idx_debug] = max(spec_debug(:));
[i_max_debug, j_max_debug] = ind2sub(size(spec_debug), max_idx_debug);
theta_est_debug = search_grid_local.theta(i_max_debug);
phi_est_debug = search_grid_local.phi(j_max_debug);

fprintf('  [调试版] 估计角度: θ=%.1f°, φ=%.1f°\n', theta_est_debug, phi_est_debug);
fprintf('  [调试版] 角度误差: Δθ=%.1f°, Δφ=%.1f°\n', ...
    abs(theta_est_debug - target_theta), abs(phi_est_debug - target_phi));
fprintf('  [调试版] 谱动态范围: %.2f\n\n', max_val_debug / min(spec_debug(:)));

%% 6. 对比：原版 vs 调试版
fprintf('[步骤6] 计算真实角度附近的MUSIC谱（原版）...\n');

% 噪声子空间
Qn = V_sorted(:, 2:end);
Qn_proj = Qn * Qn';

% 在真实角度附近搜索
theta_search = (target_theta-5):0.1:(target_theta+5);
phi_search = (target_phi-5):0.1:(target_phi+5);

spec_local = zeros(length(theta_search), length(phi_search));

for i = 1:length(theta_search)
    for j = 1:length(phi_search)
        u = [sind(theta_search(i))*cosd(phi_search(j));
             sind(theta_search(i))*sind(phi_search(j));
             cosd(theta_search(i))];
        
        A_u = estimator.build_steering_matrix(t_axis, u);
        denom = trace(A_u' * Qn_proj * A_u);
        spec_local(i,j) = 1 / abs(denom);
    end
end

[max_val, max_idx] = max(spec_local(:));
[i_max, j_max] = ind2sub(size(spec_local), max_idx);
theta_est = theta_search(i_max);
phi_est = phi_search(j_max);

fprintf('  真实角度: θ=%.1f°, φ=%.1f°\n', target_theta, target_phi);
fprintf('  估计角度: θ=%.1f°, φ=%.1f°\n', theta_est, phi_est);
fprintf('  角度误差: Δθ=%.1f°, Δφ=%.1f°\n', ...
    abs(theta_est - target_theta), abs(phi_est - target_phi));

fprintf('  谱峰值: %.2e\n', max_val);
fprintf('  谱动态范围: %.2f (max/min)\n', max_val / min(spec_local(:)));

if abs(theta_est - target_theta) < 0.5 && abs(phi_est - target_phi) < 0.5
    fprintf('  ✓ 在局部搜索中正确定位！\n');
else
    fprintf('  ❌ 即使在局部搜索中也无法正确定位！\n');
end
fprintf('\n');

%% 7. 可视化对比
figure('Position', [100, 100, 1600, 800]);

% 第一行：原版
subplot(2,3,1);
plot(1:num_snapshots, abs(snapshots(1,:)), 'b-', 'LineWidth', 1.5);
xlabel('快拍索引');
ylabel('幅度');
title('第1个阵元信号');
grid on;

subplot(2,3,2);
imagesc(phi_search, theta_search, 10*log10(spec_local));
axis xy;
colorbar;
xlabel('Phi (°)');
ylabel('Theta (°)');
title(sprintf('原版MUSIC谱 (动态范围%.1f)', max(spec_local(:))/min(spec_local(:))));
hold on;
plot(target_phi, target_theta, 'r+', 'MarkerSize', 20, 'LineWidth', 3);
plot(phi_est, theta_est, 'wo', 'MarkerSize', 15, 'LineWidth', 2);

subplot(2,3,3);
bar(1:5, eigenvalues(1:5));
xlabel('特征值索引');
ylabel('特征值');
title('协方差矩阵特征值');
grid on;
set(gca, 'YScale', 'log');

% 第二行：调试版
subplot(2,3,4);
% 找到最接近真实phi的索引
[~, phi_idx_orig] = min(abs(phi_search - target_phi));
[~, phi_idx_debug] = min(abs(search_grid_local.phi - target_phi));

% 确保维度匹配
theta_orig = theta_search;
theta_debug = search_grid_local.theta;

plot(theta_orig, 10*log10(spec_local(:, phi_idx_orig)), 'b-', 'LineWidth', 2);
hold on;
plot(theta_debug, 10*log10(spec_debug(:, phi_idx_debug)), 'r-', 'LineWidth', 2);
xlabel('Theta (°)');
ylabel('谱值 (dB)');
title(sprintf('Theta切片对比 (φ≈%.0f°)', target_phi));
legend('原版', '调试版');
grid on;

subplot(2,3,5);
imagesc(search_grid_local.phi, search_grid_local.theta, 10*log10(spec_debug));
axis xy;
colorbar;
xlabel('Phi (°)');
ylabel('Theta (°)');
title(sprintf('调试版MUSIC谱 (动态范围%.1f)', max(spec_debug(:))/min(spec_debug(:))));
hold on;
plot(target_phi, target_theta, 'r+', 'MarkerSize', 20, 'LineWidth', 3);
plot(phi_est_debug, theta_est_debug, 'wo', 'MarkerSize', 15, 'LineWidth', 2);

subplot(2,3,6);
% 对比动态范围
bar([max(spec_local(:))/min(spec_local(:)); max(spec_debug(:))/min(spec_debug(:))]);
set(gca, 'XTickLabel', {'原版', '调试版'});
ylabel('动态范围（线性）');
title('MUSIC谱动态范围对比');
grid on;

saveas(gcf, 'diagnose_steering_vector.png');
fprintf('✓ 图片已保存: diagnose_steering_vector.png\n\n');

fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║  诊断完成                                              ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n');

%% 辅助函数
function pos = generate_ula(N, spacing)
    x = ((0:N-1) * spacing - (N-1)*spacing/2)';
    pos = [x, zeros(N,1), zeros(N,1)];
end

