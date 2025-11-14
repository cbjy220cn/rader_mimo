%% 测试单快拍MUSIC vs 多快拍GMUSIC
% 对比在旋转阵列上的性能
clear; clc;

fprintf('=== 单快拍MUSIC vs 多快拍GMUSIC ===\n\n');

% 雷达参数
radar_params.fc = 77e9;
radar_params.c = physconst('LightSpeed');
radar_params.lambda = radar_params.c / radar_params.fc;
radar_params.fs = 20e6;
radar_params.T_chirp = 50e-6;
radar_params.slope = 100e12;
radar_params.BW = radar_params.slope * radar_params.T_chirp;
radar_params.num_samples = radar_params.T_chirp * radar_params.fs;
radar_params.range_res = radar_params.c / (2 * radar_params.BW);

% 4x2 URA
num_elements = 8;
spacing = radar_params.lambda / 2;
Nx = 4;
Ny = 2;
x_coords = (-(Nx-1)/2:(Nx-1)/2) * spacing;
y_coords = (-(Ny-1)/2:(Ny-1)/2) * spacing;
[X, Y] = meshgrid(x_coords, y_coords);
rect_elements = [X(:), Y(:), zeros(num_elements, 1)];

% 目标
target_range = 20;
theta_true = 30;
phi_true = 20;
target_pos = [target_range * sind(theta_true) * cosd(phi_true), ...
              target_range * sind(theta_true) * sind(phi_true), ...
              target_range * cosd(theta_true)];
target = Target(target_pos, [0, 0, 0], 1);

fprintf('目标: theta=%.1f°, phi=%.1f°\n\n', theta_true, phi_true);

% 搜索网格
search_grid.theta = 0:0.5:90;
search_grid.phi = -90:0.5:90;

% 测试180度旋转
total_rotation = 180;
num_snapshots = 64;
T_chirp = 1e-4;
t_axis = (0:num_snapshots-1) * T_chirp;
omega_dps = total_rotation / t_axis(end);

fprintf('旋转%.0f度，%d个快拍\n\n', total_rotation, num_snapshots);

% 创建旋转阵列
array_rotating = ArrayPlatform(rect_elements, 1, 1:num_elements);
rotation_traj = @(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]);
array_rotating = array_rotating.set_trajectory(rotation_traj);

% 生成信号
sig_gen = SignalGenerator(radar_params, array_rotating, {target});
snapshots = sig_gen.generate_snapshots(t_axis, inf);

% === 方法1: 标准GMUSIC（使用所有快拍的协方差） ===
fprintf('--- 方法1: 标准GMUSIC ---\n');
estimator = DoaEstimator(array_rotating, radar_params);
spectrum_gmusic = estimator.estimate_gmusic(snapshots, t_axis, 1, search_grid);
[theta_gmusic, phi_gmusic, peak_gmusic] = DoaEstimator.find_peaks(spectrum_gmusic, search_grid, 1);

fprintf('估算: theta=%.1f°, phi=%.1f°\n', theta_gmusic, phi_gmusic);
fprintf('误差: Δtheta=%.1f°, Δphi=%.1f°\n', theta_gmusic-theta_true, phi_gmusic-phi_true);
fprintf('峰值: %.2f\n\n', peak_gmusic);

% === 方法2: 单快拍MUSIC（只用第一个快拍） ===
fprintf('--- 方法2: 单快拍MUSIC (第1个快拍) ---\n');
% 使用第一个快拍，阵列姿态为t=0
single_snapshot = snapshots(:, 1);
% 构造单快拍的"协方差矩阵"（实际是外积）
Rxx_single = single_snapshot * single_snapshot';
% 手动MUSIC
[eigenvectors, eigenvalues] = eig(Rxx_single);
[~, sorted_indices] = sort(diag(eigenvalues), 'descend');
signal_subspace = eigenvectors(:, sorted_indices(1:1));  % 1个信号
noise_subspace = eigenvectors(:, sorted_indices(2:end));
Qn = noise_subspace * noise_subspace';

% 搜索
spectrum_single = zeros(length(search_grid.theta), length(search_grid.phi));
for phi_idx = 1:numel(search_grid.phi)
    phi = search_grid.phi(phi_idx);
    for theta_idx = 1:numel(search_grid.theta)
        theta = search_grid.theta(theta_idx);
        u = [sind(theta)*cosd(phi); sind(theta)*sind(phi); cosd(theta)];
        
        % 只在t=0时刻计算导向矢量
        A_u = estimator.build_steering_matrix(t_axis(1), u);
        denominator = A_u' * Qn * A_u;
        spectrum_single(theta_idx, phi_idx) = 1 / abs(denominator);
    end
end

[theta_single, phi_single, peak_single] = DoaEstimator.find_peaks(spectrum_single, search_grid, 1);
fprintf('估算: theta=%.1f°, phi=%.1f°\n', theta_single, phi_single);
fprintf('误差: Δtheta=%.1f°, Δphi=%.1f°\n', theta_single-theta_true, phi_single-phi_true);
fprintf('峰值: %.2f\n\n', peak_single);

% === 方法3: 多快拍非相干平均（每个快拍独立计算谱，然后平均） ===
fprintf('--- 方法3: 非相干平均MUSIC (所有快拍) ---\n');
spectrum_incoherent = zeros(length(search_grid.theta), length(search_grid.phi));

for k = 1:num_snapshots
    single_snap = snapshots(:, k);
    Rxx_k = single_snap * single_snap';
    
    [eigenvectors_k, eigenvalues_k] = eig(Rxx_k);
    [~, sorted_idx] = sort(diag(eigenvalues_k), 'descend');
    noise_subspace_k = eigenvectors_k(:, sorted_idx(2:end));
    Qn_k = noise_subspace_k * noise_subspace_k';
    
    % 对这个快拍计算谱
    for phi_idx = 1:numel(search_grid.phi)
        phi = search_grid.phi(phi_idx);
        for theta_idx = 1:numel(search_grid.theta)
            theta = search_grid.theta(theta_idx);
            u = [sind(theta)*cosd(phi); sind(theta)*sind(phi); cosd(theta)];
            
            % 使用该快拍时刻的导向矢量
            A_u_k = estimator.build_steering_matrix(t_axis(k), u);
            denominator = A_u_k' * Qn_k * A_u_k;
            spectrum_incoherent(theta_idx, phi_idx) = spectrum_incoherent(theta_idx, phi_idx) + 1/abs(denominator);
        end
    end
end

% 平均
spectrum_incoherent = spectrum_incoherent / num_snapshots;

[theta_incoh, phi_incoh, peak_incoh] = DoaEstimator.find_peaks(spectrum_incoherent, search_grid, 1);
fprintf('估算: theta=%.1f°, phi=%.1f°\n', theta_incoh, phi_incoh);
fprintf('误差: Δtheta=%.1f°, Δphi=%.1f°\n', theta_incoh-theta_true, phi_incoh-phi_true);
fprintf('峰值: %.2f\n\n', peak_incoh);

% === 结果对比 ===
fprintf('=== 结果对比 ===\n');
fprintf('%-20s | %-12s | %-12s | %-12s\n', '方法', 'Phi误差', '峰值', '评价');
fprintf('%s\n', repmat('-', 1, 70));
fprintf('%-20s | %-12.1f | %-12.2f | %s\n', '标准GMUSIC', phi_gmusic-phi_true, peak_gmusic, ...
    abs(phi_gmusic-phi_true) < 5 ? '✓ 良好' : '✗ 失败');
fprintf('%-20s | %-12.1f | %-12.2f | %s\n', '单快拍MUSIC', phi_single-phi_true, peak_single, ...
    abs(phi_single-phi_true) < 5 ? '✓ 良好' : '✗ 失败');
fprintf('%-20s | %-12.1f | %-12.2f | %s\n', '非相干平均', phi_incoh-phi_true, peak_incoh, ...
    abs(phi_incoh-phi_true) < 5 ? '✓ 良好' : '✗ 失败');

% 可视化对比
figure('Position', [100, 100, 1400, 400]);

subplot(1, 3, 1);
surf(search_grid.phi, search_grid.theta, spectrum_gmusic);
shading interp; view(2); colorbar;
hold on; plot(phi_true, theta_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3); hold off;
title(sprintf('标准GMUSIC\n(phi误差=%.1f°)', phi_gmusic-phi_true));
xlabel('Phi (度)'); ylabel('Theta (度)');

subplot(1, 3, 2);
surf(search_grid.phi, search_grid.theta, spectrum_single);
shading interp; view(2); colorbar;
hold on; plot(phi_true, theta_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3); hold off;
title(sprintf('单快拍MUSIC\n(phi误差=%.1f°)', phi_single-phi_true));
xlabel('Phi (度)'); ylabel('Theta (度)');

subplot(1, 3, 3);
surf(search_grid.phi, search_grid.theta, spectrum_incoherent);
shading interp; view(2); colorbar;
hold on; plot(phi_true, theta_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3); hold off;
title(sprintf('非相干平均MUSIC\n(phi误差=%.1f°)', phi_incoh-phi_true));
xlabel('Phi (度)'); ylabel('Theta (度)');

sgtitle('180度旋转：不同MUSIC方法对比');

