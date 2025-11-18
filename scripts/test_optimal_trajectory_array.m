%% 最优轨迹-阵列组合实验
% 目标：找出在相同阵元数下，哪种阵列配置+运动轨迹组合性能最好
%
% 测试内容：
%   - 4种阵列配置：线阵、圆阵、矩形阵、随机阵
%   - 6种运动轨迹：静止、直线、圆周、螺旋、8字、随机
%   - 评估指标：分辨率、有效孔径、空间采样均匀性、DOA精度

clear; clc; close all;
fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║      最优轨迹-阵列组合探索实验                        ║\n');
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

fprintf('📡 雷达参数: f₀=%.2f GHz, λ=%.3f m\n\n', f0/1e9, lambda);

%% 实验参数
num_elements = 8;           % 阵元数（固定）
num_snapshots = 64;         % 快拍数（固定）
element_spacing = 0.5 * lambda;  % 阵元间距（半波长）
aperture_size = 0.1;        % 总孔径大小（米）

% 单目标场景
target_range = 600;
theta_true = 30;
phi_true = 60;
target_pos = [target_range * sind(theta_true) * cosd(phi_true), ...
              target_range * sind(theta_true) * sind(phi_true), ...
              target_range * cosd(theta_true)];
targets = {Target(target_pos, [0,0,0], 1)};

% 搜索网格（使用智能搜索）
smart_grid.coarse_res = 5.0;
smart_grid.fine_res = 0.5;  % 稍粗一点，加速测试
smart_grid.roi_margin = 10.0;
smart_grid.theta_range = [0, 90];
smart_grid.phi_range = [0, 180];

search_grid.theta = 0:0.5:90;
search_grid.phi = 0:0.5:180;

fprintf('实验设置:\n');
fprintf('  阵元数: %d\n', num_elements);
fprintf('  快拍数: %d\n', num_snapshots);
fprintf('  目标: theta=%.1f°, phi=%.1f°, 距离=%dm\n\n', theta_true, phi_true, target_range);

%% ========================================================================
%% 定义阵列配置
%% ========================================================================
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('定义阵列配置\n');
fprintf('═══════════════════════════════════════════════════════\n');

array_configs = {};

% 配置1: 均匀线阵 (ULA)
rx_ula = zeros(num_elements, 3);
for i = 1:num_elements
    rx_ula(i, :) = [(i-1)*element_spacing - (num_elements-1)*element_spacing/2, 0, 0];
end
array_configs{1}.name = '均匀线阵(ULA)';
array_configs{1}.rx_positions = rx_ula;
array_configs{1}.description = '一维，均匀间距';
fprintf('  ✓ 配置1: 均匀线阵 (ULA) - %d元, 间距%.3fm\n', num_elements, element_spacing);

% 配置2: 圆形阵列
theta_circle = linspace(0, 2*pi, num_elements+1);
theta_circle(end) = [];
rx_circle = zeros(num_elements, 3);
radius = aperture_size / 2;
for i = 1:num_elements
    rx_circle(i, :) = [radius * cos(theta_circle(i)), radius * sin(theta_circle(i)), 0];
end
array_configs{2}.name = '圆形阵列';
array_configs{2}.rx_positions = rx_circle;
array_configs{2}.description = '圆周分布，半径%.2fm';
fprintf('  ✓ 配置2: 圆形阵列 - %d元, 半径%.3fm\n', num_elements, radius);

% 配置3: 矩形阵列 (URA)
if mod(num_elements, 2) == 0
    rows = 2;
    cols = num_elements / 2;
else
    rows = 2;
    cols = floor(num_elements / 2);
end
rx_rect = [];
for i = 1:rows
    for j = 1:cols
        if size(rx_rect, 1) < num_elements
            rx_rect = [rx_rect; [(j-1)*element_spacing - (cols-1)*element_spacing/2, ...
                                  (i-1)*element_spacing - (rows-1)*element_spacing/2, 0]];
        end
    end
end
array_configs{3}.name = '矩形阵列(URA)';
array_configs{3}.rx_positions = rx_rect;
array_configs{3}.description = sprintf('%d×%d, 间距%.3fm', rows, cols, element_spacing);
fprintf('  ✓ 配置3: 矩形阵列 (URA) - %d×%d\n', rows, cols);

% 配置4: L型阵列
rx_L = zeros(num_elements, 3);
half = floor(num_elements / 2);
for i = 1:half
    rx_L(i, :) = [(i-1)*element_spacing, 0, 0];  % 水平臂
end
for i = 1:(num_elements - half)
    rx_L(half + i, :) = [0, i*element_spacing, 0];  % 垂直臂
end
array_configs{4}.name = 'L型阵列';
array_configs{4}.rx_positions = rx_L;
array_configs{4}.description = '两个正交线阵';
fprintf('  ✓ 配置4: L型阵列 - %d+%d元\n\n', half, num_elements - half);

%% ========================================================================
%% 定义运动轨迹
%% ========================================================================
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('定义运动轨迹\n');
fprintf('═══════════════════════════════════════════════════════\n');

trajectories = {};
t_total = (num_snapshots - 1) * radar_params.T_chirp;

% 轨迹1: 静止
trajectories{1}.name = '静止';
trajectories{1}.func = @(t) struct('position', [0,0,0], 'orientation', [0,0,0]);
trajectories{1}.description = '基准对比';
fprintf('  ✓ 轨迹1: 静止（基准）\n');

% 轨迹2: 直线平移（X方向）
velocity = 0.5;  % m/s
trajectories{2}.name = '直线平移(X)';
trajectories{2}.func = @(t) struct('position', [velocity*t, 0, 0], 'orientation', [0,0,0]);
trajectories{2}.description = sprintf('X方向, %.1fm/s', velocity);
fprintf('  ✓ 轨迹2: 直线平移 - X方向, %.1fm/s\n', velocity);

% 轨迹3: 圆周旋转（绕Z轴，360°）
omega_dps = 360 / t_total;
trajectories{3}.name = '圆周旋转';
trajectories{3}.func = @(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]);
trajectories{3}.description = sprintf('360°, %.1f°/s', omega_dps);
fprintf('  ✓ 轨迹3: 圆周旋转 - 360°\n');

% 轨迹4: 螺旋运动（旋转+平移）
trajectories{4}.name = '螺旋运动';
trajectories{4}.func = @(t) struct('position', [velocity*t, 0, 0], 'orientation', [0, 0, omega_dps * t]);
trajectories{4}.description = '旋转+平移';
fprintf('  ✓ 轨迹4: 螺旋 - 旋转+平移\n');

% 轨迹5: 8字轨迹（Lissajous曲线）
A = 0.5;  % 振幅
trajectories{5}.name = '8字轨迹';
trajectories{5}.func = @(t) struct('position', [A*sin(2*pi*t/t_total), A*sin(4*pi*t/t_total), 0], ...
                                    'orientation', [0, 0, 0]);
trajectories{5}.description = 'Lissajous曲线';
fprintf('  ✓ 轨迹5: 8字轨迹\n');

% 轨迹6: 圆形平移轨迹
R_circle = 0.3;
trajectories{6}.name = '圆形平移';
trajectories{6}.func = @(t) struct('position', [R_circle*cos(2*pi*t/t_total), R_circle*sin(2*pi*t/t_total), 0], ...
                                    'orientation', [0, 0, 0]);
trajectories{6}.description = sprintf('圆形路径, 半径%.1fm', R_circle);
fprintf('  ✓ 轨迹6: 圆形平移 - 半径%.1fm\n\n', R_circle);

%% ========================================================================
%% 运行所有组合实验
%% ========================================================================
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('运行实验: %d阵列 × %d轨迹 = %d 组合\n', ...
    length(array_configs), length(trajectories), ...
    length(array_configs) * length(trajectories));
fprintf('═══════════════════════════════════════════════════════\n\n');

results = cell(length(array_configs), length(trajectories));
t_axis = (0:num_snapshots-1) * radar_params.T_chirp;

total_combinations = length(array_configs) * length(trajectories);
current_idx = 0;

for arr_idx = 1:length(array_configs)
    for traj_idx = 1:length(trajectories)
        current_idx = current_idx + 1;
        
        fprintf('[%2d/%2d] %s + %s ... ', current_idx, total_combinations, ...
            array_configs{arr_idx}.name, trajectories{traj_idx}.name);
        tic;
        
        % 创建阵列平台
        array_platform = ArrayPlatform(array_configs{arr_idx}.rx_positions, 1, 1:num_elements);
        array_platform = array_platform.set_trajectory(trajectories{traj_idx}.func);
        
        % 生成信号
        sig_gen = SignalGenerator(radar_params, array_platform, targets);
        snapshots = sig_gen.generate_snapshots(t_axis, inf);
        
        % DOA估计
        if traj_idx == 1  % 静止用相干MUSIC
            estimator = DoaEstimator(array_platform, radar_params);
            [spectrum, ~] = smart_doa_search(estimator, snapshots, t_axis, 1, smart_grid, struct('verbose', false));
        else  % 运动用非相干MUSIC
            estimator = DoaEstimatorIncoherent(array_platform, radar_params);
            [spectrum, ~] = smart_doa_search(estimator, snapshots, t_axis, 1, smart_grid, struct('verbose', false, 'weighting', 'uniform'));
        end
        
        % 找峰值
        if traj_idx == 1
            [theta_est, phi_est, peak_val] = DoaEstimator.find_peaks(spectrum, search_grid, 1);
        else
            [theta_est, phi_est, peak_val] = DoaEstimatorIncoherent.find_peaks(spectrum, search_grid, 1);
        end
        
        % 计算波束宽度（3dB）
        [~, theta_idx] = min(abs(search_grid.theta - theta_true));
        slice = spectrum(theta_idx, :);
        slice_norm = slice / max(slice);
        [~, peak_idx] = max(slice_norm);
        left_idx = find(slice_norm(1:peak_idx) < 0.5, 1, 'last');
        right_idx = peak_idx + find(slice_norm(peak_idx:end) < 0.5, 1, 'first') - 1;
        if isempty(left_idx), left_idx = 1; end
        if isempty(right_idx), right_idx = length(slice_norm); end
        beamwidth = (right_idx - left_idx) * (search_grid.phi(2) - search_grid.phi(1));
        
        % 计算空间覆盖（虚拟阵元位置的唯一性）
        virtual_positions = [];
        for k = 1:length(t_axis)
            vp = array_platform.get_mimo_virtual_positions(t_axis(k));
            virtual_positions = [virtual_positions; vp];
        end
        % 空间采样点数（去重，容差1cm）
        unique_positions = uniquetol(virtual_positions, 0.01, 'ByRows', true);
        spatial_coverage = size(unique_positions, 1);
        
        % 保存结果
        results{arr_idx, traj_idx}.array_name = array_configs{arr_idx}.name;
        results{arr_idx, traj_idx}.traj_name = trajectories{traj_idx}.name;
        results{arr_idx, traj_idx}.theta_est = theta_est;
        results{arr_idx, traj_idx}.phi_est = phi_est;
        results{arr_idx, traj_idx}.theta_error = abs(theta_est - theta_true);
        results{arr_idx, traj_idx}.phi_error = abs(phi_est - phi_true);
        results{arr_idx, traj_idx}.peak_val = peak_val;
        results{arr_idx, traj_idx}.beamwidth = beamwidth;
        results{arr_idx, traj_idx}.spatial_coverage = spatial_coverage;
        results{arr_idx, traj_idx}.spectrum = spectrum;
        
        elapsed = toc;
        fprintf('完成 (%.1fs) [BW:%.1f°, 覆盖:%d点, 误差:θ=%.2f°,φ=%.2f°]\n', ...
            elapsed, beamwidth, spatial_coverage, ...
            results{arr_idx, traj_idx}.theta_error, results{arr_idx, traj_idx}.phi_error);
    end
end

fprintf('\n✓ 所有实验完成\n\n');

%% ========================================================================
%% 分析结果
%% ========================================================================
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('结果分析\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

% 提取指标矩阵
beamwidth_matrix = zeros(length(array_configs), length(trajectories));
coverage_matrix = zeros(length(array_configs), length(trajectories));
error_matrix = zeros(length(array_configs), length(trajectories));

for i = 1:length(array_configs)
    for j = 1:length(trajectories)
        beamwidth_matrix(i, j) = results{i, j}.beamwidth;
        coverage_matrix(i, j) = results{i, j}.spatial_coverage;
        error_matrix(i, j) = sqrt(results{i, j}.theta_error^2 + results{i, j}.phi_error^2);
    end
end

% 找最优组合
fprintf('📊 分辨率性能排名 (波束宽度越小越好):\n');
[~, best_idx] = min(beamwidth_matrix(:));
[best_arr, best_traj] = ind2sub(size(beamwidth_matrix), best_idx);
fprintf('   🥇 最优: %s + %s (%.2f°)\n', ...
    array_configs{best_arr}.name, trajectories{best_traj}.name, beamwidth_matrix(best_arr, best_traj));

top_combinations = [];
for i = 1:length(array_configs)
    for j = 1:length(trajectories)
        top_combinations = [top_combinations; beamwidth_matrix(i,j), i, j];
    end
end
top_combinations = sortrows(top_combinations, 1);
for k = 2:min(3, size(top_combinations, 1))
    i = top_combinations(k, 2);
    j = top_combinations(k, 3);
    improvement = beamwidth_matrix(i, 1) / beamwidth_matrix(i, j);  % vs 静止
    fprintf('   Top%d: %s + %s (%.2f°, 提升%.2fx)\n', k, ...
        array_configs{i}.name, trajectories{j}.name, beamwidth_matrix(i, j), improvement);
end
fprintf('\n');

fprintf('🌐 空间覆盖排名 (采样点越多越好):\n');
[~, best_cov_idx] = max(coverage_matrix(:));
[best_cov_arr, best_cov_traj] = ind2sub(size(coverage_matrix), best_cov_idx);
fprintf('   🥇 最优: %s + %s (%d点)\n', ...
    array_configs{best_cov_arr}.name, trajectories{best_cov_traj}.name, coverage_matrix(best_cov_arr, best_cov_traj));
fprintf('\n');

fprintf('🎯 精度排名 (综合角度误差越小越好):\n');
[~, best_acc_idx] = min(error_matrix(:));
[best_acc_arr, best_acc_traj] = ind2sub(size(error_matrix), best_acc_idx);
fprintf('   🥇 最优: %s + %s (误差%.3f°)\n', ...
    array_configs{best_acc_arr}.name, trajectories{best_acc_traj}.name, error_matrix(best_acc_arr, best_acc_traj));
fprintf('\n');

%% ========================================================================
%% 可视化
%% ========================================================================
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('生成可视化图表\n');
fprintf('═══════════════════════════════════════════════════════\n');

% 图1: 热力图 - 波束宽度
figure('Position', [50, 50, 1200, 400]);
subplot(1,3,1);
imagesc(beamwidth_matrix);
colorbar;
title('波束宽度 (°)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('轨迹');
ylabel('阵列配置');
set(gca, 'XTick', 1:length(trajectories), 'XTickLabel', cellfun(@(x) x.name, trajectories, 'UniformOutput', false), 'XTickLabelRotation', 45);
set(gca, 'YTick', 1:length(array_configs), 'YTickLabel', cellfun(@(x) x.name, array_configs, 'UniformOutput', false));
colormap('jet');

% 图2: 热力图 - 空间覆盖
subplot(1,3,2);
imagesc(coverage_matrix);
colorbar;
title('空间采样点数', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('轨迹');
set(gca, 'XTick', 1:length(trajectories), 'XTickLabel', cellfun(@(x) x.name, trajectories, 'UniformOutput', false), 'XTickLabelRotation', 45);
set(gca, 'YTick', 1:length(array_configs), 'YTickLabel', cellfun(@(x) x.name, array_configs, 'UniformOutput', false));
colormap('jet');

% 图3: 热力图 - DOA误差
subplot(1,3,3);
imagesc(error_matrix);
colorbar;
title('DOA综合误差 (°)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('轨迹');
set(gca, 'XTick', 1:length(trajectories), 'XTickLabel', cellfun(@(x) x.name, trajectories, 'UniformOutput', false), 'XTickLabelRotation', 45);
set(gca, 'YTick', 1:length(array_configs), 'YTickLabel', cellfun(@(x) x.name, array_configs, 'UniformOutput', false));
colormap('jet');

sgtitle('阵列-轨迹组合性能热力图', 'FontSize', 14, 'FontWeight', 'bold');

% 图2: 对比柱状图
figure('Position', [100, 100, 1400, 500]);

% 针对每种阵列，对比不同轨迹
for arr_idx = 1:length(array_configs)
    subplot(2, 2, arr_idx);
    
    bw = beamwidth_matrix(arr_idx, :);
    baseline = bw(1);  % 静止作为基准
    improvement = baseline ./ bw;
    
    bar(improvement);
    hold on;
    yline(1, 'r--', '基准', 'LineWidth', 1.5);
    
    title(array_configs{arr_idx}.name, 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('轨迹', 'FontSize', 10);
    ylabel('分辨率提升倍数', 'FontSize', 10);
    set(gca, 'XTickLabel', cellfun(@(x) x.name, trajectories, 'UniformOutput', false), 'XTickLabelRotation', 45);
    grid on;
    ylim([0, max(improvement)*1.2]);
    
    % 标注数值
    for j = 1:length(bw)
        text(j, improvement(j)+0.1, sprintf('%.2fx\n%.1f°', improvement(j), bw(j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8);
    end
end

sgtitle('各阵列配置下不同轨迹的分辨率提升', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('✓ 图表生成完成\n\n');

%% ========================================================================
%% 结论与建议
%% ========================================================================
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('✅ 实验结论\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

fprintf('最佳组合:\n');
fprintf('  🏆 分辨率最优: %s + %s\n', ...
    array_configs{best_arr}.name, trajectories{best_traj}.name);
fprintf('     - 波束宽度: %.2f°\n', beamwidth_matrix(best_arr, best_traj));
fprintf('     - 相比静止提升: %.2fx\n\n', beamwidth_matrix(best_arr, 1) / beamwidth_matrix(best_arr, best_traj));

fprintf('设计建议:\n');
fprintf('  1. 阵列配置:\n');
avg_improvement = mean(beamwidth_matrix(:, 2:end) ./ beamwidth_matrix(:, 1), 2);
[~, sorted_arr] = sort(avg_improvement, 'descend');
for i = 1:length(array_configs)
    idx = sorted_arr(i);
    fprintf('     %d) %s: 平均提升%.2fx\n', i, array_configs{idx}.name, avg_improvement(idx));
end

fprintf('\n  2. 运动轨迹:\n');
avg_improvement_traj = mean(beamwidth_matrix(:, 2:end) ./ beamwidth_matrix(:, 1), 1);
[~, sorted_traj] = sort(avg_improvement_traj, 'descend');
for i = 1:length(trajectories)-1
    idx = sorted_traj(i) + 1;  % +1因为跳过了静止
    fprintf('     %d) %s: 平均提升%.2fx\n', i, trajectories{idx}.name, avg_improvement_traj(i));
end

fprintf('\n  3. 综合建议:\n');
fprintf('     - 优先选择: %s\n', array_configs{best_arr}.name);
fprintf('     - 推荐轨迹: %s\n', trajectories{best_traj}.name);
fprintf('     - 预期性能: 分辨率提升%.1fx，空间采样%d点\n', ...
    beamwidth_matrix(best_arr, 1) / beamwidth_matrix(best_arr, best_traj), ...
    coverage_matrix(best_arr, best_traj));
fprintf('\n');

