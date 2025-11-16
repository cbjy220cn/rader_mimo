function run_trajectory_array_experiment(radar_params, num_snapshots_base, element_spacing, lambda, ...
    smart_grid, search_grid, output_dir, USE_SMART_SEARCH)
% RUN_TRAJECTORY_ARRAY_EXPERIMENT 运行最优轨迹-阵列组合探索实验
%
% 测试4种阵列配置 × 3种关键轨迹 = 12组组合
% 评估指标：波束宽度、空间覆盖、DOA精度

fprintf('测试: 4种阵列 × 3种轨迹 = 12组组合\n\n');

%% 定义阵列配置（8元）
num_elements = 8;
aperture_size = 0.1;
array_configs = {};

% 配置1: 均匀线阵 (ULA)
rx_ula = zeros(num_elements, 3);
for i = 1:num_elements
    rx_ula(i, :) = [(i-1)*element_spacing - (num_elements-1)*element_spacing/2, 0, 0];
end
array_configs{1}.name = '均匀线阵(ULA)';
array_configs{1}.rx_positions = rx_ula;

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

% 配置3: 矩形阵列 (URA) 2×4
rx_rect = zeros(num_elements, 3);
rows = 2; cols = 4;
idx = 1;
for i = 1:rows
    for j = 1:cols
        rx_rect(idx, :) = [(j-1)*element_spacing - (cols-1)*element_spacing/2, ...
                           (i-1)*element_spacing - (rows-1)*element_spacing/2, 0];
        idx = idx + 1;
    end
end
array_configs{3}.name = '矩形阵列(2×4)';
array_configs{3}.rx_positions = rx_rect;

% 配置4: L型阵列
rx_L = zeros(num_elements, 3);
half = 4;
for i = 1:half
    rx_L(i, :) = [(i-1)*element_spacing, 0, 0];  % 水平臂
end
for i = 1:(num_elements - half)
    rx_L(half + i, :) = [0, i*element_spacing, 0];  % 垂直臂
end
array_configs{4}.name = 'L型阵列';
array_configs{4}.rx_positions = rx_L;

fprintf('阵列配置:\n');
for i = 1:length(array_configs)
    fprintf('  %d. %s\n', i, array_configs{i}.name);
end
fprintf('\n');

%% 定义运动轨迹（关键3种）
trajectories = {};
t_total = (num_snapshots_base - 1) * radar_params.T_chirp;

% 轨迹1: 静止（基准）
trajectories{1}.name = '静止';
trajectories{1}.func = @(t) struct('position', [0,0,0], 'orientation', [0,0,0]);

% 轨迹2: 圆周旋转（360°）
omega_dps = 360 / t_total;
trajectories{2}.name = '圆周旋转';
trajectories{2}.func = @(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]);

% 轨迹3: 螺旋运动（旋转+平移）
velocity = 0.5;
trajectories{3}.name = '螺旋运动';
trajectories{3}.func = @(t) struct('position', [velocity*t, 0, 0], 'orientation', [0, 0, omega_dps * t]);

fprintf('运动轨迹:\n');
for i = 1:length(trajectories)
    fprintf('  %d. %s\n', i, trajectories{i}.name);
end
fprintf('\n');

%% 单目标场景
target_range = 600;
theta_true = 30;
phi_true = 60;
target_pos = [target_range * sind(theta_true) * cosd(phi_true), ...
              target_range * sind(theta_true) * sind(phi_true), ...
              target_range * cosd(theta_true)];
targets = {Target(target_pos, [0,0,0], 1)};

%% 运行所有组合
t_axis = (0:num_snapshots_base-1) * radar_params.T_chirp;
results = cell(length(array_configs), length(trajectories));

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
            if USE_SMART_SEARCH
                [spectrum, ~] = smart_doa_search(estimator, snapshots, t_axis, 1, smart_grid, struct('verbose', false));
            else
                spectrum = estimator.estimate_gmusic(snapshots, t_axis, 1, search_grid);
            end
            [theta_est, phi_est, peak_val] = DoaEstimator.find_peaks(spectrum, search_grid, 1);
        else  % 运动用非相干MUSIC
            estimator = DoaEstimatorIncoherent(array_platform, radar_params);
            if USE_SMART_SEARCH
                [spectrum, ~] = smart_doa_search(estimator, snapshots, t_axis, 1, smart_grid, struct('verbose', false, 'weighting', 'uniform'));
            else
                options.verbose = false;
                options.weighting = 'uniform';
                spectrum = estimator.estimate_incoherent_music(snapshots, t_axis, 1, search_grid, options);
            end
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
        
        % 计算空间覆盖
        virtual_positions = [];
        for k = 1:length(t_axis)
            vp = array_platform.get_mimo_virtual_positions(t_axis(k));
            virtual_positions = [virtual_positions; vp];
        end
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
        
        elapsed = toc;
        fprintf('完成 (%.1fs) [BW:%.1f°, 覆盖:%d点]\n', elapsed, beamwidth, spatial_coverage);
    end
end

%% 分析结果
fprintf('\n分析结果:\n');

% 提取指标矩阵
beamwidth_matrix = zeros(length(array_configs), length(trajectories));
coverage_matrix = zeros(length(array_configs), length(trajectories));

for i = 1:length(array_configs)
    for j = 1:length(trajectories)
        beamwidth_matrix(i, j) = results{i, j}.beamwidth;
        coverage_matrix(i, j) = results{i, j}.spatial_coverage;
    end
end

% 找最优组合
[min_bw, best_idx] = min(beamwidth_matrix(:));
[best_arr, best_traj] = ind2sub(size(beamwidth_matrix), best_idx);
fprintf('  🥇 最优组合: %s + %s (波束宽度%.1f°)\n', ...
    array_configs{best_arr}.name, trajectories{best_traj}.name, min_bw);

improvement = beamwidth_matrix(best_arr, 1) / beamwidth_matrix(best_arr, best_traj);
fprintf('     相比静止提升: %.2fx\n\n', improvement);

%% 可视化
figure('Position', [50, 50, 1200, 400]);

% 图1: 波束宽度热力图
subplot(1,3,1);
imagesc(beamwidth_matrix);
colorbar;
title('波束宽度 (°)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('轨迹');
ylabel('阵列配置');
set(gca, 'XTick', 1:length(trajectories), 'XTickLabel', cellfun(@(x) x.name, trajectories, 'UniformOutput', false));
set(gca, 'YTick', 1:length(array_configs), 'YTickLabel', cellfun(@(x) x.name, array_configs, 'UniformOutput', false));
colormap('jet');

% 图2: 空间覆盖热力图
subplot(1,3,2);
imagesc(coverage_matrix);
colorbar;
title('空间采样点数', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('轨迹');
set(gca, 'XTick', 1:length(trajectories), 'XTickLabel', cellfun(@(x) x.name, trajectories, 'UniformOutput', false));
set(gca, 'YTick', 1:length(array_configs), 'YTickLabel', cellfun(@(x) x.name, array_configs, 'UniformOutput', false));
colormap('jet');

% 图3: 对比柱状图（分辨率提升）
subplot(1,3,3);
improvement_matrix = beamwidth_matrix(:, 1) ./ beamwidth_matrix(:, 2:end);
bar(improvement_matrix');
legend(cellfun(@(x) x.name, array_configs, 'UniformOutput', false), 'Location', 'best');
set(gca, 'XTickLabel', cellfun(@(x) x.name, trajectories(2:end), 'UniformOutput', false));
ylabel('分辨率提升倍数', 'FontSize', 11);
title('相比静止的提升', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

sgtitle('最优轨迹-阵列组合分析', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(output_dir, '5_trajectory_array_analysis.png'));
fprintf('  ✓ 保存: 5_trajectory_array_analysis.png\n');

% 保存结果
save(fullfile(output_dir, 'exp4_trajectory_array_results.mat'), 'results', 'array_configs', 'trajectories', ...
    'beamwidth_matrix', 'coverage_matrix');
fprintf('  ✓ 保存: exp4_trajectory_array_results.mat\n');

end

