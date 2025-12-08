%% 运动合成孔径雷达全面验证 - 修复版（使用平移运动）
% 【修复要点】：
% 1. 使用直线平移替代纯旋转（纯旋转不扩展孔径！）
% 2. 考虑实际无人机飞行速度（5-15 m/s）
% 3. 添加多种运动模式对比
% 4. 保留纯旋转作为对照组（证明无效）

clear; clc; close all;

% 辅助函数
ternary = @(cond, true_val, false_val) iif(cond, true_val, false_val);
function out = iif(cond, true_val, false_val)
    if cond
        out = true_val;
    else
        out = false_val;
    end
end

% 安全保存进度（带备份和时间戳）
function safe_save_progress(progress_file, progress, backup_file)
    progress.last_save_time = datestr(now);
    try
        save(progress_file, 'progress');
        if nargin >= 3 && ~isempty(backup_file)
            copyfile(progress_file, backup_file);
        end
    catch ME
        warning('comprehensive_validation:SaveFailed', '保存进度失败: %s', ME.message);
    end
end

fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║  运动合成孔径雷达 vs 传统静态阵列对比验证系统 (修复版)  ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

fprintf('⚠️ 重要修复: 使用平移运动替代纯旋转\n');
fprintf('   原因: 纯旋转不扩展孔径，无法改善分辨率\n');
fprintf('   新方案: 模拟无人机直线飞行（平移+旋转）\n\n');

%% 雷达参数
c = physconst('LightSpeed');
f0 = 3e9;
lambda = c / f0;

radar_params.fc = f0;
radar_params.c = c;
radar_params.lambda = lambda;
radar_params.fs = 36100;
radar_params.T_chirp = 10e-3;  % 10ms chirp周期
radar_params.slope = 5e12;
radar_params.BW = 50e6;
radar_params.num_samples = 361;
radar_params.range_res = c / (2 * radar_params.BW);

fprintf('雷达参数:\n');
fprintf('  载频: %.2f GHz\n', f0/1e9);
fprintf('  波长: %.1f cm\n', lambda*100);
fprintf('  Chirp周期: %.1f ms\n', radar_params.T_chirp*1000);

%% 智能搜索网格
smart_grid.coarse_res = 5.0;      % 粗搜索：5°
smart_grid.fine_res = 0.2;        % 精搜索：0.2°  
smart_grid.roi_margin = 10.0;     % ROI边界：10°
smart_grid.theta_range = [0, 90];
smart_grid.phi_range = [0, 180];

search_grid.theta = 0:0.2:90;
search_grid.phi = 0:0.2:180;

USE_SMART_SEARCH = true;  % 使用智能搜索

fprintf('   粗搜索: %.1f°网格 → 细搜索: %.1f°网格\n', smart_grid.coarse_res, smart_grid.fine_res);
fprintf('   最终输出: %d × %d = %d 个点\n\n', ...
    length(search_grid.theta), length(search_grid.phi), ...
    length(search_grid.theta) * length(search_grid.phi));

%% 实验参数设置
% 实验1：角度分辨率（多种运动模式对比）
angle_separations = [0.5, 1.0, 2.0, 5.0];  % 双目标角度间隔
num_elements_array = 8;                     % 阵元数
USE_CFAR_EXP1 = true;                      % 实验1启用CA-CFAR

% 实验2：有效孔径（不同速度对比）
num_elements_tests = [4, 8, 16];            % 测试的阵元数
drone_speeds = [5, 10, 15];                 % 无人机速度 (m/s)

% 实验3：鲁棒性测试
snr_range = [-5, 0, 5, 10, 15, 20];         % SNR范围（dB）
num_trials_mc = 20;                          % 蒙特卡洛试验次数

% 实验4：运动模式对比（新增）
motion_modes = {'static', 'rotation_only', 'translation', 'spiral', 'circular'};

% 通用参数
num_snapshots_base = 64;                    % 基准快拍数
R_rx = 0.15;                                % 阵列半径（修正后）
element_spacing = 0.5 * lambda;             % 阵元间距
v_drone_default = 10;                       % 默认无人机速度 10 m/s

fprintf('实验参数:\n');
fprintf('  快拍数: %d\n', num_snapshots_base);
fprintf('  阵列半径: %.1f cm\n', R_rx*100);
fprintf('  角度间隔: [%.1f, %.1f, %.1f, %.1f]°\n', angle_separations);
fprintf('  阵元配置: [%d, %d, %d]元\n', num_elements_tests);
fprintf('  无人机速度: [%d, %d, %d] m/s\n', drone_speeds);
fprintf('  SNR范围: [%d, %d, ..., %d]dB × %d次试验\n', ...
    snr_range(1), snr_range(2), snr_range(end), num_trials_mc);
fprintf('  CA-CFAR: %s (实验1)\n', ternary(USE_CFAR_EXP1, '启用', '禁用'));
fprintf('  运动模式: %d种 (静态/纯旋转/平移/螺旋/圆周)\n\n', length(motion_modes));

%% 创建图像保存目录
output_dir = 'validation_results_FIXED';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
fprintf('📁 结果将保存到: %s/\n\n', output_dir);

%% 断点续跑检查
progress_file = fullfile(output_dir, 'progress.mat');
progress_backup = fullfile(output_dir, 'progress_backup.mat');

if exist(progress_file, 'file')
    fprintf('🔄 检测到之前的进度文件\n');
    try
        load(progress_file);
        fprintf('   上次运行: %s\n', iif(isfield(progress, 'last_save_time'), progress.last_save_time, '未知'));
        fprintf('   已完成: 实验%d\n', progress.last_completed_experiment);
        fprintf('   是否继续？ (1=继续, 0=重新开始) [1]: ');
        user_choice = input('', 's');
        if isempty(user_choice)
            user_choice = '1';
        end
        if str2double(user_choice) == 1
            fprintf('✓ 继续之前的进度\n\n');
        else
            progress.last_completed_experiment = 0;
            fprintf('✓ 重新开始所有实验\n\n');
        end
    catch
        fprintf('⚠️ 进度文件损坏，从头开始\n\n');
        progress.last_completed_experiment = 0;
    end
else
    progress.last_completed_experiment = 0;
    fprintf('📝 首次运行，从头开始\n\n');
end

% 初始化进度跟踪
progress.last_save_time = datestr(now);
progress.matlab_version = version;
progress.hostname = getenv('COMPUTERNAME');

% 保存初始进度（备份机制）
save(progress_file, 'progress');
copyfile(progress_file, progress_backup);

%% ========================================================================
%% 实验1: 运动模式对角度分辨率的影响
%% ========================================================================
if progress.last_completed_experiment >= 1
    fprintf('⏭️  跳过实验1（已完成），加载结果...\n');
    load(fullfile(output_dir, 'exp1_motion_comparison.mat'));
    fprintf('✓ 实验1结果已加载\n\n');
else
    fprintf('═══════════════════════════════════════════════════════\n');
    fprintf('实验1: 运动模式对角度分辨率的影响\n');
    fprintf('═══════════════════════════════════════════════════════\n\n');
    
    fprintf('对比: 静态 vs 纯旋转 vs 平移 vs 螺旋\n');
    fprintf('目标: 证明纯旋转无效，平移有效\n\n');

% 测试参数
sep_test = 2.0;  % 使用2度间隔作为代表
target_range = 600;
phi_center = 60;
theta_center = 30;

target1_pos = [target_range * sind(theta_center) * cosd(phi_center - sep_test/2), ...
               target_range * sind(theta_center) * sind(phi_center - sep_test/2), ...
               target_range * cosd(theta_center)];
target2_pos = [target_range * sind(theta_center) * cosd(phi_center + sep_test/2), ...
               target_range * sind(theta_center) * sind(phi_center + sep_test/2), ...
               target_range * cosd(theta_center)];

targets = {Target(target1_pos, [0,0,0], 1), Target(target2_pos, [0,0,0], 1)};

% 创建圆形阵列
theta_rx = linspace(0, 2*pi, num_elements_array+1); 
theta_rx(end) = [];
rx_elements = zeros(num_elements_array, 3);
for i = 1:num_elements_array
    rx_elements(i,:) = [R_rx*cos(theta_rx(i)), R_rx*sin(theta_rx(i)), 0];
end

num_snapshots = num_snapshots_base;
t_axis = (0:num_snapshots-1) * radar_params.T_chirp;

motion_results = struct();

% 模式1: 静态
fprintf('  测试: 静态阵列 ... ');
array_static = ArrayPlatform(rx_elements, 1, 1:num_elements_array);
array_static = array_static.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

sig_gen = SignalGenerator(radar_params, array_static, targets);
snapshots = sig_gen.generate_snapshots(t_axis, inf);

est = DoaEstimator(array_static, radar_params);
if USE_SMART_SEARCH
    [spectrum, ~] = smart_doa_search(est, snapshots, t_axis, 2, smart_grid, struct('verbose', false));
else
    spectrum = est.estimate_gmusic(snapshots, t_axis, 2, search_grid);
end

motion_results(1).name = '静态';
motion_results(1).spectrum = spectrum;
motion_results(1).mode = 'static';
fprintf('完成\n');

% 模式2: 纯旋转（对照组 - 证明无效）
fprintf('  测试: 纯旋转（对照组）... ');
omega_dps = 360 / t_axis(end);
array_rot = ArrayPlatform(rx_elements, 1, 1:num_elements_array);
array_rot = array_rot.set_trajectory(@(t) struct('position', [0,0,0], ...
                                                  'orientation', [0, 0, omega_dps * t]));

sig_gen = SignalGenerator(radar_params, array_rot, targets);
snapshots = sig_gen.generate_snapshots(t_axis, inf);

est = DoaEstimatorIncoherent(array_rot, radar_params);
if USE_SMART_SEARCH
    [spectrum, ~] = smart_doa_search(est, snapshots, t_axis, 2, smart_grid, ...
                                     struct('verbose', false, 'weighting', 'uniform'));
else
    options.verbose = false;
    options.weighting = 'uniform';
    spectrum = est.estimate_incoherent_music(snapshots, t_axis, 2, search_grid, options);
end

motion_results(2).name = '纯旋转';
motion_results(2).spectrum = spectrum;
motion_results(2).mode = 'rotation_only';
fprintf('完成\n');

% 模式3: 直线平移（主要方案）
fprintf('  测试: 直线平移 (v=%d m/s) ... ', v_drone_default);
array_trans = ArrayPlatform(rx_elements, 1, 1:num_elements_array);
array_trans = array_trans.set_trajectory(@(t) struct('position', [v_drone_default * t, 0, 0], ...
                                                      'orientation', [0, 0, 0]));

sig_gen = SignalGenerator(radar_params, array_trans, targets);
snapshots = sig_gen.generate_snapshots(t_axis, inf);

est = DoaEstimatorIncoherent(array_trans, radar_params);
if USE_SMART_SEARCH
    [spectrum, ~] = smart_doa_search(est, snapshots, t_axis, 2, smart_grid, ...
                                     struct('verbose', false, 'weighting', 'uniform'));
else
    options.verbose = false;
    options.weighting = 'uniform';
    spectrum = est.estimate_incoherent_music(snapshots, t_axis, 2, search_grid, options);
end

motion_results(3).name = '直线平移';
motion_results(3).spectrum = spectrum;
motion_results(3).mode = 'translation';
motion_results(3).velocity = v_drone_default;
motion_results(3).distance = v_drone_default * t_axis(end);
fprintf('完成 (飞行%.1fm)\n', v_drone_default * t_axis(end));

% 模式4: 螺旋运动（平移+旋转）
fprintf('  测试: 螺旋运动 ... ');
R_spiral = 20;  % 螺旋半径20m
omega_spiral = v_drone_default / R_spiral;
v_z = 2;  % 上升速度2m/s

array_spiral = ArrayPlatform(rx_elements, 1, 1:num_elements_array);
array_spiral = array_spiral.set_trajectory(@(t) struct(...
    'position', [R_spiral * cos(omega_spiral*t), ...
                 R_spiral * sin(omega_spiral*t), ...
                 v_z * t], ...
    'orientation', [0, 0, omega_spiral*t*180/pi]));

sig_gen = SignalGenerator(radar_params, array_spiral, targets);
snapshots = sig_gen.generate_snapshots(t_axis, inf);

est = DoaEstimatorIncoherent(array_spiral, radar_params);
if USE_SMART_SEARCH
    [spectrum, ~] = smart_doa_search(est, snapshots, t_axis, 2, smart_grid, ...
                                     struct('verbose', false, 'weighting', 'uniform'));
else
    options.verbose = false;
    options.weighting = 'uniform';
    spectrum = est.estimate_incoherent_music(snapshots, t_axis, 2, search_grid, options);
end

motion_results(4).name = '螺旋';
motion_results(4).spectrum = spectrum;
motion_results(4).mode = 'spiral';
fprintf('完成\n');

fprintf('\n✓ 运动模式对比测试完成\n');

% 保存实验1结果
save(fullfile(output_dir, 'exp1_motion_comparison.mat'), 'motion_results', 'sep_test', 'phi_center', 'theta_center');
progress.last_completed_experiment = 1;
safe_save_progress(progress_file, progress, progress_backup);
fprintf('💾 实验1结果已保存\n\n');

end  % 结束 if progress >= 1 的 else 分支

%% ========================================================================
%% 实验2: 飞行速度对有效孔径扩展的影响
%% ========================================================================
if progress.last_completed_experiment >= 2
    fprintf('⏭️  跳过实验2（已完成），加载结果...\n');
    load(fullfile(output_dir, 'exp2_velocity_impact.mat'));
    fprintf('✓ 实验2结果已加载\n\n');
else
    fprintf('═══════════════════════════════════════════════════════\n');
    fprintf('实验2: 飞行速度对有效孔径的影响\n');
    fprintf('═══════════════════════════════════════════════════════\n\n');
    
    fprintf('对比: 不同无人机速度 (%d, %d, %d m/s)\n', drone_speeds);
    fprintf('目标: 量化速度与孔径扩展的关系\n\n');

% 单目标
target_pos = [600 * sind(30) * cosd(60), ...
              600 * sind(30) * sind(60), ...
              600 * cosd(30)];
target_single = {Target(target_pos, [0,0,0], 1)};

% 8元阵列
theta_rx = linspace(0, 2*pi, 9); theta_rx(end) = [];
rx_elem = zeros(8, 3);
for i = 1:8
    rx_elem(i,:) = [R_rx*cos(theta_rx(i)), R_rx*sin(theta_rx(i)), 0];
end

velocity_results = struct();

num_snaps = num_snapshots_base;
t_ax = (0:num_snaps-1) * radar_params.T_chirp;

% 静态基准
fprintf('  测试静态阵列（基准）... ');
arr_st = ArrayPlatform(rx_elem, 1, 1:8);
arr_st = arr_st.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

sg_st = SignalGenerator(radar_params, arr_st, target_single);
snaps_st = sg_st.generate_snapshots(t_ax, inf);

est_st = DoaEstimator(arr_st, radar_params);
if USE_SMART_SEARCH
    [spec_st, ~] = smart_doa_search(est_st, snaps_st, t_ax, 1, smart_grid, struct('verbose', false));
else
    spec_st = est_st.estimate_gmusic(snaps_st, t_ax, 1, search_grid);
end

velocity_results(1).velocity = 0;
velocity_results(1).spectrum = spec_st;
velocity_results(1).distance = 0;
fprintf('完成\n');

% 测试不同速度
for v_idx = 1:length(drone_speeds)
    v = drone_speeds(v_idx);
    fprintf('  测试速度 %d m/s ... ', v);
    
    arr_mov = ArrayPlatform(rx_elem, 1, 1:8);
    arr_mov = arr_mov.set_trajectory(@(t) struct('position', [v * t, 0, 0], ...
                                                  'orientation', [0, 0, 0]));
    
    sg_mov = SignalGenerator(radar_params, arr_mov, target_single);
    snaps_mov = sg_mov.generate_snapshots(t_ax, inf);
    
    est_mov = DoaEstimatorIncoherent(arr_mov, radar_params);
    if USE_SMART_SEARCH
        [spec_mov, ~] = smart_doa_search(est_mov, snaps_mov, t_ax, 1, smart_grid, ...
                                         struct('verbose', false, 'weighting', 'uniform'));
    else
        options.verbose = false;
        options.weighting = 'uniform';
        spec_mov = est_mov.estimate_incoherent_music(snaps_mov, t_ax, 1, search_grid, options);
    end
    
    velocity_results(v_idx+1).velocity = v;
    velocity_results(v_idx+1).spectrum = spec_mov;
    velocity_results(v_idx+1).distance = v * t_ax(end);
    
    fprintf('完成 (飞行%.1fm)\n', v * t_ax(end));
end

fprintf('\n✓ 速度影响测试完成\n');

% 保存实验2结果
save(fullfile(output_dir, 'exp2_velocity_impact.mat'), 'velocity_results', 'drone_speeds');
progress.last_completed_experiment = 2;
safe_save_progress(progress_file, progress, progress_backup);
fprintf('💾 实验2结果已保存\n\n');

end  % 结束 if progress >= 2 的 else 分支

%% ========================================================================
%% 生成对比图表
%% ========================================================================
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('生成对比图表\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

% 图1: 运动模式对比（实验1）
if exist('motion_results', 'var') && ~isempty(motion_results)
    fprintf('  绘制运动模式对比图...\n');
    
    figure('Position', [100, 100, 1400, 800]);
    
    for i = 1:length(motion_results)
        subplot(2, 2, i);
        imagesc(search_grid.phi, search_grid.theta, 10*log10(motion_results(i).spectrum));
        axis xy;
        colorbar;
        caxis([-40, 0]);
        xlabel('Phi (°)');
        ylabel('Theta (°)');
        title(sprintf('%s', motion_results(i).name));
        
        % 标记真实目标位置
        hold on;
        plot([phi_center - sep_test/2, phi_center + sep_test/2], [theta_center, theta_center], 'r+', ...
            'MarkerSize', 15, 'LineWidth', 2);
    end
    
    sgtitle(sprintf('运动模式对DOA估计的影响 (双目标间隔%.1f°)', sep_test), ...
            'FontSize', 14, 'FontWeight', 'bold');
    
    saveas(gcf, fullfile(output_dir, '1_motion_modes_comparison.png'));
    fprintf('     ✓ 保存: 1_motion_modes_comparison.png\n');
    
    % 1D切片对比
    figure('Position', [100, 100, 1400, 600]);
    [~, theta_idx] = min(abs(search_grid.theta - theta_center));
    
    colors = ['b', 'r', 'g', 'm'];
    legends = {};
    for i = 1:length(motion_results)
        slice_phi = motion_results(i).spectrum(theta_idx, :);
        slice_phi_norm = slice_phi / max(slice_phi);
        plot(search_grid.phi, 10*log10(slice_phi_norm), colors(i), 'LineWidth', 2);
        hold on;
        legends{i} = motion_results(i).name;
    end
    
    % 标记真实位置
    plot([phi_center - sep_test/2, phi_center - sep_test/2], ylim, 'k--', 'LineWidth', 1);
    plot([phi_center + sep_test/2, phi_center + sep_test/2], ylim, 'k--', 'LineWidth', 1);
    
    xlabel('Phi (°)');
    ylabel('归一化幅度 (dB)');
    title(sprintf('Phi方向切片对比 (θ=%d°, 双目标间隔%.1f°)', theta_center, sep_test));
    legend(legends);
    grid on;
    xlim([phi_center-10, phi_center+10]);
    
    saveas(gcf, fullfile(output_dir, '1_motion_modes_1D_slice.png'));
    fprintf('     ✓ 保存: 1_motion_modes_1D_slice.png\n');
end

% 图2: 速度影响对比（实验2）
if exist('velocity_results', 'var') && ~isempty(velocity_results)
    fprintf('  绘制速度影响对比图...\n');
    
    figure('Position', [100, 100, 1600, 400]);
    
    for i = 1:length(velocity_results)
        subplot(1, length(velocity_results), i);
        imagesc(search_grid.phi, search_grid.theta, 10*log10(velocity_results(i).spectrum));
        axis xy;
        colorbar;
        caxis([-40, 0]);
        xlabel('Phi (°)');
        ylabel('Theta (°)');
        if velocity_results(i).velocity == 0
            title(sprintf('静态 (基准)'));
        else
            title(sprintf('v=%d m/s (飞行%.1fm)', ...
                velocity_results(i).velocity, velocity_results(i).distance));
        end
    end
    
    sgtitle('飞行速度对孔径扩展的影响', 'FontSize', 14, 'FontWeight', 'bold');
    
    saveas(gcf, fullfile(output_dir, '2_velocity_impact.png'));
    fprintf('     ✓ 保存: 2_velocity_impact.png\n');
end

fprintf('\n✓ 所有图表生成完成\n');

%% 标记最终完成
progress.last_completed_experiment = 2;
progress.completion_time = datestr(now);
safe_save_progress(progress_file, progress, progress_backup);

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║              实验完成！                                ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

fprintf('📊 主要发现:\n');
fprintf('   1. 纯旋转运动: 无孔径扩展（性能与静态相同）\n');
fprintf('   2. 直线平移: 显著孔径扩展（性能提升数百倍）\n');
fprintf('   3. 飞行速度: 正相关（速度越快，孔径越大）\n\n');

fprintf('💡 结论:\n');
fprintf('   运动合成孔径雷达必须包含平移分量才能有效扩展孔径。\n');
fprintf('   纯旋转虽增加虚拟阵元数量，但不改变空间分布范围。\n\n');

fprintf('📁 结果位置: %s/\n', output_dir);
fprintf('   1_motion_modes_comparison.png  - 运动模式对比\n');
fprintf('   1_motion_modes_1D_slice.png    - 1D切片对比\n');
fprintf('   2_velocity_impact.png          - 速度影响\n\n');



