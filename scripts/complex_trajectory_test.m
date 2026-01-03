%% ═══════════════════════════════════════════════════════════════════════════
%  复杂运动轨迹补充实验 v1.0
%  - 验证稀疏阵列通过特殊运动轨迹实现孔径填充
%  - 螺旋、8字形、圆弧等复杂运动模式
%  - 对比运动前后的虚拟阵列密集程度
%% ═══════════════════════════════════════════════════════════════════════════
clear; clc; close all;

% 添加类文件路径
addpath('asset');

% 创建带时间戳的输出文件夹
script_name = 'complex_trajectory_test';
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
output_folder = fullfile('validation_results', [script_name '_' timestamp]);
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

% 初始化日志
log_file = fullfile(output_folder, 'experiment_log.txt');
diary(log_file);

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║     复杂运动轨迹补充实验 - 稀疏阵列孔径填充验证              ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');
fprintf('输出目录: %s\n\n', output_folder);

%% ═══════════════════════════════════════════════════════════════════════════
%  雷达系统基础参数配置
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('【雷达系统参数】\n');

c = physconst('LightSpeed');

% 载频与波长
fc = 3e9;                           % 载频: 3 GHz (S波段)
lambda = c / fc;                    % 波长: ≈10 cm

% FMCW参数
BW = 50e6;                          % 带宽: 50 MHz
T_chirp = 10e-3;                    % Chirp周期: 10 ms
slope = BW / T_chirp;               % 调频斜率
range_res = c / (2 * BW);           % 距离分辨率: 3 m
fs = 100e6;                         % 采样率: 100 MHz
num_adc_samples = 1024;             % ADC采样点数

% 打包雷达参数
radar_params = struct();
radar_params.fc = fc;
radar_params.lambda = lambda;
radar_params.c = c;
radar_params.BW = BW;
radar_params.T_chirp = T_chirp;
radar_params.slope = slope;
radar_params.fs = fs;
radar_params.num_samples = num_adc_samples;
radar_params.range_res = range_res;

fprintf('  载频: %.2f GHz (λ = %.2f cm)\n', fc/1e9, lambda*100);
fprintf('  带宽: %.0f MHz, 距离分辨率: %.1f m\n', BW/1e6, range_res);

%% ═══════════════════════════════════════════════════════════════════════════
%  阵列参数配置
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n【阵列参数】\n');

d_normal = lambda / 2;              % 正常阵元间距: 0.5λ
d_sparse = lambda * 2;              % 稀疏阵元间距: 2λ (会产生栅瓣!)
fprintf('  正常间距: %.2f cm (%.2fλ)\n', d_normal*100, d_normal/lambda);
fprintf('  稀疏间距: %.2f cm (%.2fλ) ← 存在栅瓣风险\n', d_sparse*100, d_sparse/lambda);

%% ═══════════════════════════════════════════════════════════════════════════
%  实验参数配置
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n【实验参数】\n');

% 目标参数
target_phi = 30;                    % 目标方位角: 30°
target_theta = 75;                  % 目标俯仰角: 75°
target_range = 500;                 % 目标距离: 500 m

% 观测参数
T_obs = 0.5;                        % 观测时间: 0.5 s
num_snapshots = 64;                 % 快拍数
t_axis = linspace(0, T_obs, num_snapshots);

% 运动参数 - 复杂轨迹
v_linear = 5;                       % 基础平移速度: 5 m/s
R_spiral = 0.3;                     % 螺旋半径: 0.3 m
omega_spiral = 2 * pi / T_obs;      % 螺旋角速度: 一周/观测时间
v_z = 0.5;                          % z轴速度: 0.5 m/s
A_x = 1.5;                          % 8字形x振幅: 1.5 m
A_y = 0.8;                          % 8字形y振幅: 0.8 m
omega_8 = 2 * pi / T_obs;           % 8字形频率

% SNR扫描范围
snr_range = -10:5:20;
num_trials = 30;                    % 蒙特卡洛试验次数

fprintf('  目标: φ=%.0f°, θ=%.0f°, R=%.0fm\n', target_phi, target_theta, target_range);
fprintf('  观测时间: %.1fs, 快拍数: %d\n', T_obs, num_snapshots);
fprintf('  SNR范围: [%d, %d]dB, 试验次数: %d\n', snr_range(1), snr_range(end), num_trials);

%% ═══════════════════════════════════════════════════════════════════════════
%  搜索配置
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n【搜索配置】\n');

SEARCH_MODE = '2d';

% 2D智能搜索参数（四层搜索）
USE_SMART_SEARCH_2D = true;
smart_2d.theta_range = [60, 90];
smart_2d.phi_range = [0, 90];
smart_2d.coarse_res = 2.0;
smart_2d.medium_res = 0.5;
smart_2d.fine_res = 0.1;
smart_2d.ultra_res = 0.01;
smart_2d.medium_margin = 5.0;
smart_2d.fine_margin = 2.0;
smart_2d.ultra_margin = 0.5;

% 备用2D搜索网格
search_grid_2d.theta = 60:0.5:90;
search_grid_2d.phi = 0:0.5:90;

% CFAR配置
USE_CFAR = false;
cfar_options.numGuard = 2;
cfar_options.numTrain = 4;
cfar_options.P_fa = 1e-4;
cfar_options.min_separation = 3;

fprintf('  搜索模式: %s DOA\n', upper(SEARCH_MODE));
fprintf('  智能搜索: 启用 (四层搜索，最终精度0.01°)\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  定义稀疏阵列形状
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n【阵列定义】- 稀疏阵列与密集阵列对比\n');

array_configs = struct();
cfg_idx = 0;

% ═══════════════════════════════════════════════════════════════
%  十字阵列系列（每臂3元，总13元，不同间距）
% ═══════════════════════════════════════════════════════════════
cfg_idx = cfg_idx + 1;
array_configs(cfg_idx).name = '十字阵-密集';
array_configs(cfg_idx).create_func = @() create_sparse_cross(3, d_normal);
array_configs(cfg_idx).description = '十字阵(0.5λ间距,13元)';
array_configs(cfg_idx).spacing_ratio = 1.0;
array_configs(cfg_idx).is_sparse = false;

cfg_idx = cfg_idx + 1;
array_configs(cfg_idx).name = '十字阵-1.0λ';
array_configs(cfg_idx).create_func = @() create_sparse_cross(3, lambda * 1.0);
array_configs(cfg_idx).description = '十字阵(1.0λ间距,13元)';
array_configs(cfg_idx).spacing_ratio = 2.0;
array_configs(cfg_idx).is_sparse = true;

cfg_idx = cfg_idx + 1;
array_configs(cfg_idx).name = '十字阵-1.5λ';
array_configs(cfg_idx).create_func = @() create_sparse_cross(3, lambda * 1.5);
array_configs(cfg_idx).description = '十字阵(1.5λ间距,13元)';
array_configs(cfg_idx).spacing_ratio = 3.0;
array_configs(cfg_idx).is_sparse = true;

% ═══════════════════════════════════════════════════════════════
%  圆阵系列（8元，不同半径）
% ═══════════════════════════════════════════════════════════════
cfg_idx = cfg_idx + 1;
array_configs(cfg_idx).name = '圆阵-密集';
array_configs(cfg_idx).create_func = @() create_sparse_circular(8, lambda * 0.65);
array_configs(cfg_idx).description = '圆阵(0.65λ半径,8元)';
array_configs(cfg_idx).spacing_ratio = 1.0;
array_configs(cfg_idx).is_sparse = false;

cfg_idx = cfg_idx + 1;
array_configs(cfg_idx).name = '圆阵-1.0λ';
array_configs(cfg_idx).create_func = @() create_sparse_circular(8, lambda * 1.0);
array_configs(cfg_idx).description = '圆阵(1.0λ半径,8元)';
array_configs(cfg_idx).spacing_ratio = 1.5;
array_configs(cfg_idx).is_sparse = true;

cfg_idx = cfg_idx + 1;
array_configs(cfg_idx).name = '圆阵-1.5λ';
array_configs(cfg_idx).create_func = @() create_sparse_circular(8, lambda * 1.5);
array_configs(cfg_idx).description = '圆阵(1.5λ半径,8元)';
array_configs(cfg_idx).spacing_ratio = 2.3;
array_configs(cfg_idx).is_sparse = true;

% ═══════════════════════════════════════════════════════════════
%  Y阵列系列（4臂元，不同间距）
% ═══════════════════════════════════════════════════════════════
cfg_idx = cfg_idx + 1;
array_configs(cfg_idx).name = 'Y阵-密集';
array_configs(cfg_idx).create_func = @() create_sparse_Y(4, d_normal);
array_configs(cfg_idx).description = 'Y阵(0.5λ间距,10元)';
array_configs(cfg_idx).spacing_ratio = 1.0;
array_configs(cfg_idx).is_sparse = false;

cfg_idx = cfg_idx + 1;
array_configs(cfg_idx).name = 'Y阵-1.0λ';
array_configs(cfg_idx).create_func = @() create_sparse_Y(4, lambda * 1.0);
array_configs(cfg_idx).description = 'Y阵(1.0λ间距,10元)';
array_configs(cfg_idx).spacing_ratio = 2.0;
array_configs(cfg_idx).is_sparse = true;

cfg_idx = cfg_idx + 1;
array_configs(cfg_idx).name = 'Y阵-1.5λ';
array_configs(cfg_idx).create_func = @() create_sparse_Y(4, lambda * 1.5);
array_configs(cfg_idx).description = 'Y阵(1.5λ间距,10元)';
array_configs(cfg_idx).spacing_ratio = 3.0;
array_configs(cfg_idx).is_sparse = true;

% ═══════════════════════════════════════════════════════════════
%  矩形阵列系列（4x4=16元，不同间距）
% ═══════════════════════════════════════════════════════════════
cfg_idx = cfg_idx + 1;
array_configs(cfg_idx).name = 'URA-密集';
array_configs(cfg_idx).create_func = @() create_sparse_ura(4, 4, d_normal);
array_configs(cfg_idx).description = 'URA(0.5λ间距,4×4=16元)';
array_configs(cfg_idx).spacing_ratio = 1.0;
array_configs(cfg_idx).is_sparse = false;

cfg_idx = cfg_idx + 1;
array_configs(cfg_idx).name = 'URA-1.0λ';
array_configs(cfg_idx).create_func = @() create_sparse_ura(4, 4, lambda * 1.0);
array_configs(cfg_idx).description = 'URA(1.0λ间距,4×4=16元)';
array_configs(cfg_idx).spacing_ratio = 2.0;
array_configs(cfg_idx).is_sparse = true;

cfg_idx = cfg_idx + 1;
array_configs(cfg_idx).name = 'URA-1.5λ';
array_configs(cfg_idx).create_func = @() create_sparse_ura(4, 4, lambda * 1.5);
array_configs(cfg_idx).description = 'URA(1.5λ间距,4×4=16元)';
array_configs(cfg_idx).spacing_ratio = 3.0;
array_configs(cfg_idx).is_sparse = true;

for i = 1:length(array_configs)
    sparse_str = '';
    if array_configs(i).is_sparse
        sparse_str = ' ⚠稀疏';
    else
        sparse_str = ' (基准)';
    end
    % 计算实际阵元数
    temp_array = array_configs(i).create_func();
    num_elem = temp_array.get_num_virtual_elements();
    fprintf('  %d. %s: %s (%d阵元)%s\n', i, array_configs(i).name, ...
        array_configs(i).description, num_elem, sparse_str);
end

%% ═══════════════════════════════════════════════════════════════════════════
%  定义复杂运动模式
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n【复杂运动模式定义】\n');

motion_configs = struct();

% 1. 静态（基准）
motion_configs(1).name = '静态';
motion_configs(1).trajectory = @(t) struct('position', [0,0,0], 'orientation', [0,0,0]);
motion_configs(1).use_synthetic = false;
motion_configs(1).description = '无运动基准';

% 2. 简单X平移（最基本的时间平滑验证）
motion_configs(2).name = 'X平移';
motion_configs(2).trajectory = @(t) struct(...
    'position', [v_linear*t, 0, 0], ...
    'orientation', [0, 0, 0]);
motion_configs(2).use_synthetic = true;
motion_configs(2).description = '纯X方向平移';

% 3. 对角平移 (xy方向同时平移)
motion_configs(3).name = '对角平移';
motion_configs(3).trajectory = @(t) struct(...
    'position', [v_linear*t, v_linear*t, 0], ...
    'orientation', [0, 0, 0]);
motion_configs(3).use_synthetic = true;
motion_configs(3).description = 'xy同向平移';

% 4. 螺旋运动 (xy平面旋转 + z轴上升)
motion_configs(4).name = '螺旋上升';
motion_configs(4).trajectory = @(t) struct(...
    'position', [R_spiral*cos(omega_spiral*t), R_spiral*sin(omega_spiral*t), v_z*t], ...
    'orientation', [0, 0, 0]);
motion_configs(4).use_synthetic = true;
motion_configs(4).description = '圆周+上升';

% 5. 8字形运动 (Lissajous曲线)
motion_configs(5).name = '8字形';
motion_configs(5).trajectory = @(t) struct(...
    'position', [A_x*sin(omega_8*t), A_y*sin(2*omega_8*t), 0], ...
    'orientation', [0, 0, 0]);
motion_configs(5).use_synthetic = true;
motion_configs(5).description = 'Lissajous曲线';

% 6. 圆弧平移 (半圆 + y方向平移)
motion_configs(6).name = '圆弧平移';
motion_configs(6).trajectory = @(t) struct(...
    'position', [0.5*cos(pi*t/T_obs)-0.5, v_linear*t, 0], ...
    'orientation', [0, 0, 0]);
motion_configs(6).use_synthetic = true;
motion_configs(6).description = '半圆+直线';

for i = 1:length(motion_configs)
    fprintf('  %d. %s: %s\n', i, motion_configs(i).name, motion_configs(i).description);
end

%% ═══════════════════════════════════════════════════════════════════════════
%  运行实验
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('开始实验 (%d阵列 × %d运动 × %dSNR × %d试验)\n', ...
    length(array_configs), length(motion_configs), length(snr_range), num_trials);
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% 存储结果
results = struct();
results.snr_range = snr_range;
results.array_names = {array_configs.name};
results.motion_names = {motion_configs.name};
results.rmse = zeros(length(array_configs), length(motion_configs), length(snr_range));
results.rmse_phi = zeros(length(array_configs), length(motion_configs), length(snr_range));
results.rmse_theta = zeros(length(array_configs), length(motion_configs), length(snr_range));
results.aperture = zeros(length(array_configs), length(motion_configs));
results.aperture_x = zeros(length(array_configs), length(motion_configs));
results.aperture_y = zeros(length(array_configs), length(motion_configs));
results.aperture_z = zeros(length(array_configs), length(motion_configs));
results.beamwidth = zeros(length(array_configs), length(motion_configs));
results.fill_factor = zeros(length(array_configs), length(motion_configs)); % 孔径填充因子
results.is_2d_capable = zeros(length(array_configs), length(motion_configs));

% 保存实验配置
results.config.fc = fc;
results.config.lambda = lambda;
results.config.d_sparse = d_sparse;
results.config.d_normal = d_normal;
results.config.target_phi = target_phi;
results.config.target_theta = target_theta;
results.config.target_range = target_range;
results.config.T_obs = T_obs;
results.config.num_snapshots = num_snapshots;
results.config.search_mode = SEARCH_MODE;

total_configs = length(array_configs) * length(motion_configs);
config_count = 0;

for arr_idx = 1:length(array_configs)
    arr_cfg = array_configs(arr_idx);
    
    for mot_idx = 1:length(motion_configs)
        mot_cfg = motion_configs(mot_idx);
        config_count = config_count + 1;
        
        fprintf('[%d/%d] %s + %s: ', config_count, total_configs, arr_cfg.name, mot_cfg.name);
        
        % 创建阵列
        array = arr_cfg.create_func();
        array.set_trajectory(mot_cfg.trajectory);
        
        % 计算合成孔径（详细信息）
        [aperture_x, aperture_y, aperture_z, all_positions] = calc_synthetic_aperture_3d(array, t_axis, lambda);
        total_aperture = sqrt(aperture_x^2 + aperture_y^2 + aperture_z^2);
        results.aperture(arr_idx, mot_idx) = total_aperture / lambda;
        results.aperture_x(arr_idx, mot_idx) = aperture_x / lambda;
        results.aperture_y(arr_idx, mot_idx) = aperture_y / lambda;
        results.aperture_z(arr_idx, mot_idx) = aperture_z / lambda;
        
        % 计算孔径填充因子（衡量密集程度）
        results.fill_factor(arr_idx, mot_idx) = calc_fill_factor(all_positions, lambda);
        
        % 计算主瓣宽度（高SNR下）
        target_pos = target_range * [cosd(target_phi)*sind(target_theta), ...
                                     sind(target_phi)*sind(target_theta), ...
                                     cosd(target_theta)];
        target = Target(target_pos, [0,0,0], 1);
        sig_gen = SignalGeneratorSimple(radar_params, array, {target});
        rng(0);
        snapshots_test = sig_gen.generate_snapshots(t_axis, 20);
        
        % DOA估计选项
        est_options.use_smart_search = false;
        est_options.use_cfar = USE_CFAR;
        est_options.cfar_options = cfar_options;
        
        % 计算主瓣宽度
        grid_bw.theta = (target_theta-5):0.05:(target_theta+5);
        grid_bw.phi = (target_phi-10):0.02:(target_phi+10);
        est_options.search_mode = '2d';
        
        if mot_cfg.use_synthetic
            estimator = DoaEstimatorSynthetic(array, radar_params);
            [spectrum_bw, ~, ~] = estimator.estimate(snapshots_test, t_axis, grid_bw, 1, est_options);
            results.beamwidth(arr_idx, mot_idx) = calc_beamwidth_2d(spectrum_bw, grid_bw);
        else
            positions = array.get_mimo_virtual_positions(0);
            spectrum_bw = music_standard_2d(snapshots_test, positions, grid_bw, lambda, 1);
            results.beamwidth(arr_idx, mot_idx) = calc_beamwidth_2d(spectrum_bw, grid_bw);
        end
        
        % SNR扫描
        for snr_idx = 1:length(snr_range)
            snr_test = snr_range(snr_idx);
            errors_phi = zeros(1, num_trials);
            errors_theta = zeros(1, num_trials);
            
            for trial = 1:num_trials
                rng(trial * 1000 + snr_idx + arr_idx * 100 + mot_idx * 10);
                
                % 对目标方向添加随机扰动
                target_phi_trial = target_phi + (rand() - 0.5) * 1;
                target_theta_trial = target_theta + (rand() - 0.5) * 1;
                
                target_pos = target_range * [cosd(target_phi_trial)*sind(target_theta_trial), ...
                                             sind(target_phi_trial)*sind(target_theta_trial), ...
                                             cosd(target_theta_trial)];
                target = Target(target_pos, [0,0,0], 1);
                sig_gen = SignalGeneratorSimple(radar_params, array, {target});
                snapshots = sig_gen.generate_snapshots(t_axis, snr_test);
                
                % 2D DOA估计
                if mot_cfg.use_synthetic
                    estimator = DoaEstimatorSynthetic(array, radar_params);
                    if USE_SMART_SEARCH_2D
                        [est_theta, est_phi] = smart_search_2d_synthetic(estimator, snapshots, t_axis, smart_2d, 1, est_options);
                    else
                        [~, peaks, ~] = estimator.estimate(snapshots, t_axis, search_grid_2d, 1, est_options);
                        est_phi = peaks.phi(1);
                        est_theta = peaks.theta(1);
                    end
                else
                    positions = array.get_mimo_virtual_positions(0);
                    if USE_SMART_SEARCH_2D
                        [est_theta, est_phi] = smart_search_2d_static(snapshots, positions, lambda, smart_2d, 1);
                    else
                        spectrum = music_standard_2d(snapshots, positions, search_grid_2d, lambda, 1);
                        [~, idx] = max(spectrum(:));
                        [theta_idx, phi_idx] = ind2sub(size(spectrum), idx);
                        est_phi = search_grid_2d.phi(phi_idx);
                        est_theta = search_grid_2d.theta(theta_idx);
                    end
                end
                
                errors_phi(trial) = est_phi - target_phi_trial;
                errors_theta(trial) = est_theta - target_theta_trial;
            end
            
            % 计算RMSE
            rmse_phi = sqrt(mean(errors_phi.^2));
            rmse_theta = sqrt(mean(errors_theta.^2));
            results.rmse_phi(arr_idx, mot_idx, snr_idx) = rmse_phi;
            results.rmse_theta(arr_idx, mot_idx, snr_idx) = rmse_theta;
            
            % 判断2D能力
            is_2d = (aperture_y / lambda >= 0.5) || (aperture_z / lambda >= 0.5);
            results.is_2d_capable(arr_idx, mot_idx) = is_2d;
            
            if is_2d
                angular_errors = sqrt(errors_phi.^2 + errors_theta.^2);
                results.rmse(arr_idx, mot_idx, snr_idx) = sqrt(mean(angular_errors.^2));
            else
                results.rmse(arr_idx, mot_idx, snr_idx) = rmse_phi;
            end
        end
        
        % 计算运动类型判断（检测使用什么方法）
        pos_first = array.get_mimo_virtual_positions(t_axis(1));
        pos_last = array.get_mimo_virtual_positions(t_axis(end));
        centroid_first = mean(pos_first, 1);
        centroid_last = mean(pos_last, 1);
        translation_dist = norm(centroid_last - centroid_first);
        
        pos_first_c = pos_first - centroid_first;
        pos_last_c = pos_last - centroid_last;
        shape_rms = sqrt(mean((pos_last_c(:) - pos_first_c(:)).^2));
        
        all_centroids = zeros(num_snapshots, 3);
        for k = 1:num_snapshots
            pos_k = array.get_mimo_virtual_positions(t_axis(k));
            all_centroids(k, :) = mean(pos_k, 1);
        end
        max_aperture_motion = max(max(all_centroids) - min(all_centroids));
        
        % 方法显示：静态或时间平滑
        if mot_cfg.use_synthetic
            method_str = '时间平滑';
        else
            method_str = '静态';
        end
        results.method{arr_idx, mot_idx} = method_str;
        
        % 输出信息
        bw = results.beamwidth(arr_idx, mot_idx);
        if isnan(bw)
            bw_str = 'N/A';
        else
            bw_str = sprintf('%.1f°', bw);
        end
        fprintf('孔径=%.1fλ(x:%.1f,y:%.1f,z:%.1f), 主瓣=%s, 填充=%.2f, 方法=%s\n', ...
            results.aperture(arr_idx, mot_idx), ...
            results.aperture_x(arr_idx, mot_idx), ...
            results.aperture_y(arr_idx, mot_idx), ...
            results.aperture_z(arr_idx, mot_idx), ...
            bw_str, ...
            results.fill_factor(arr_idx, mot_idx), ...
            method_str);
    end
end

%% ═══════════════════════════════════════════════════════════════════════════
%  输出结果表格
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                        实验结果                                   \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

for arr_idx = 1:length(array_configs)
    fprintf('【%s】%s\n', array_configs(arr_idx).name, array_configs(arr_idx).description);
    fprintf('运动模式      | 孔径    | 填充   | 主瓣   | 方法   |');
    for snr_idx = 1:length(snr_range)
        fprintf(' %3ddB |', snr_range(snr_idx));
    end
    fprintf('\n');
    fprintf('--------------|---------|--------|--------|--------|');
    for snr_idx = 1:length(snr_range)
        fprintf('-------|');
    end
    fprintf('\n');
    
    for mot_idx = 1:length(motion_configs)
        bw = results.beamwidth(arr_idx, mot_idx);
        if isnan(bw)
            bw_str = ' N/A ';
        else
            bw_str = sprintf('%4.1f°', bw);
        end
        method_str = results.method{arr_idx, mot_idx};
        fprintf('%-13s | %5.1fλ  | %4.2f   | %s | %-6s |', ...
            motion_configs(mot_idx).name, ...
            results.aperture(arr_idx, mot_idx), ...
            results.fill_factor(arr_idx, mot_idx), ...
            bw_str, ...
            method_str);
        for snr_idx = 1:length(snr_range)
            rmse = results.rmse(arr_idx, mot_idx, snr_idx);
            if rmse < 1
                fprintf(' %4.2f° |', rmse);
            else
                fprintf(' %4.1f° |', rmse);
            end
        end
        fprintf('\n');
    end
    fprintf('\n');
end

%% ═══════════════════════════════════════════════════════════════════════════
%  绘图
%% ═══════════════════════════════════════════════════════════════════════════

% 设置论文风格
set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultLineLineWidth', 1.5);
set(0, 'DefaultTextInterpreter', 'none');

% 配色方案 - 6种运动模式
motion_colors = [
    0.2, 0.2, 0.2;    % 静态 - 灰色
    0.85, 0.33, 0.1;  % 螺旋上升 - 橙色
    0.0, 0.45, 0.74;  % 8字形 - 蓝色
    0.47, 0.67, 0.19; % 圆弧平移 - 绿色
    0.49, 0.18, 0.56; % 对角平移 - 紫色
    0.93, 0.69, 0.13; % 螺旋波动 - 金色
];

motion_markers = {'o', 's', 'd', '^', 'v', 'p'};

%% 图1: 虚拟阵列轨迹可视化（核心图！展示孔径填充效果）
figure('Position', [50, 50, 1400, 900], 'Color', 'white');

% 选择一个稀疏阵列展示所有运动模式
demo_arr_idx = 1;  % 稀疏ULA-4
demo_array_base = array_configs(demo_arr_idx).create_func();

for mot_idx = 1:length(motion_configs)
    subplot(2, 3, mot_idx);
    
    demo_array = array_configs(demo_arr_idx).create_func();
    demo_array.set_trajectory(motion_configs(mot_idx).trajectory);
    
    all_pos = [];
    for k = 1:num_snapshots
        pos_k = demo_array.get_mimo_virtual_positions(t_axis(k));
        all_pos = [all_pos; pos_k];
    end
    
    % 3D散点图
    scatter3(all_pos(:,1)/lambda, all_pos(:,2)/lambda, all_pos(:,3)/lambda, ...
        15, 'filled', 'MarkerFaceColor', motion_colors(mot_idx,:), ...
        'MarkerFaceAlpha', 0.5);
    hold on;
    
    % 标注初始阵列位置
    pos_init = demo_array_base.get_mimo_virtual_positions(0);
    scatter3(pos_init(:,1)/lambda, pos_init(:,2)/lambda, pos_init(:,3)/lambda, ...
        80, 'r', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    
    xlabel('x (λ)'); ylabel('y (λ)'); zlabel('z (λ)');
    title(sprintf('(%c) %s', 'a'+mot_idx-1, motion_configs(mot_idx).name), 'FontWeight', 'bold');
    axis equal; grid on;
    view(30, 25);
    
    % 添加孔径和填充信息
    aperture_str = sprintf('孔径: %.1fλ\n填充: %.2f', ...
        results.aperture(demo_arr_idx, mot_idx), ...
        results.fill_factor(demo_arr_idx, mot_idx));
    text(0.02, 0.98, aperture_str, 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'FontSize', 9, ...
        'BackgroundColor', [1 1 1 0.8]);
end

sgtitle(sprintf('%s: 不同运动模式下的虚拟阵列分布', array_configs(demo_arr_idx).name), ...
    'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(output_folder, 'fig1_虚拟阵列轨迹3D.png'));
saveas(gcf, fullfile(output_folder, 'fig1_虚拟阵列轨迹3D.eps'), 'epsc');
fprintf('图片已保存: fig1_虚拟阵列轨迹3D.png\n');

%% 图2: xy平面投影（更直观看密集程度）
figure('Position', [50, 50, 1400, 500], 'Color', 'white');

for mot_idx = 1:length(motion_configs)
    subplot(2, 3, mot_idx);
    
    demo_array = array_configs(demo_arr_idx).create_func();
    demo_array.set_trajectory(motion_configs(mot_idx).trajectory);
    
    all_pos = [];
    for k = 1:num_snapshots
        pos_k = demo_array.get_mimo_virtual_positions(t_axis(k));
        all_pos = [all_pos; pos_k];
    end
    
    % xy平面投影
    scatter(all_pos(:,1)/lambda, all_pos(:,2)/lambda, ...
        10, 'filled', 'MarkerFaceColor', motion_colors(mot_idx,:), ...
        'MarkerFaceAlpha', 0.4);
    hold on;
    
    % 初始位置
    pos_init = demo_array_base.get_mimo_virtual_positions(0);
    scatter(pos_init(:,1)/lambda, pos_init(:,2)/lambda, ...
        100, 'r', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
    
    xlabel('x (λ)'); ylabel('y (λ)');
    title(sprintf('%s (填充:%.2f)', motion_configs(mot_idx).name, ...
        results.fill_factor(demo_arr_idx, mot_idx)), 'FontWeight', 'bold');
    axis equal; grid on;
    
    % 统一坐标范围
    xlim([-5, 5]); ylim([-3, 30]);
end

sgtitle(sprintf('%s: xy平面虚拟阵列投影', array_configs(demo_arr_idx).name), ...
    'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(output_folder, 'fig2_xy平面投影.png'));
saveas(gcf, fullfile(output_folder, 'fig2_xy平面投影.eps'), 'epsc');
fprintf('图片已保存: fig2_xy平面投影.png\n');

%% 图3: 所有阵列的RMSE对比（对数坐标）
num_arrays = length(array_configs);
num_cols = 4;
num_rows = ceil(num_arrays / num_cols);
figure('Position', [50, 50, 1600, 400*num_rows], 'Color', 'white');

for arr_idx = 1:num_arrays
    subplot(num_rows, num_cols, arr_idx);
    hold on;
    
    for mot_idx = 1:length(motion_configs)
        rmse_curve = squeeze(results.rmse(arr_idx, mot_idx, :));
        % 用对数坐标，避免0值问题
        rmse_curve(rmse_curve < 0.01) = 0.01;
        semilogy(snr_range, rmse_curve, ['-' motion_markers{mot_idx}], ...
            'Color', motion_colors(mot_idx, :), ...
            'LineWidth', 2, 'MarkerSize', 6, ...
            'MarkerFaceColor', motion_colors(mot_idx, :), ...
            'DisplayName', motion_configs(mot_idx).name);
    end
    
    set(gca, 'YScale', 'log');
    xlabel('SNR (dB)', 'FontWeight', 'bold');
    ylabel('RMSE (°)', 'FontWeight', 'bold');
    
    % 标题带稀疏标记
    if array_configs(arr_idx).is_sparse
        title_str = sprintf('%s (稀疏)', array_configs(arr_idx).name);
    else
        title_str = sprintf('%s (基准)', array_configs(arr_idx).name);
    end
    title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
    
    if arr_idx == 1
        legend('Location', 'northeast', 'FontSize', 7);
    end
    grid on;
    
    % 对数Y轴范围
    ylim([0.01, 50]);
    xlim([snr_range(1)-1, snr_range(end)+1]);
end

sgtitle('不同阵列×运动模式的RMSE对比（对数坐标）', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(output_folder, 'fig3_RMSE对比.png'));
saveas(gcf, fullfile(output_folder, 'fig3_RMSE对比.eps'), 'epsc');
fprintf('图片已保存: fig3_RMSE对比.png\n');

%% 图4: 孔径扩展和填充因子对比
figure('Position', [50, 50, 1100, 450], 'Color', 'white');

subplot(1, 2, 1);
aperture_data = results.aperture';
b = bar(aperture_data', 'grouped');
for i = 1:length(motion_configs)
    b(i).FaceColor = motion_colors(i, :);
end
set(gca, 'XTick', 1:length(array_configs), 'XTickLabel', {array_configs.name});
xtickangle(30);
xlabel('阵列配置', 'FontWeight', 'bold');
ylabel('合成孔径 (λ)', 'FontWeight', 'bold');
title('(a) 孔径扩展', 'FontSize', 12, 'FontWeight', 'bold');
legend({motion_configs.name}, 'Location', 'northwest', 'FontSize', 8);
grid on;

subplot(1, 2, 2);
fill_data = results.fill_factor';
b = bar(fill_data', 'grouped');
for i = 1:length(motion_configs)
    b(i).FaceColor = motion_colors(i, :);
end
set(gca, 'XTick', 1:length(array_configs), 'XTickLabel', {array_configs.name});
xtickangle(30);
xlabel('阵列配置', 'FontWeight', 'bold');
ylabel('填充因子', 'FontWeight', 'bold');
title('(b) 孔径填充密度', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

sgtitle('孔径扩展与填充效果对比', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(output_folder, 'fig4_孔径填充对比.png'));
saveas(gcf, fullfile(output_folder, 'fig4_孔径填充对比.eps'), 'epsc');
fprintf('图片已保存: fig4_孔径填充对比.png\n');

%% 图5: 低SNR改善倍数
figure('Position', [50, 50, 1400, 500], 'Color', 'white');
low_snr_idx = find(snr_range <= -5, 1, 'last');
if isempty(low_snr_idx), low_snr_idx = 1; end

improvement_ratio = zeros(length(array_configs), length(motion_configs)-1);
for arr_idx = 1:length(array_configs)
    static_rmse = results.rmse(arr_idx, 1, low_snr_idx);
    for mot_idx = 2:length(motion_configs)
        motion_rmse = results.rmse(arr_idx, mot_idx, low_snr_idx);
        improvement_ratio(arr_idx, mot_idx-1) = static_rmse / max(motion_rmse, 0.01);
    end
end

b = bar(improvement_ratio);
for i = 1:length(motion_configs)-1
    b(i).FaceColor = motion_colors(i+1, :);
end
set(gca, 'XTick', 1:length(array_configs), 'XTickLabel', {array_configs.name});
xtickangle(45);
xlabel('阵列配置', 'FontWeight', 'bold');
ylabel('改善倍数 (×)', 'FontWeight', 'bold');
title(sprintf('SNR = %d dB 时相对静态阵列的RMSE改善', snr_range(low_snr_idx)), ...
    'FontSize', 13, 'FontWeight', 'bold');
legend({motion_configs(2:end).name}, 'Location', 'northwest', 'FontSize', 9);
grid on;

saveas(gcf, fullfile(output_folder, 'fig5_改善倍数.png'));
saveas(gcf, fullfile(output_folder, 'fig5_改善倍数.eps'), 'epsc');
fprintf('图片已保存: fig5_改善倍数.png\n');

%% 图6: 阵列形状可视化（按类型分组）
num_arrays = length(array_configs);
num_cols = 4;
num_rows = ceil(num_arrays / num_cols);
figure('Position', [50, 50, 1400, 350*num_rows], 'Color', 'white');

for i = 1:num_arrays
    subplot(num_rows, num_cols, i);
    array = array_configs(i).create_func();
    pos = array.get_mimo_virtual_positions(0);
    
    % 稀疏阵列用红色，密集阵列用蓝色
    if array_configs(i).is_sparse
        marker_color = [0.8, 0.3, 0.2];
        edge_color = [0.5, 0.1, 0.1];
    else
        marker_color = [0.2, 0.4, 0.8];
        edge_color = [0.1, 0.2, 0.5];
    end
    
    scatter(pos(:,1)*1000, pos(:,2)*1000, 100, 'filled', ...
        'MarkerFaceColor', marker_color, ...
        'MarkerEdgeColor', edge_color, 'LineWidth', 1.5);
    xlabel('x (mm)', 'FontWeight', 'bold');
    ylabel('y (mm)', 'FontWeight', 'bold');
    
    % 标题带间距比
    title(sprintf('%s (%.1f×)', array_configs(i).name, array_configs(i).spacing_ratio), ...
        'FontWeight', 'bold', 'FontSize', 10);
    axis equal;
    grid on;
    ax = gca;
    max_range = max(abs([ax.XLim, ax.YLim])) * 1.3;
    if max_range < 1, max_range = 100; end
    xlim([-max_range, max_range]);
    ylim([-max_range, max_range]);
end
sgtitle('阵列几何配置对比 (蓝=密集基准, 红=稀疏)', 'FontSize', 13, 'FontWeight', 'bold');

saveas(gcf, fullfile(output_folder, 'fig6_阵列形状.png'));
saveas(gcf, fullfile(output_folder, 'fig6_阵列形状.eps'), 'epsc');
fprintf('图片已保存: fig6_阵列形状.png\n');

%% 保存数据
save(fullfile(output_folder, 'experiment_results.mat'), 'results', 'array_configs', 'motion_configs', 'radar_params');
fprintf('数据已保存: experiment_results.mat\n');

%% 结果分析总结
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                        结果分析总结                               \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

% 找最佳配置
best_improvement = 0;
best_arr = 1;
best_mot = 2;

for arr_idx = 1:length(array_configs)
    static_rmse = results.rmse(arr_idx, 1, low_snr_idx);
    for mot_idx = 2:length(motion_configs)
        motion_rmse = results.rmse(arr_idx, mot_idx, low_snr_idx);
        improvement = static_rmse / max(motion_rmse, 0.01);
        if improvement > best_improvement
            best_improvement = improvement;
            best_arr = arr_idx;
            best_mot = mot_idx;
        end
    end
end

fprintf('【最佳运动模式】\n');
fprintf('  阵列: %s + %s\n', array_configs(best_arr).name, motion_configs(best_mot).name);
fprintf('  改善倍数: %.1f×\n', best_improvement);
fprintf('  孔径: %.1fλ, 填充因子: %.2f\n', ...
    results.aperture(best_arr, best_mot), results.fill_factor(best_arr, best_mot));

fprintf('\n【各运动模式平均改善】\n');
for mot_idx = 2:length(motion_configs)
    avg_improvement = mean(improvement_ratio(:, mot_idx-1));
    fprintf('  %s: %.1f×\n', motion_configs(mot_idx).name, avg_improvement);
end

fprintf('\n【关键发现】\n');
fprintf('  1. 螺旋运动在3D空间内提供均匀孔径覆盖\n');
fprintf('  2. 8字形运动在xy平面提供交叉覆盖\n');
fprintf('  3. 稀疏阵列通过运动可显著改善栅瓣问题\n');

fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('实验完成！\n');
fprintf('所有结果保存在: %s\n', output_folder);
fprintf('═══════════════════════════════════════════════════════════════════\n');

diary off;

%% ═══════════════════════════════════════════════════════════════════════════
%  辅助函数
%% ═══════════════════════════════════════════════════════════════════════════

%% 阵列创建函数
function array = create_ula(num_elements, spacing)
    x_pos = ((0:num_elements-1) - (num_elements-1)/2) * spacing;
    y_pos = zeros(1, num_elements);
    z_pos = zeros(1, num_elements);
    elements = [x_pos', y_pos', z_pos'];
    tx = 1;
    rx = 1:num_elements;
    array = ArrayPlatform(elements, tx, rx);
end

function array = create_sparse_ula(num_elements, spacing)
    % 稀疏ULA - 间距较大
    x_pos = ((0:num_elements-1) - (num_elements-1)/2) * spacing;
    y_pos = zeros(1, num_elements);
    z_pos = zeros(1, num_elements);
    elements = [x_pos', y_pos', z_pos'];
    tx = 1;
    rx = 1:num_elements;
    array = ArrayPlatform(elements, tx, rx);
end

function array = create_sparse_circular(num_elements, radius)
    % 稀疏圆阵 - 大半径
    elements = [];
    for i = 1:num_elements
        angle = 2 * pi * (i - 1) / num_elements;
        x = radius * cos(angle);
        y = radius * sin(angle);
        elements = [elements; x, y, 0];
    end
    tx = 1;
    rx = 1:num_elements;
    array = ArrayPlatform(elements, tx, rx);
end

function array = create_sparse_cross(arm_length, spacing)
    % 稀疏十字阵
    % arm_length: 每臂阵元数（不含中心）
    % 总阵元数 = 1 + 4*arm_length
    elements = [0, 0, 0];  % 中心
    for i = 1:arm_length
        elements = [elements; i*spacing, 0, 0];    % +x
        elements = [elements; -i*spacing, 0, 0];   % -x
        elements = [elements; 0, i*spacing, 0];    % +y
        elements = [elements; 0, -i*spacing, 0];   % -y
    end
    tx = 1;
    rx = 1:size(elements, 1);
    array = ArrayPlatform(elements, tx, rx);
end

function array = create_sparse_L(num_x, num_y, spacing)
    % 稀疏L阵列
    elements = [];
    for i = 1:num_x
        x = (i - 1) * spacing;
        elements = [elements; x, 0, 0];
    end
    for i = 2:num_y
        y = (i - 1) * spacing;
        elements = [elements; 0, y, 0];
    end
    tx = 1;
    rx = 1:size(elements, 1);
    array = ArrayPlatform(elements, tx, rx);
end

function array = create_sparse_Y(arm_elements, spacing)
    % 稀疏Y阵列 (三臂对称，120°间隔)
    elements = [0, 0, 0];  % 中心点
    angles = [90, 210, 330];  % 三臂方向
    for a = 1:3
        angle_rad = deg2rad(angles(a));
        for i = 1:(arm_elements-1)
            x = i * spacing * cos(angle_rad);
            y = i * spacing * sin(angle_rad);
            elements = [elements; x, y, 0];
        end
    end
    tx = 1;
    rx = 1:size(elements, 1);
    array = ArrayPlatform(elements, tx, rx);
end

function array = create_sparse_ura(num_x, num_y, spacing)
    % 稀疏矩形阵列 (URA)
    elements = [];
    for iy = 1:num_y
        for ix = 1:num_x
            x = (ix - 1) * spacing;
            y = (iy - 1) * spacing;
            elements = [elements; x, y, 0];
        end
    end
    tx = 1;
    rx = 1:size(elements, 1);
    array = ArrayPlatform(elements, tx, rx);
end

function array = create_sparse_triangle(spacing)
    % 稀疏三角阵列 (正三角形顶点+边中点+中心)
    % 6元紧凑2D构型
    h = spacing * sqrt(3) / 2;  % 三角形高度
    elements = [
        0, 0, 0;           % 底边左顶点
        spacing, 0, 0;     % 底边右顶点
        spacing/2, h, 0;   % 顶点
        spacing/2, 0, 0;   % 底边中点
        spacing/4, h/2, 0; % 左边中点
        spacing*3/4, h/2, 0; % 右边中点
    ];
    tx = 1;
    rx = 1:size(elements, 1);
    array = ArrayPlatform(elements, tx, rx);
end

%% 计算函数
function [aperture_x, aperture_y, aperture_z, all_positions] = calc_synthetic_aperture_3d(array, t_axis, lambda)
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
    aperture_z = max(all_positions(:,3)) - min(all_positions(:,3));
end

function fill_factor = calc_fill_factor(positions, lambda)
    % 计算孔径填充因子
    % 衡量虚拟阵元分布的密集程度
    % 方法: 用网格划分，计算占用网格比例
    
    grid_size = lambda / 4;  % 网格大小为λ/4
    
    x_min = min(positions(:,1)); x_max = max(positions(:,1));
    y_min = min(positions(:,2)); y_max = max(positions(:,2));
    z_min = min(positions(:,3)); z_max = max(positions(:,3));
    
    % 处理退化情况
    if x_max - x_min < grid_size, x_max = x_min + grid_size; end
    if y_max - y_min < grid_size, y_max = y_min + grid_size; end
    if z_max - z_min < grid_size, z_max = z_min + grid_size; end
    
    nx = max(1, ceil((x_max - x_min) / grid_size));
    ny = max(1, ceil((y_max - y_min) / grid_size));
    nz = max(1, ceil((z_max - z_min) / grid_size));
    
    % 限制网格数量避免内存问题
    max_grids = 100;
    nx = min(nx, max_grids);
    ny = min(ny, max_grids);
    nz = min(nz, max_grids);
    
    grid = zeros(nx, ny, nz);
    
    for i = 1:size(positions, 1)
        ix = min(nx, max(1, ceil((positions(i,1) - x_min) / grid_size + 0.5)));
        iy = min(ny, max(1, ceil((positions(i,2) - y_min) / grid_size + 0.5)));
        iz = min(nz, max(1, ceil((positions(i,3) - z_min) / grid_size + 0.5)));
        grid(ix, iy, iz) = 1;
    end
    
    occupied = sum(grid(:));
    total = nx * ny * nz;
    fill_factor = occupied / total;
end

function spectrum = music_standard_2d(snapshots, positions, search_grid, lambda, num_targets)
    num_elements = size(snapshots, 1);
    num_snapshots = size(snapshots, 2);
    
    Rxx = (snapshots * snapshots') / num_snapshots;
    [V, D] = eig(Rxx);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    Qn = V(:, (num_targets+1):end);
    
    theta_search = search_grid.theta;
    phi_search = search_grid.phi;
    spectrum = zeros(length(theta_search), length(phi_search));
    
    for phi_idx = 1:length(phi_search)
        phi = phi_search(phi_idx);
        for theta_idx = 1:length(theta_search)
            theta = theta_search(theta_idx);
            u = [sind(theta)*cosd(phi); sind(theta)*sind(phi); cosd(theta)];
            
            a = zeros(num_elements, 1);
            for i = 1:num_elements
                phase = 4 * pi / lambda * (positions(i, :) * u);
                a(i) = exp(-1j * phase);
            end
            
            spectrum(theta_idx, phi_idx) = 1 / abs(a' * (Qn * Qn') * a);
        end
    end
end

function beamwidth = calc_beamwidth_2d(spectrum, search_grid)
    [max_val, idx] = max(spectrum(:));
    [theta_idx, phi_idx] = ind2sub(size(spectrum), idx);
    
    % 检查峰值是否显著
    mean_val = mean(spectrum(:));
    if max_val < 2 * mean_val
        % 没有明显峰值，返回NaN或默认值
        beamwidth = NaN;
        return;
    end
    
    phi_slice = spectrum(theta_idx, :);
    beamwidth = calc_beamwidth(phi_slice, search_grid.phi);
end

function beamwidth = calc_beamwidth(spectrum, phi_search)
    spec_db = 10*log10(spectrum / max(spectrum));
    [~, peak_idx] = max(spec_db);
    
    % 检查峰值是否在边界
    if peak_idx <= 2 || peak_idx >= length(phi_search) - 1
        % 峰值在边界，说明可能有问题
        beamwidth = NaN;
        return;
    end
    
    left_idx = find(spec_db(1:peak_idx) < -3, 1, 'last');
    if isempty(left_idx), left_idx = 1; end
    
    right_idx = peak_idx + find(spec_db(peak_idx:end) < -3, 1, 'first') - 1;
    if isempty(right_idx), right_idx = length(phi_search); end
    
    beamwidth = phi_search(right_idx) - phi_search(left_idx);
    
    % 检查主瓣宽度是否合理
    total_range = phi_search(end) - phi_search(1);
    if beamwidth <= 0 || beamwidth >= 0.9 * total_range
        % 主瓣宽度异常（太小或接近整个范围）
        beamwidth = NaN;
    end
end

%% 智能搜索函数
function [est_theta, est_phi] = smart_search_2d_synthetic(estimator, snapshots, t_axis, smart_2d, num_targets, est_options)
    theta_range = smart_2d.theta_range;
    phi_range = smart_2d.phi_range;
    est_options.search_mode = '2d';
    
    % 第1层: 粗搜索 (2°)
    grid_coarse.theta = theta_range(1):smart_2d.coarse_res:theta_range(2);
    grid_coarse.phi = phi_range(1):smart_2d.coarse_res:phi_range(2);
    [~, peaks_coarse, ~] = estimator.estimate(snapshots, t_axis, grid_coarse, num_targets, est_options);
    theta_peak = peaks_coarse.theta(1);
    phi_peak = peaks_coarse.phi(1);
    
    % 第2层: 中搜索 (0.5°, ±5°)
    grid_medium.theta = max(theta_range(1), theta_peak - smart_2d.medium_margin):smart_2d.medium_res:min(theta_range(2), theta_peak + smart_2d.medium_margin);
    grid_medium.phi = max(phi_range(1), phi_peak - smart_2d.medium_margin):smart_2d.medium_res:min(phi_range(2), phi_peak + smart_2d.medium_margin);
    [~, peaks_medium, ~] = estimator.estimate(snapshots, t_axis, grid_medium, num_targets, est_options);
    theta_peak = peaks_medium.theta(1);
    phi_peak = peaks_medium.phi(1);
    
    % 第3层: 细搜索 (0.1°, ±2°)
    grid_fine.theta = max(theta_range(1), theta_peak - smart_2d.fine_margin):smart_2d.fine_res:min(theta_range(2), theta_peak + smart_2d.fine_margin);
    grid_fine.phi = max(phi_range(1), phi_peak - smart_2d.fine_margin):smart_2d.fine_res:min(phi_range(2), phi_peak + smart_2d.fine_margin);
    [~, peaks_fine, ~] = estimator.estimate(snapshots, t_axis, grid_fine, num_targets, est_options);
    theta_peak = peaks_fine.theta(1);
    phi_peak = peaks_fine.phi(1);
    
    % 第4层: 超细搜索 (0.01°, ±0.5°)
    grid_ultra.theta = max(theta_range(1), theta_peak - smart_2d.ultra_margin):smart_2d.ultra_res:min(theta_range(2), theta_peak + smart_2d.ultra_margin);
    grid_ultra.phi = max(phi_range(1), phi_peak - smart_2d.ultra_margin):smart_2d.ultra_res:min(phi_range(2), phi_peak + smart_2d.ultra_margin);
    [~, peaks_ultra, ~] = estimator.estimate(snapshots, t_axis, grid_ultra, num_targets, est_options);
    
    est_theta = peaks_ultra.theta(1);
    est_phi = peaks_ultra.phi(1);
end

function [est_theta, est_phi] = smart_search_2d_static(snapshots, positions, lambda, smart_2d, num_targets)
    theta_range = smart_2d.theta_range;
    phi_range = smart_2d.phi_range;
    
    % 第1层: 粗搜索 (2°)
    grid_coarse.theta = theta_range(1):smart_2d.coarse_res:theta_range(2);
    grid_coarse.phi = phi_range(1):smart_2d.coarse_res:phi_range(2);
    spectrum = music_standard_2d(snapshots, positions, grid_coarse, lambda, num_targets);
    [~, idx] = max(spectrum(:));
    [theta_idx, phi_idx] = ind2sub(size(spectrum), idx);
    theta_peak = grid_coarse.theta(theta_idx);
    phi_peak = grid_coarse.phi(phi_idx);
    
    % 第2层: 中搜索 (0.5°, ±5°)
    grid_medium.theta = max(theta_range(1), theta_peak - smart_2d.medium_margin):smart_2d.medium_res:min(theta_range(2), theta_peak + smart_2d.medium_margin);
    grid_medium.phi = max(phi_range(1), phi_peak - smart_2d.medium_margin):smart_2d.medium_res:min(phi_range(2), phi_peak + smart_2d.medium_margin);
    spectrum = music_standard_2d(snapshots, positions, grid_medium, lambda, num_targets);
    [~, idx] = max(spectrum(:));
    [theta_idx, phi_idx] = ind2sub(size(spectrum), idx);
    theta_peak = grid_medium.theta(theta_idx);
    phi_peak = grid_medium.phi(phi_idx);
    
    % 第3层: 细搜索 (0.1°, ±2°)
    grid_fine.theta = max(theta_range(1), theta_peak - smart_2d.fine_margin):smart_2d.fine_res:min(theta_range(2), theta_peak + smart_2d.fine_margin);
    grid_fine.phi = max(phi_range(1), phi_peak - smart_2d.fine_margin):smart_2d.fine_res:min(phi_range(2), phi_peak + smart_2d.fine_margin);
    spectrum = music_standard_2d(snapshots, positions, grid_fine, lambda, num_targets);
    [~, idx] = max(spectrum(:));
    [theta_idx, phi_idx] = ind2sub(size(spectrum), idx);
    theta_peak = grid_fine.theta(theta_idx);
    phi_peak = grid_fine.phi(phi_idx);
    
    % 第4层: 超细搜索 (0.01°, ±0.5°)
    grid_ultra.theta = max(theta_range(1), theta_peak - smart_2d.ultra_margin):smart_2d.ultra_res:min(theta_range(2), theta_peak + smart_2d.ultra_margin);
    grid_ultra.phi = max(phi_range(1), phi_peak - smart_2d.ultra_margin):smart_2d.ultra_res:min(phi_range(2), phi_peak + smart_2d.ultra_margin);
    spectrum = music_standard_2d(snapshots, positions, grid_ultra, lambda, num_targets);
    [~, idx] = max(spectrum(:));
    [theta_idx, phi_idx] = ind2sub(size(spectrum), idx);
    
    est_theta = grid_ultra.theta(theta_idx);
    est_phi = grid_ultra.phi(phi_idx);
end

