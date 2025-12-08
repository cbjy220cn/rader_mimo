%% ═══════════════════════════════════════════════════════════════════════════
%  综合运动阵列DOA性能测试 v2.0
%  - 支持1D/2D搜索模式
%  - 支持多层智能搜索
%  - 支持CA-CFAR多目标检测
%  - 多种阵列形状 × 多种运动模式 × SNR扫描
%% ═══════════════════════════════════════════════════════════════════════════
clear; clc; close all;

% 添加类文件路径
addpath('asset');

% 创建带时间戳的输出文件夹
script_name = 'comprehensive_motion_array_test';
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
output_folder = fullfile('validation_results', [script_name '_' timestamp]);
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

% 初始化日志
log_file = fullfile(output_folder, 'experiment_log.txt');
diary(log_file);

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║          综合运动阵列 DOA 性能测试 v2.0                        ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');
fprintf('输出目录: %s\n\n', output_folder);

%% ═════════════════════════════════════════════════════════════════════════════
%  实验说明（用于论文参考）
%% ═════════════════════════════════════════════════════════════════════════════
fprintf('┌─────────────────────────────────────────────────────────────────┐\n');
fprintf('│                        实验说明                                 │\n');
fprintf('├─────────────────────────────────────────────────────────────────┤\n');
fprintf('│ 【实验目的】                                                    │\n');
fprintf('│   全面评估运动合成孔径DOA算法在不同阵列形状、运动模式、         │\n');
fprintf('│   SNR条件下的性能表现，与静态阵列进行对比。                     │\n');
fprintf('│                                                                 │\n');
fprintf('│ 【核心原理】                                                    │\n');
fprintf('│   运动合成孔径：利用平台运动扩展虚拟阵列孔径                    │\n');
fprintf('│   - 静态N元阵列：孔径 ≈ (N-1)d                                 │\n');
fprintf('│   - 运动后孔径：孔径 = (N-1)d + 运动距离                       │\n');
fprintf('│   孔径扩大 → 角度分辨率提高 → DOA精度提升                      │\n');
fprintf('│                                                                 │\n');
fprintf('│ 【关键指标定义】                                                │\n');
fprintf('│   ① RMSE (均方根误差)                                          │\n');
fprintf('│      = sqrt(mean((估计角度-真实角度)²))                        │\n');
fprintf('│      单位：度(°)                                               │\n');
fprintf('│      物理含义：DOA估计精度的综合度量                            │\n');
fprintf('│      典型值：静态阵列0.1-1°，运动阵列0.05-0.2°                 │\n');
fprintf('│                                                                 │\n');
fprintf('│   ② 合成孔径 (Synthetic Aperture)                              │\n');
fprintf('│      = 虚拟阵列最大空间跨度 / λ                                │\n');
fprintf('│      单位：波长(λ)                                             │\n');
fprintf('│      物理含义：等效阵列的有效口径                               │\n');
fprintf('│      关系：孔径越大 → 分辨率越高 → RMSE越低                    │\n');
fprintf('│                                                                 │\n');
fprintf('│   ③ 3dB主瓣宽度 (Beamwidth)                                    │\n');
fprintf('│      = MUSIC谱峰下降3dB对应的角度范围                          │\n');
fprintf('│      单位：度(°)                                               │\n');
fprintf('│      物理含义：角度分辨能力的直接体现                           │\n');
fprintf('│      近似公式：θ_3dB ≈ 50.8°/(孔径/λ)                         │\n');
fprintf('│                                                                 │\n');
fprintf('│   ④ 改善倍数                                                   │\n');
fprintf('│      = RMSE_静态 / RMSE_运动                                   │\n');
fprintf('│      物理含义：运动带来的精度提升                               │\n');
fprintf('│      典型值：平移运动可达10-100倍改善                          │\n');
fprintf('│                                                                 │\n');
fprintf('│ 【运动模式说明】                                                │\n');
fprintf('│   静态：阵列不动，作为对照基准                                  │\n');
fprintf('│   x平移：沿x轴直线运动，扩展y方向孔径                          │\n');
fprintf('│   y平移：沿y轴直线运动，扩展x方向孔径                          │\n');
fprintf('│   旋转：原地旋转，不扩展孔径（对照组）                          │\n');
fprintf('│   平移+旋转：复合运动，同时扩展孔径并改变观测角                 │\n');
fprintf('│                                                                 │\n');
fprintf('│ 【阵列形状说明】                                                │\n');
fprintf('│   ULA: 均匀线阵，结构简单，单方向分辨                          │\n');
fprintf('│   URA: 均匀矩形阵，可二维分辨                                  │\n');
fprintf('│   圆阵: 各向同性，适合全向探测                                 │\n');
fprintf('│   L/T/Y/十字: 稀疏阵列，孔径大但阵元少                         │\n');
fprintf('│                                                                 │\n');
fprintf('│ 【结果解读】                                                    │\n');
fprintf('│   - 平移运动显著优于静态和纯旋转                               │\n');
fprintf('│   - 纯旋转不扩展孔径，性能与静态相近                           │\n');
fprintf('│   - 孔径扩展倍数约等于RMSE改善倍数                             │\n');
fprintf('└─────────────────────────────────────────────────────────────────┘\n\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  雷达系统基础参数配置
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('【雷达系统参数】\n');

c = physconst('LightSpeed');

% 载频与波长
fc = 3e9;                           % 载频: 3 GHz (S波段)
lambda = c / fc;                    % 波长: ≈10 cm

% FMCW参数（用于完整仿真模式）
BW = 50e6;                          % 带宽: 50 MHz
T_chirp = 10e-3;                    % Chirp周期: 10 ms
slope = BW / T_chirp;               % 调频斜率: 5e12 Hz/s
range_res = c / (2 * BW);           % 距离分辨率: 3 m

% 采样参数
% 注意：FMCW拍频 = slope × τ = slope × 2R/c
% 对于 R=1000m, 拍频 = 5e12 × 6.67e-6 = 33.3 MHz
% 若要避免混叠，需 fs > 2 × 33.3 MHz = 66.6 MHz
% 当前简化模式不使用FMCW，所以不需要高采样率
fs = 100e6;                         % 采样率: 100 MHz (足以覆盖最大拍频)
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
fprintf('  Chirp周期: %.0f ms\n', T_chirp*1000);
fprintf('  采样率: %.0f MHz (避免混叠需 > %.1f MHz)\n', fs/1e6, 2*slope*2000/c/1e6);

%% ═══════════════════════════════════════════════════════════════════════════
%  阵列参数配置
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n【阵列参数】\n');

d = lambda / 2;                     % 阵元间距: 0.5λ (≈5 cm)，避免栅瓣
fprintf('  阵元间距: %.2f cm (%.2fλ)\n', d*100, d/lambda);

%% ═══════════════════════════════════════════════════════════════════════════
%  实验参数配置
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n【实验参数】\n');

% 目标参数
target_phi = 30;                    % 目标方位角: 30°
target_theta = 90;                  % 目标俯仰角: 90° (水平面)
target_range = 500;                 % 目标距离: 500 m

% 观测参数
T_obs = 0.5;                        % 观测时间: 0.5 s
num_snapshots = 64;                 % 快拍数
t_axis = linspace(0, T_obs, num_snapshots);

% 运动参数
v_linear = 5;                       % 平移速度: 5 m/s
omega_deg = 90;                     % 旋转角度: 90°

% SNR扫描范围
snr_range = -15:5:20;
num_trials = 30;                    % 蒙特卡洛试验次数

fprintf('  目标: φ=%.0f°, θ=%.0f°, R=%.0fm\n', target_phi, target_theta, target_range);
fprintf('  观测时间: %.1fs, 快拍数: %d\n', T_obs, num_snapshots);
fprintf('  运动: v=%.0fm/s, 旋转%.0f°\n', v_linear, omega_deg);
fprintf('  SNR范围: [%d, %d]dB, 试验次数: %d\n', snr_range(1), snr_range(end), num_trials);

%% ═══════════════════════════════════════════════════════════════════════════
%  搜索配置
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n【搜索配置】\n');

% 选择搜索模式: '1d' 或 '2d'
SEARCH_MODE = '1d';                 % ULA使用1D搜索效率更高

% 智能搜索参数（2D模式下使用）
USE_SMART_SEARCH = false;
smart_grid.coarse_res = 5.0;        % 粗搜索: 5°
smart_grid.fine_res = 0.5;          % 细搜索: 0.5°
smart_grid.roi_margin = 10.0;       % ROI边界: ±10°
smart_grid.theta_range = [60, 90];
smart_grid.phi_range = [0, 90];

% CFAR配置
USE_CFAR = false;
cfar_options.numGuard = 2;
cfar_options.numTrain = 4;
cfar_options.P_fa = 1e-4;
cfar_options.min_separation = 3;    % 最小峰值间隔（度）

% 1D搜索网格
phi_search = 0:0.1:90;
search_grid_1d = struct('phi', phi_search);

% 2D搜索网格
search_grid_2d.theta = 60:0.5:90;
search_grid_2d.phi = 0:0.5:90;

fprintf('  搜索模式: %s\n', upper(SEARCH_MODE));
if strcmp(SEARCH_MODE, '1d')
    fprintf('  搜索范围: φ ∈ [%.0f°, %.0f°], 步进 %.1f°\n', ...
        phi_search(1), phi_search(end), phi_search(2)-phi_search(1));
else
    fprintf('  搜索范围: θ ∈ [%.0f°, %.0f°], φ ∈ [%.0f°, %.0f°]\n', ...
        search_grid_2d.theta(1), search_grid_2d.theta(end), ...
        search_grid_2d.phi(1), search_grid_2d.phi(end));
    fprintf('  智能搜索: %s\n', iff(USE_SMART_SEARCH, '启用', '禁用'));
    fprintf('  CA-CFAR: %s\n', iff(USE_CFAR, '启用', '禁用'));
end

%% ═══════════════════════════════════════════════════════════════════════════
%  定义阵列形状
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n【阵列形状定义】\n');

array_configs = struct();

% 1. 8元ULA (线阵)
array_configs(1).name = 'ULA-8';
array_configs(1).num_elements = 8;
array_configs(1).create_func = @() create_ula(8, d);
array_configs(1).description = '8元均匀线阵';

% 2. 3×3 URA (方阵)
array_configs(2).name = 'URA-3x3';
array_configs(2).num_elements = 9;
array_configs(2).create_func = @() create_ura(3, 3, d);
array_configs(2).description = '3×3均匀矩形阵';

% 3. 4×2 URA (矩形阵)
array_configs(3).name = 'URA-4x2';
array_configs(3).num_elements = 8;
array_configs(3).create_func = @() create_ura(4, 2, d);
array_configs(3).description = '4×2均匀矩形阵';

% 4. L形阵列 (4+4元)
array_configs(4).name = 'L阵列';
array_configs(4).num_elements = 7;
array_configs(4).create_func = @() create_L_array(4, 4, d);
array_configs(4).description = 'L形阵列(4+4-1)';

% 5. T形阵列
array_configs(5).name = 'T阵列';
array_configs(5).num_elements = 9;
array_configs(5).create_func = @() create_T_array(5, 5, d);
array_configs(5).description = 'T形阵列(5+5-1)';

% 6. 圆形阵列 (8元)
array_configs(6).name = '圆阵-8';
array_configs(6).num_elements = 8;
array_configs(6).create_func = @() create_circular_array(8, lambda);
array_configs(6).description = '8元圆形阵列';

% 7. 十字形阵列
array_configs(7).name = '十字阵列';
array_configs(7).num_elements = 9;
array_configs(7).create_func = @() create_cross_array(5, d);
array_configs(7).description = '十字形阵列';

% 8. Y形阵列
array_configs(8).name = 'Y阵列';
array_configs(8).num_elements = 10;
array_configs(8).create_func = @() create_Y_array(4, d);
array_configs(8).description = 'Y形阵列(3臂)';

for i = 1:length(array_configs)
    fprintf('  %d. %s: %s (%d阵元)\n', i, array_configs(i).name, ...
        array_configs(i).description, array_configs(i).num_elements);
end

%% ═══════════════════════════════════════════════════════════════════════════
%  定义运动模式
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n【运动模式定义】\n');

motion_configs = struct();

% 1. 静态（基准）
motion_configs(1).name = '静态';
motion_configs(1).trajectory = @(t) struct('position', [0,0,0], 'orientation', [0,0,0]);
motion_configs(1).use_synthetic = false;

% 2. x方向平移
motion_configs(2).name = 'x平移';
motion_configs(2).trajectory = @(t) struct('position', [v_linear*t, 0, 0], 'orientation', [0,0,0]);
motion_configs(2).use_synthetic = true;

% 3. y方向平移
motion_configs(3).name = 'y平移';
motion_configs(3).trajectory = @(t) struct('position', [0, v_linear*t, 0], 'orientation', [0,0,0]);
motion_configs(3).use_synthetic = true;

% 4. 绕中心旋转
motion_configs(4).name = '旋转';
motion_configs(4).trajectory = @(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_deg*t/T_obs]);
motion_configs(4).use_synthetic = true;

% 5. 平移+旋转
motion_configs(5).name = '平移+旋转';
motion_configs(5).trajectory = @(t) struct('position', [0, v_linear*t, 0], 'orientation', [0, 0, omega_deg*t/T_obs]);
motion_configs(5).use_synthetic = true;

for i = 1:length(motion_configs)
    fprintf('  %d. %s\n', i, motion_configs(i).name);
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
results.aperture = zeros(length(array_configs), length(motion_configs));
results.beamwidth = zeros(length(array_configs), length(motion_configs));

% 保存实验配置
results.config.fc = fc;
results.config.lambda = lambda;
results.config.element_spacing = d;
results.config.target_phi = target_phi;
results.config.target_theta = target_theta;
results.config.target_range = target_range;
results.config.T_obs = T_obs;
results.config.num_snapshots = num_snapshots;
results.config.v_linear = v_linear;
results.config.omega_deg = omega_deg;
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
        
        % 计算合成孔径
        [aperture_x, aperture_y, ~] = calc_synthetic_aperture(array, t_axis, lambda);
        total_aperture = sqrt(aperture_x^2 + aperture_y^2);
        results.aperture(arr_idx, mot_idx) = total_aperture / lambda;
        
        % 计算主瓣宽度（高SNR下）
        target_pos = target_range * [cosd(target_phi)*sind(target_theta), ...
                                     sind(target_phi)*sind(target_theta), ...
                                     cosd(target_theta)];
        target = Target(target_pos, [0,0,0], 1);
        sig_gen = SignalGeneratorSimple(radar_params, array, {target});
        rng(0);
        snapshots_test = sig_gen.generate_snapshots(t_axis, 20);
        
        % 选择搜索网格
        if strcmp(SEARCH_MODE, '1d')
            search_grid = search_grid_1d;
        else
            if USE_SMART_SEARCH
                search_grid = smart_grid;
            else
                search_grid = search_grid_2d;
            end
        end
        
        % DOA估计
        est_options.search_mode = SEARCH_MODE;
        est_options.use_smart_search = USE_SMART_SEARCH;
        est_options.use_cfar = USE_CFAR;
        est_options.cfar_options = cfar_options;
        
        if mot_cfg.use_synthetic
            estimator = DoaEstimatorSynthetic(array, radar_params);
            [spectrum, ~, ~] = estimator.estimate(snapshots_test, t_axis, search_grid, 1, est_options);
        else
            positions = array.get_mimo_virtual_positions(0);
            if strcmp(SEARCH_MODE, '1d')
                spectrum = music_standard_1d(snapshots_test, positions, phi_search, lambda, 1);
            else
                spectrum = music_standard_2d(snapshots_test, positions, search_grid_2d, lambda, 1);
            end
        end
        
        if strcmp(SEARCH_MODE, '1d')
            results.beamwidth(arr_idx, mot_idx) = calc_beamwidth(spectrum, phi_search);
        else
            results.beamwidth(arr_idx, mot_idx) = calc_beamwidth_2d(spectrum, search_grid);
        end
        
        % SNR扫描
        for snr_idx = 1:length(snr_range)
            snr_test = snr_range(snr_idx);
            errors = zeros(1, num_trials);
            
            for trial = 1:num_trials
                rng(trial * 1000 + snr_idx + arr_idx * 100 + mot_idx * 10);
                target_phi_trial = target_phi + (rand() - 0.5) * 1;
                
                target_pos = target_range * [cosd(target_phi_trial)*sind(target_theta), ...
                                             sind(target_phi_trial)*sind(target_theta), ...
                                             cosd(target_theta)];
                target = Target(target_pos, [0,0,0], 1);
                sig_gen = SignalGeneratorSimple(radar_params, array, {target});
                snapshots = sig_gen.generate_snapshots(t_axis, snr_test);
                
                if mot_cfg.use_synthetic
                    estimator = DoaEstimatorSynthetic(array, radar_params);
                    [~, peaks, ~] = estimator.estimate(snapshots, t_axis, search_grid, 1, est_options);
                    est_phi = peaks.phi(1);
                else
                    positions = array.get_mimo_virtual_positions(0);
                    if strcmp(SEARCH_MODE, '1d')
                        spectrum = music_standard_1d(snapshots, positions, phi_search, lambda, 1);
                        [~, peak_idx] = max(spectrum);
                        est_phi = phi_search(peak_idx);
                    else
                        spectrum = music_standard_2d(snapshots, positions, search_grid_2d, lambda, 1);
                        [~, idx] = max(spectrum(:));
                        [~, phi_idx] = ind2sub(size(spectrum), idx);
                        est_phi = search_grid_2d.phi(phi_idx);
                    end
                end
                
                errors(trial) = est_phi - target_phi_trial;
            end
            
            results.rmse(arr_idx, mot_idx, snr_idx) = sqrt(mean(errors.^2));
        end
        
        fprintf('孔径=%.1fλ, 主瓣=%.1f°\n', results.aperture(arr_idx, mot_idx), results.beamwidth(arr_idx, mot_idx));
    end
end

%% ═══════════════════════════════════════════════════════════════════════════
%  输出结果
%% ═══════════════════════════════════════════════════════════════════════════
fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('                        实验结果                                   \n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

for arr_idx = 1:length(array_configs)
    fprintf('【%s】\n', array_configs(arr_idx).name);
    fprintf('运动模式      | 孔径    | 主瓣   |');
    for snr_idx = 1:length(snr_range)
        fprintf(' %3ddB |', snr_range(snr_idx));
    end
    fprintf('\n');
    fprintf('--------------|---------|--------|');
    for snr_idx = 1:length(snr_range)
        fprintf('-------|');
    end
    fprintf('\n');
    
    for mot_idx = 1:length(motion_configs)
        fprintf('%-13s | %5.1fλ  | %4.1f°  |', ...
            motion_configs(mot_idx).name, ...
            results.aperture(arr_idx, mot_idx), ...
            results.beamwidth(arr_idx, mot_idx));
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
%  绘图 - 论文风格（中文）
%% ═══════════════════════════════════════════════════════════════════════════
% output_folder已在脚本开头定义

% 设置论文风格（支持中文）
set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultLineLineWidth', 1.5);
set(0, 'DefaultTextInterpreter', 'none');

% 配色方案
color_static = [0.2, 0.2, 0.2];
color_x_trans = [0.85, 0.33, 0.1];
color_y_trans = [0.0, 0.45, 0.74];
color_rotation = [0.47, 0.67, 0.19];
color_combined = [0.49, 0.18, 0.56];
motion_colors = [color_static; color_x_trans; color_y_trans; color_rotation; color_combined];

%% 图1: 静态vs运动阵列RMSE对比 (核心结果)
figure('Position', [50, 50, 1200, 500], 'Color', 'white');

% 选择3个代表性阵列
selected_arrays = [1, 2, 6];  % ULA, URA, 圆阵
arr_labels = {'ULA-8', 'URA-3×3', '圆阵-8'};
markers = {'o', 's', 'd'};

subplot(1, 2, 1);
hold on;
for i = 1:length(selected_arrays)
    arr_idx = selected_arrays(i);
    rmse_static = squeeze(results.rmse(arr_idx, 1, :));  % 静态
    plot(snr_range, rmse_static, ['-' markers{i}], 'Color', color_static, ...
        'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', [arr_labels{i} ' (静态)']);
end
set(gca, 'YScale', 'log');
xlabel('信噪比 (dB)', 'FontWeight', 'bold');
ylabel('RMSE (°)', 'FontWeight', 'bold');
title('(a) 静态阵列', 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 9);
grid on;
ylim([0.01, 100]);
xlim([snr_range(1)-1, snr_range(end)+1]);

subplot(1, 2, 2);
hold on;
for i = 1:length(selected_arrays)
    arr_idx = selected_arrays(i);
    rmse_motion = squeeze(results.rmse(arr_idx, 3, :));  % y平移
    plot(snr_range, rmse_motion, ['-' markers{i}], 'Color', color_y_trans, ...
        'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', [arr_labels{i} ' (运动)']);
end
set(gca, 'YScale', 'log');
xlabel('信噪比 (dB)', 'FontWeight', 'bold');
ylabel('RMSE (°)', 'FontWeight', 'bold');
title('(b) 运动阵列 (Y平移)', 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 9);
grid on;
ylim([0.01, 100]);
xlim([snr_range(1)-1, snr_range(end)+1]);

sgtitle('DOA估计性能: 静态阵列 vs 运动阵列', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(output_folder, 'fig1_静态vs运动阵列.png'));
saveas(gcf, fullfile(output_folder, 'fig1_静态vs运动阵列.eps'), 'epsc');
fprintf('图片已保存: fig1_静态vs运动阵列.png\n');

%% 图2: 所有运动模式对比（ULA示例）
figure('Position', [50, 50, 700, 550], 'Color', 'white');
arr_idx = 1;  % ULA
hold on;
motion_markers = {'o', 's', 'd', '^', 'v'};
for mot_idx = 1:length(motion_configs)
    rmse_curve = squeeze(results.rmse(arr_idx, mot_idx, :));
    plot(snr_range, rmse_curve, ['-' motion_markers{mot_idx}], ...
        'Color', motion_colors(mot_idx, :), ...
        'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', motion_colors(mot_idx, :), ...
        'DisplayName', motion_configs(mot_idx).name);
end
set(gca, 'YScale', 'log');
xlabel('信噪比 (dB)', 'FontWeight', 'bold');
ylabel('RMSE (°)', 'FontWeight', 'bold');
title('8元ULA: 不同运动模式RMSE对比', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 10);
grid on;
ylim([0.01, 100]);
xlim([snr_range(1)-1, snr_range(end)+1]);

% 添加注释
low_snr_idx = find(snr_range == -10, 1);
if isempty(low_snr_idx), low_snr_idx = 2; end
improvement = results.rmse(arr_idx, 1, low_snr_idx) / results.rmse(arr_idx, 3, low_snr_idx);
if improvement > 1
    annotation('textbox', [0.15, 0.17, 0.25, 0.1], ...
        'String', sprintf('Y平移改善倍数\nSNR=%ddB: %.1f倍', snr_range(low_snr_idx), improvement), ...
        'FontSize', 9, 'BackgroundColor', [1 1 0.9], 'EdgeColor', [0.5 0.5 0.5]);
end

saveas(gcf, fullfile(output_folder, 'fig2_运动模式对比.png'));
saveas(gcf, fullfile(output_folder, 'fig2_运动模式对比.eps'), 'epsc');
fprintf('图片已保存: fig2_运动模式对比.png\n');

%% 图3: 孔径扩展对比柱状图
figure('Position', [50, 50, 900, 500], 'Color', 'white');
aperture_data = results.aperture';  % [运动模式 × 阵列]
b = bar(aperture_data', 'grouped');
for i = 1:5
    b(i).FaceColor = motion_colors(i, :);
end
set(gca, 'XTick', 1:length(array_configs), 'XTickLabel', {array_configs.name});
xtickangle(30);
xlabel('阵列配置', 'FontWeight', 'bold');
ylabel('合成孔径 (λ)', 'FontWeight', 'bold');
title('不同运动模式下的孔径扩展', 'FontSize', 13, 'FontWeight', 'bold');
legend({motion_configs.name}, 'Location', 'northwest', 'FontSize', 9);
grid on;
ylim([0, 35]);

saveas(gcf, fullfile(output_folder, 'fig3_孔径扩展对比.png'));
saveas(gcf, fullfile(output_folder, 'fig3_孔径扩展对比.eps'), 'epsc');
fprintf('图片已保存: fig3_孔径扩展对比.png\n');

%% 图4: 低SNR改善倍数条形图
figure('Position', [50, 50, 900, 450], 'Color', 'white');
low_snr_idx = find(snr_range <= -10, 1, 'last');
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
for i = 1:4
    b(i).FaceColor = motion_colors(i+1, :);
end
set(gca, 'XTick', 1:length(array_configs), 'XTickLabel', {array_configs.name});
xtickangle(30);
xlabel('阵列配置', 'FontWeight', 'bold');
ylabel('改善倍数 (×)', 'FontWeight', 'bold');
title(sprintf('SNR = %d dB 时相对静态阵列的RMSE改善', snr_range(low_snr_idx)), ...
    'FontSize', 13, 'FontWeight', 'bold');
legend({motion_configs(2:end).name}, 'Location', 'northwest', 'FontSize', 9);
grid on;

saveas(gcf, fullfile(output_folder, 'fig4_改善倍数.png'));
saveas(gcf, fullfile(output_folder, 'fig4_改善倍数.eps'), 'epsc');
fprintf('图片已保存: fig4_改善倍数.png\n');

%% 图5: 阵列形状可视化（简化版）
figure('Position', [50, 100, 1000, 400], 'Color', 'white');
selected_for_display = [1, 3, 4, 6];  % ULA, URA, L, 圆阵
display_names = {'ULA-8', 'URA-4×2', 'L阵列', '圆阵-8'};

for i = 1:4
    subplot(1, 4, i);
    arr_idx = selected_for_display(i);
    array = array_configs(arr_idx).create_func();
    pos = array.get_mimo_virtual_positions(0);
    scatter(pos(:,1)*1000, pos(:,2)*1000, 120, 'filled', ...
        'MarkerFaceColor', [0.2, 0.4, 0.8], ...
        'MarkerEdgeColor', [0.1, 0.2, 0.5], 'LineWidth', 1.5);
    xlabel('x (mm)', 'FontWeight', 'bold');
    ylabel('y (mm)', 'FontWeight', 'bold');
    title(display_names{i}, 'FontWeight', 'bold', 'FontSize', 12);
    axis equal;
    grid on;
    ax = gca;
    max_range = max(abs([ax.XLim, ax.YLim])) * 1.3;
    if max_range < 1, max_range = 100; end
    xlim([-max_range, max_range]);
    ylim([-max_range, max_range]);
end
sgtitle('阵列几何配置', 'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, fullfile(output_folder, 'fig5_阵列形状.png'));
saveas(gcf, fullfile(output_folder, 'fig5_阵列形状.eps'), 'epsc');
fprintf('图片已保存: fig5_阵列形状.png\n');

%% 图6: 虚拟阵列轨迹对比
figure('Position', [50, 50, 1100, 350], 'Color', 'white');

% 重新定义绘图所需参数
v_demo = 5;  % m/s
T_obs_demo = 0.5;  % s
num_snaps_demo = 64;
t_demo = linspace(0, T_obs_demo, num_snaps_demo);

% 创建ULA用于演示
arr_demo = create_ula(8, d);

% 静态
subplot(1, 3, 1);
arr_demo.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));
pos_static = arr_demo.get_mimo_virtual_positions(0);
scatter(pos_static(:,1)/lambda, pos_static(:,2)/lambda, 80, 'filled', 'MarkerFaceColor', color_static);
xlabel('x (λ)'); ylabel('y (λ)');
title('(a) 静态', 'FontWeight', 'bold');
axis equal; grid on;
xlim([-5, 5]); ylim([-2, 28]);
text(0, -1.5, sprintf('孔径: %.1fλ', (max(pos_static(:,1))-min(pos_static(:,1)))/lambda), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% Y平移
subplot(1, 3, 2);
all_pos = [];
for k = 1:num_snaps_demo
    y_offset = v_demo * t_demo(k);
    arr_demo.set_trajectory(@(t) struct('position', [0, y_offset, 0], 'orientation', [0,0,0]));
    pos_k = arr_demo.get_mimo_virtual_positions(0);
    all_pos = [all_pos; pos_k];
end
scatter(all_pos(:,1)/lambda, all_pos(:,2)/lambda, 8, 'filled', ...
    'MarkerFaceColor', color_y_trans, 'MarkerFaceAlpha', 0.4);
hold on;
quiver(0, 0, 0, v_demo*T_obs_demo/lambda*0.9, 0, 'r', 'LineWidth', 2, 'MaxHeadSize', 1);
hold off;
xlabel('x (λ)'); ylabel('y (λ)');
title('(b) Y平移', 'FontWeight', 'bold');
axis equal; grid on;
xlim([-5, 5]); ylim([-2, 28]);
aperture_y = (max(all_pos(:,2))-min(all_pos(:,2)))/lambda;
text(0, -1.5, sprintf('孔径: %.1fλ', aperture_y), 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% 旋转
subplot(1, 3, 3);
all_pos = [];
for k = 1:num_snaps_demo
    rot_angle = 90 * t_demo(k) / T_obs_demo;
    arr_demo.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0, 0, rot_angle]));
    pos_k = arr_demo.get_mimo_virtual_positions(0);
    all_pos = [all_pos; pos_k];
end
scatter(all_pos(:,1)/lambda, all_pos(:,2)/lambda, 8, 'filled', ...
    'MarkerFaceColor', color_rotation, 'MarkerFaceAlpha', 0.4);
xlabel('x (λ)'); ylabel('y (λ)');
title('(c) 旋转 (90°)', 'FontWeight', 'bold');
axis equal; grid on;
xlim([-5, 5]); ylim([-5, 5]);
aperture_rot = sqrt((max(all_pos(:,1))-min(all_pos(:,1)))^2 + (max(all_pos(:,2))-min(all_pos(:,2)))^2)/lambda;
text(0, -4, sprintf('孔径: %.1fλ', aperture_rot), 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

sgtitle('不同运动模式下的虚拟阵列位置', 'FontSize', 13, 'FontWeight', 'bold');
saveas(gcf, fullfile(output_folder, 'fig6_虚拟阵列轨迹.png'));
saveas(gcf, fullfile(output_folder, 'fig6_虚拟阵列轨迹.eps'), 'epsc');
fprintf('图片已保存: fig6_虚拟阵列轨迹.png\n');

%% 保存数据
save(fullfile(output_folder, 'experiment_results.mat'), 'results', 'array_configs', 'motion_configs', 'radar_params');
fprintf('数据已保存: experiment_results.mat\n');

fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('实验完成！\n');
fprintf('所有结果保存在: %s\n', output_folder);
fprintf('═══════════════════════════════════════════════════════════════════\n');

% 关闭日志
diary off;

%% ═══════════════════════════════════════════════════════════════════════════
%  辅助函数
%% ═══════════════════════════════════════════════════════════════════════════

function out = iff(cond, true_val, false_val)
    if cond
        out = true_val;
    else
        out = false_val;
    end
end

function array = create_ula(num_elements, spacing)
    x_pos = ((0:num_elements-1) - (num_elements-1)/2) * spacing;
    y_pos = zeros(1, num_elements);
    z_pos = zeros(1, num_elements);
    elements = [x_pos', y_pos', z_pos'];
    tx = 1;
    rx = 1:num_elements;
    array = ArrayPlatform(elements, tx, rx);
end

function array = create_ura(num_x, num_y, spacing)
    elements = [];
    for iy = 1:num_y
        for ix = 1:num_x
            x = (ix - 1 - (num_x-1)/2) * spacing;
            y = (iy - 1 - (num_y-1)/2) * spacing;
            elements = [elements; x, y, 0];
        end
    end
    tx = 1;
    rx = 1:size(elements, 1);
    array = ArrayPlatform(elements, tx, rx);
end

function array = create_L_array(num_x, num_y, spacing)
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

function array = create_T_array(num_horizontal, num_vertical, spacing)
    elements = [];
    for i = 1:num_horizontal
        x = (i - 1 - (num_horizontal-1)/2) * spacing;
        elements = [elements; x, 0, 0];
    end
    for i = 2:num_vertical
        y = -(i - 1) * spacing;
        elements = [elements; 0, y, 0];
    end
    tx = 1;
    rx = 1:size(elements, 1);
    array = ArrayPlatform(elements, tx, rx);
end

function array = create_circular_array(num_elements, radius)
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

function array = create_cross_array(arm_length, spacing)
    elements = [0, 0, 0];
    for i = 1:(arm_length-1)/2
        elements = [elements; i*spacing, 0, 0];
        elements = [elements; -i*spacing, 0, 0];
        elements = [elements; 0, i*spacing, 0];
        elements = [elements; 0, -i*spacing, 0];
    end
    tx = 1;
    rx = 1:size(elements, 1);
    array = ArrayPlatform(elements, tx, rx);
end

function array = create_Y_array(arm_length, spacing)
    elements = [0, 0, 0];
    angles = [90, 210, 330];
    for ang = angles
        for i = 1:arm_length-1
            x = i * spacing * cosd(ang);
            y = i * spacing * sind(ang);
            elements = [elements; x, y, 0];
        end
    end
    tx = 1;
    rx = 1:size(elements, 1);
    array = ArrayPlatform(elements, tx, rx);
end

function [aperture_x, aperture_y, all_positions] = calc_synthetic_aperture(array, t_axis, lambda)
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

function spectrum = music_standard_1d(snapshots, positions, phi_search, lambda, num_targets)
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
                a(i) = exp(1j * phase);
            end
            
            spectrum(theta_idx, phi_idx) = 1 / abs(a' * (Qn * Qn') * a);
        end
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
        beamwidth = 0.5;
    end
end

function beamwidth = calc_beamwidth_2d(spectrum, search_grid)
    [~, idx] = max(spectrum(:));
    [theta_idx, phi_idx] = ind2sub(size(spectrum), idx);
    
    phi_slice = spectrum(theta_idx, :);
    beamwidth = calc_beamwidth(phi_slice, search_grid.phi);
end
