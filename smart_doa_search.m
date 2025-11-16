function [spectrum_full, search_grid_fine] = smart_doa_search(estimator, snapshots, t_axis, num_targets, search_grid_coarse, options)
% SMART_DOA_SEARCH 两步智能DOA搜索：粗搜索定位 + 细搜索精化
%
% 输入:
%   estimator         - DoaEstimator 或 DoaEstimatorIncoherent 对象
%   snapshots         - 快拍数据
%   t_axis            - 时间轴
%   num_targets       - 目标数量
%   search_grid_coarse - 粗搜索网格结构体 (coarse_res, fine_res, roi_margin)
%   options           - 可选参数
%
% 输出:
%   spectrum_full     - 完整的高分辨率谱
%   search_grid_fine  - 细网格搜索空间
%
% 策略:
%   1. 粗搜索 (5°网格): 快速定位峰值
%   2. 细搜索 (0.2°网格): 在峰值±margin范围内精确扫描
%   3. 合并: 将细搜索结果嵌入粗搜索，插值填充

    if nargin < 6
        options.verbose = true;
    end
    
    verbose = options.verbose;
    
    % 提取参数
    coarse_res = search_grid_coarse.coarse_res;      % 粗搜索分辨率 (如 5°)
    fine_res = search_grid_coarse.fine_res;          % 细搜索分辨率 (如 0.2°)
    roi_margin = search_grid_coarse.roi_margin;      % ROI边界扩展 (如 10°)
    theta_range = search_grid_coarse.theta_range;    % [theta_min, theta_max]
    phi_range = search_grid_coarse.phi_range;        % [phi_min, phi_max]
    
    %% ===== 第一步：粗搜索 =====
    if verbose
        fprintf('    🔍 第1步：粗搜索 (%.1f°网格) ... ', coarse_res);
        tic;
    end
    
    % 构建粗网格
    theta_coarse = theta_range(1):coarse_res:theta_range(2);
    phi_coarse = phi_range(1):coarse_res:phi_range(2);
    grid_coarse.theta = theta_coarse;
    grid_coarse.phi = phi_coarse;
    
    num_points_coarse = length(theta_coarse) * length(phi_coarse);
    
    % 执行粗搜索
    if isa(estimator, 'DoaEstimatorIncoherent')
        opts_coarse = struct('verbose', false);
        if isfield(options, 'weighting')
            opts_coarse.weighting = options.weighting;
        end
        spectrum_coarse = estimator.estimate_incoherent_music(snapshots, t_axis, num_targets, grid_coarse, opts_coarse);
    else
        spectrum_coarse = estimator.estimate_gmusic(snapshots, t_axis, num_targets, grid_coarse);
    end
    
    if verbose
        elapsed_coarse = toc;
        fprintf('完成 (%d点, %.1f秒)\n', num_points_coarse, elapsed_coarse);
    end
    
    %% ===== 第二步：定位峰值 =====
    if verbose
        fprintf('    🎯 第2步：定位峰值 ... ');
    end
    
    % 找到所有峰值位置
    if isa(estimator, 'DoaEstimatorIncoherent')
        [theta_peaks, phi_peaks, ~] = DoaEstimatorIncoherent.find_peaks(spectrum_coarse, grid_coarse, num_targets);
    else
        [theta_peaks, phi_peaks, ~] = DoaEstimator.find_peaks(spectrum_coarse, grid_coarse, num_targets);
    end
    
    if verbose
        fprintf('找到 %d 个峰值\n', length(theta_peaks));
        for i = 1:length(theta_peaks)
            fprintf('       峰值%d: theta=%.1f°, phi=%.1f°\n', i, theta_peaks(i), phi_peaks(i));
        end
    end
    
    %% ===== 第三步：细搜索（每个峰值附近）=====
    if verbose
        fprintf('    🔬 第3步：细搜索 (%.1f°网格，±%.1f°范围) ...\n', fine_res, roi_margin);
    end
    
    % 准备细搜索的ROI列表
    roi_list = [];
    for i = 1:length(theta_peaks)
        roi.theta_center = theta_peaks(i);
        roi.phi_center = phi_peaks(i);
        roi.theta_min = max(theta_range(1), theta_peaks(i) - roi_margin);
        roi.theta_max = min(theta_range(2), theta_peaks(i) + roi_margin);
        roi.phi_min = max(phi_range(1), phi_peaks(i) - roi_margin);
        roi.phi_max = min(phi_range(2), phi_peaks(i) + roi_margin);
        roi_list = [roi_list; roi];
    end
    
    % 对每个ROI进行细搜索
    fine_regions = {};
    for i = 1:length(roi_list)
        if verbose
            fprintf('       ROI%d: theta[%.1f, %.1f], phi[%.1f, %.1f] ... ', ...
                i, roi_list(i).theta_min, roi_list(i).theta_max, ...
                roi_list(i).phi_min, roi_list(i).phi_max);
            tic;
        end
        
        % 构建细网格
        theta_fine_roi = roi_list(i).theta_min:fine_res:roi_list(i).theta_max;
        phi_fine_roi = roi_list(i).phi_min:fine_res:roi_list(i).phi_max;
        grid_fine_roi.theta = theta_fine_roi;
        grid_fine_roi.phi = phi_fine_roi;
        
        num_points_roi = length(theta_fine_roi) * length(phi_fine_roi);
        
        % 执行细搜索
        if isa(estimator, 'DoaEstimatorIncoherent')
            opts_fine = struct('verbose', false);
            if isfield(options, 'weighting')
                opts_fine.weighting = options.weighting;
            end
            spectrum_fine_roi = estimator.estimate_incoherent_music(snapshots, t_axis, num_targets, grid_fine_roi, opts_fine);
        else
            spectrum_fine_roi = estimator.estimate_gmusic(snapshots, t_axis, num_targets, grid_fine_roi);
        end
        
        % 保存结果
        fine_regions{i}.theta = theta_fine_roi;
        fine_regions{i}.phi = phi_fine_roi;
        fine_regions{i}.spectrum = spectrum_fine_roi;
        fine_regions{i}.roi = roi_list(i);
        
        if verbose
            elapsed_roi = toc;
            fprintf('%d点, %.1f秒\n', num_points_roi, elapsed_roi);
        end
    end
    
    %% ===== 第四步：合并谱 =====
    if verbose
        fprintf('    🔗 第4步：合并谱 ... ');
        tic;
    end
    
    % 创建最终的细网格
    theta_fine = theta_range(1):fine_res:theta_range(2);
    phi_fine = phi_range(1):fine_res:phi_range(2);
    [Theta_fine, Phi_fine] = meshgrid(phi_fine, theta_fine);
    
    % 从粗网格插值到细网格（作为背景）
    [Theta_coarse, Phi_coarse] = meshgrid(phi_coarse, theta_coarse);
    spectrum_full = interp2(Theta_coarse, Phi_coarse, spectrum_coarse, Theta_fine, Phi_fine, 'linear');
    
    % 用细搜索结果覆盖对应区域
    for i = 1:length(fine_regions)
        theta_roi = fine_regions{i}.theta;
        phi_roi = fine_regions{i}.phi;
        spectrum_roi = fine_regions{i}.spectrum;
        
        % 找到在全局网格中的索引
        [~, theta_idx_start] = min(abs(theta_fine - theta_roi(1)));
        [~, theta_idx_end] = min(abs(theta_fine - theta_roi(end)));
        [~, phi_idx_start] = min(abs(phi_fine - phi_roi(1)));
        [~, phi_idx_end] = min(abs(phi_fine - phi_roi(end)));
        
        % 覆盖
        spectrum_full(theta_idx_start:theta_idx_end, phi_idx_start:phi_idx_end) = spectrum_roi;
    end
    
    % 填充NaN（边界外的点）
    spectrum_full(isnan(spectrum_full)) = 0;
    
    search_grid_fine.theta = theta_fine;
    search_grid_fine.phi = phi_fine;
    
    if verbose
        elapsed_merge = toc;
        fprintf('完成 (%.1f秒)\n', elapsed_merge);
        
        % 统计加速效果
        total_fine_points = length(theta_fine) * length(phi_fine);
        actual_computed = num_points_coarse + sum(cellfun(@(x) numel(x.spectrum), fine_regions));
        speedup = total_fine_points / actual_computed;
        
        fprintf('    ⚡ 加速效果: 实际计算 %d / 全细搜索 %d = %.1fx 加速\n', ...
            actual_computed, total_fine_points, speedup);
    end
end

