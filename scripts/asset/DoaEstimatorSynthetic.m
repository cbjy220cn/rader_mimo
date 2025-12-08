classdef DoaEstimatorSynthetic
    % DoaEstimatorSynthetic: 合成虚拟阵列DOA估计器
    %
    % 核心思想：将运动阵列的时间维度展开为空间维度
    % - 每个时刻的每个物理阵元作为一个"虚拟阵元"
    % - M个物理阵元 × K个时刻 = M×K个虚拟阵元
    % - 利用运动产生的孔径扩展提升角度分辨率
    %
    % 功能特性：
    % - 支持1D搜索（仅phi）和2D搜索（theta+phi）
    % - 支持多层智能搜索（粗搜索→细搜索）
    % - 支持CA-CFAR多目标检测
    %
    % 用法示例：
    %   estimator = DoaEstimatorSynthetic(array_platform, radar_params);
    %   [spectrum, peaks, info] = estimator.estimate(snapshots, t_axis, search_grid, num_targets);
    %
    %   % 使用2D智能搜索
    %   options.search_mode = '2d';
    %   options.use_smart_search = true;
    %   options.use_cfar = true;
    %   [spectrum, peaks, info] = estimator.estimate(snapshots, t_axis, search_grid, num_targets, options);
    
    properties
        array_platform  % ArrayPlatform实例
        radar_params    % 雷达参数结构体
        lambda          % 波长
        use_gpu         % 是否使用GPU加速
        
        % 配置选项
        max_virtual_elements  % 最大虚拟阵元数（控制计算量）
        subsample_method      % 子采样方法: 'uniform', 'random', 'none'
    end
    
    methods
        function obj = DoaEstimatorSynthetic(array_platform, radar_params, options)
            % 构造函数
            %
            % 输入:
            %   array_platform - ArrayPlatform实例
            %   radar_params   - 雷达参数结构体（必须包含 .fc 或 .lambda）
            %   options        - (可选) 配置选项
            %     .max_virtual_elements - 最大虚拟阵元数 (默认512)
            %     .subsample_method     - 子采样方法 (默认'uniform')
            
            if nargin > 0
                obj.array_platform = array_platform;
                obj.radar_params = radar_params;
                
                % 计算波长
                if isfield(radar_params, 'lambda')
                    obj.lambda = radar_params.lambda;
                else
                    c = physconst('LightSpeed');
                    obj.lambda = c / radar_params.fc;
                end
                
                % GPU检测
                obj.use_gpu = (gpuDeviceCount > 0);
                
                % 默认配置
                obj.max_virtual_elements = 512;
                obj.subsample_method = 'uniform';
                
                % 用户自定义配置
                if nargin > 2 && ~isempty(options)
                    if isfield(options, 'max_virtual_elements')
                        obj.max_virtual_elements = options.max_virtual_elements;
                    end
                    if isfield(options, 'subsample_method')
                        obj.subsample_method = options.subsample_method;
                    end
                end
            end
        end
        
        function [spectrum, peaks, info] = estimate(obj, snapshots, t_axis, search_grid, num_targets, options)
            % 执行DOA估计（主入口）
            %
            % 输入:
            %   snapshots    - [M × K] 快拍矩阵 (M=阵元数, K=快拍数)
            %   t_axis       - [1 × K] 时间轴
            %   search_grid  - 搜索网格
            %                  1D: struct with .phi (方位角数组)
            %                  2D: struct with .theta 和 .phi
            %                  智能搜索: struct with .coarse_res, .fine_res, .roi_margin, .theta_range, .phi_range
            %   num_targets  - 目标数量
            %   options      - (可选) 搜索选项
            %     .search_mode      - '1d' 或 '2d' (默认自动检测)
            %     .use_smart_search - 是否使用多层搜索 (默认false)
            %     .use_cfar         - 是否使用CFAR检测 (默认false)
            %     .cfar_options     - CFAR参数
            %     .verbose          - 是否显示进度 (默认false)
            %
            % 输出:
            %   spectrum - MUSIC谱
            %   peaks    - 峰值位置结构体 (.phi, .theta, .vals)
            %   info     - 附加信息
            
            if nargin < 6
                options = struct();
            end
            
            % 默认选项
            use_smart_search = get_opt(options, 'use_smart_search', false);
            use_cfar = get_opt(options, 'use_cfar', false);
            verbose = get_opt(options, 'verbose', false);
            search_mode = get_opt(options, 'search_mode', 'auto');
            
            % 自动检测搜索模式
            if strcmp(search_mode, 'auto')
                if isfield(search_grid, 'coarse_res')
                    % 智能搜索模式
                    use_smart_search = true;
                    search_mode = '2d';
                elseif isfield(search_grid, 'theta') && ~isempty(search_grid.theta)
                    search_mode = '2d';
                else
                    search_mode = '1d';
                end
            end
            
            % 1. 构建虚拟阵列
            [virtual_positions, virtual_signals, selected_indices] = ...
                obj.build_virtual_array(snapshots, t_axis);
            
            num_virtual = size(virtual_positions, 1);
            
            % 2. 计算合成孔径
            aperture = obj.calc_aperture(virtual_positions);
            
            % 3. 构建协方差矩阵并特征分解
            Rxx = virtual_signals * virtual_signals';
            [V, D] = eig(Rxx);
            [eigenvalues, idx] = sort(diag(D), 'descend');
            V = V(:, idx);
            
            % 确保噪声子空间维度正确
            noise_dim = num_virtual - num_targets;
            if noise_dim < 1
                warning('虚拟阵元数(%d)不足以分辨%d个目标', num_virtual, num_targets);
                noise_dim = 1;
            end
            Qn = V(:, (num_targets+1):end);
            
            % 4. 执行搜索
            if use_smart_search
                % 多层智能搜索
                [spectrum, search_grid_out] = obj.smart_search(virtual_positions, Qn, search_grid, num_targets, options);
                search_grid = search_grid_out;
            else
                % 常规搜索
                if strcmp(search_mode, '1d')
                    [spectrum, ~] = obj.search_1d(virtual_positions, Qn, search_grid.phi, num_targets);
                else
                    [spectrum, ~] = obj.search_2d(virtual_positions, Qn, search_grid, num_targets);
                end
            end
            
            % 5. 峰值检测
            if use_cfar && strcmp(search_mode, '2d')
                cfar_options = get_opt(options, 'cfar_options', struct());
                [theta_peaks, phi_peaks, peak_vals, ~] = find_peaks_cfar(spectrum, search_grid, num_targets, cfar_options);
                peaks.theta = theta_peaks;
                peaks.phi = phi_peaks;
                peaks.vals = peak_vals;
            else
                if strcmp(search_mode, '1d')
                    [~, peak_indices] = maxk(spectrum, num_targets);
                    peaks.phi = search_grid.phi(peak_indices);
                    peaks.theta = 90 * ones(size(peaks.phi));
                    peaks.vals = spectrum(peak_indices);
                else
                    [peaks.theta, peaks.phi, peaks.vals] = obj.find_peaks_2d(spectrum, search_grid, num_targets);
                end
            end
            
            % 6. 输出附加信息
            info = struct();
            info.virtual_positions = virtual_positions;
            info.synthetic_aperture = aperture;
            info.num_virtual = num_virtual;
            info.selected_indices = selected_indices;
            info.eigenvalues = eigenvalues;
            info.search_mode = search_mode;
            info.search_grid = search_grid;
        end
        
        function [spectrum, search_grid_fine] = smart_search(obj, virtual_positions, Qn, smart_grid, num_targets, options)
            % 多层智能搜索：粗搜索定位 + 细搜索精化
            %
            % 输入:
            %   smart_grid - 智能搜索参数
            %     .coarse_res  - 粗搜索分辨率（度）
            %     .fine_res    - 细搜索分辨率（度）
            %     .roi_margin  - ROI边界扩展（度）
            %     .theta_range - [theta_min, theta_max]
            %     .phi_range   - [phi_min, phi_max]
            
            verbose = get_opt(options, 'verbose', false);
            
            coarse_res = smart_grid.coarse_res;
            fine_res = smart_grid.fine_res;
            roi_margin = smart_grid.roi_margin;
            theta_range = smart_grid.theta_range;
            phi_range = smart_grid.phi_range;
            
            %% 第一步：粗搜索
            if verbose
                fprintf('    🔍 粗搜索 (%.1f°网格) ... ', coarse_res);
                tic;
            end
            
            theta_coarse = theta_range(1):coarse_res:theta_range(2);
            phi_coarse = phi_range(1):coarse_res:phi_range(2);
            grid_coarse.theta = theta_coarse;
            grid_coarse.phi = phi_coarse;
            
            [spectrum_coarse, ~] = obj.search_2d(virtual_positions, Qn, grid_coarse, num_targets);
            
            if verbose
                fprintf('完成 (%.2fs)\n', toc);
            end
            
            %% 第二步：找峰值
            [theta_peaks, phi_peaks, ~] = obj.find_peaks_2d(spectrum_coarse, grid_coarse, num_targets);
            
            if verbose
                fprintf('    🎯 找到 %d 个峰值\n', length(theta_peaks));
            end
            
            %% 第三步：细搜索（每个峰值附近）
            if verbose
                fprintf('    🔬 细搜索 (%.1f°网格) ... ', fine_res);
                tic;
            end
            
            fine_regions = {};
            for i = 1:length(theta_peaks)
                theta_min = max(theta_range(1), theta_peaks(i) - roi_margin);
                theta_max = min(theta_range(2), theta_peaks(i) + roi_margin);
                phi_min = max(phi_range(1), phi_peaks(i) - roi_margin);
                phi_max = min(phi_range(2), phi_peaks(i) + roi_margin);
                
                theta_fine_roi = theta_min:fine_res:theta_max;
                phi_fine_roi = phi_min:fine_res:phi_max;
                grid_fine_roi.theta = theta_fine_roi;
                grid_fine_roi.phi = phi_fine_roi;
                
                [spectrum_fine_roi, ~] = obj.search_2d(virtual_positions, Qn, grid_fine_roi, num_targets);
                
                fine_regions{i}.theta = theta_fine_roi;
                fine_regions{i}.phi = phi_fine_roi;
                fine_regions{i}.spectrum = spectrum_fine_roi;
            end
            
            if verbose
                fprintf('完成 (%.2fs)\n', toc);
            end
            
            %% 第四步：合并谱
            theta_fine = theta_range(1):fine_res:theta_range(2);
            phi_fine = phi_range(1):fine_res:phi_range(2);
            
            % 从粗网格插值
            [Theta_coarse, Phi_coarse] = meshgrid(phi_coarse, theta_coarse);
            [Theta_fine, Phi_fine] = meshgrid(phi_fine, theta_fine);
            spectrum = interp2(Theta_coarse, Phi_coarse, spectrum_coarse, Theta_fine, Phi_fine, 'linear');
            
            % 用细搜索结果覆盖
            for i = 1:length(fine_regions)
                theta_roi = fine_regions{i}.theta;
                phi_roi = fine_regions{i}.phi;
                spectrum_roi = fine_regions{i}.spectrum;
                
                [~, t_start] = min(abs(theta_fine - theta_roi(1)));
                [~, t_end] = min(abs(theta_fine - theta_roi(end)));
                [~, p_start] = min(abs(phi_fine - phi_roi(1)));
                [~, p_end] = min(abs(phi_fine - phi_roi(end)));
                
                spectrum(t_start:t_end, p_start:p_end) = spectrum_roi;
            end
            
            spectrum(isnan(spectrum)) = 0;
            
            search_grid_fine.theta = theta_fine;
            search_grid_fine.phi = phi_fine;
            
            if verbose
                total_points = length(theta_fine) * length(phi_fine);
                coarse_points = length(theta_coarse) * length(phi_coarse);
                fine_points = sum(cellfun(@(x) numel(x.spectrum), fine_regions));
                actual = coarse_points + fine_points;
                fprintf('    ⚡ 加速: %.1fx (计算 %d / 全搜索 %d)\n', total_points/actual, actual, total_points);
            end
        end
        
        function [virtual_positions, virtual_signals, selected_indices] = ...
                build_virtual_array(obj, snapshots, t_axis)
            % 构建虚拟阵列
            
            [num_elements, num_snapshots] = size(snapshots);
            total_virtual = num_elements * num_snapshots;
            
            % 子采样策略
            if total_virtual > obj.max_virtual_elements
                switch obj.subsample_method
                    case 'uniform'
                        subsample_factor = ceil(total_virtual / obj.max_virtual_elements);
                        selected_snapshots = 1:subsample_factor:num_snapshots;
                    case 'random'
                        num_selected = floor(obj.max_virtual_elements / num_elements);
                        selected_snapshots = sort(randperm(num_snapshots, min(num_selected, num_snapshots)));
                    otherwise
                        selected_snapshots = 1:num_snapshots;
                end
            else
                selected_snapshots = 1:num_snapshots;
            end
            
            num_selected = length(selected_snapshots);
            num_virtual = num_elements * num_selected;
            
            virtual_positions = zeros(num_virtual, 3);
            virtual_signals = zeros(num_virtual, 1);
            
            for k = 1:num_selected
                snapshot_idx = selected_snapshots(k);
                t_k = t_axis(snapshot_idx);
                
                pos_k = obj.array_platform.get_mimo_virtual_positions(t_k);
                
                idx_start = (k-1)*num_elements + 1;
                idx_end = k*num_elements;
                virtual_positions(idx_start:idx_end, :) = pos_k;
                virtual_signals(idx_start:idx_end) = snapshots(:, snapshot_idx);
            end
            
            selected_indices = selected_snapshots;
        end
        
        function aperture = calc_aperture(obj, positions)
            % 计算合成孔径
            aperture = struct();
            aperture.x = max(positions(:,1)) - min(positions(:,1));
            aperture.y = max(positions(:,2)) - min(positions(:,2));
            aperture.z = max(positions(:,3)) - min(positions(:,3));
            aperture.total = sqrt(aperture.x^2 + aperture.y^2 + aperture.z^2);
            aperture.x_lambda = aperture.x / obj.lambda;
            aperture.y_lambda = aperture.y / obj.lambda;
            aperture.total_lambda = aperture.total / obj.lambda;
        end
        
        function [spectrum, peaks] = search_1d(obj, positions, Qn, phi_search, num_targets)
            % 1D MUSIC搜索（只搜索方位角phi，假设theta=90°）
            
            num_phi = length(phi_search);
            spectrum = zeros(1, num_phi);
            Qn_proj = Qn * Qn';
            
            for phi_idx = 1:num_phi
                phi = phi_search(phi_idx);
                u = [cosd(phi); sind(phi); 0];
                a = obj.build_steering_vector(positions, u);
                
                denominator = a' * Qn_proj * a;
                spectrum(phi_idx) = 1 / abs(denominator);
            end
            
            [~, peak_indices] = maxk(spectrum, num_targets);
            peaks.phi = phi_search(peak_indices);
            peaks.theta = 90 * ones(size(peaks.phi));
        end
        
        function [spectrum, peaks] = search_2d(obj, positions, Qn, search_grid, num_targets)
            % 2D MUSIC搜索（搜索theta和phi）
            
            theta_search = search_grid.theta;
            phi_search = search_grid.phi;
            num_theta = length(theta_search);
            num_phi = length(phi_search);
            
            spectrum = zeros(num_theta, num_phi);
            Qn_proj = Qn * Qn';
            
            for phi_idx = 1:num_phi
                phi = phi_search(phi_idx);
                for theta_idx = 1:num_theta
                    theta = theta_search(theta_idx);
                    
                    u = [sind(theta)*cosd(phi); sind(theta)*sind(phi); cosd(theta)];
                    a = obj.build_steering_vector(positions, u);
                    
                    denominator = a' * Qn_proj * a;
                    spectrum(theta_idx, phi_idx) = 1 / abs(denominator);
                end
            end
            
            [peaks.theta, peaks.phi, peaks.vals] = obj.find_peaks_2d(spectrum, search_grid, num_targets);
        end
        
        function a = build_steering_vector(obj, positions, u)
            % 构建导向矢量
            % 相位 = 4π/λ × (位置 · 方向)，FMCW雷达双程传播
            phase = 4 * pi / obj.lambda * (positions * u);
            a = exp(1j * phase);
        end
        
        function [theta_peaks, phi_peaks, peak_vals] = find_peaks_2d(obj, spectrum, search_grid, num_peaks)
            % 2D峰值查找
            [sorted_vals, sort_idx] = sort(spectrum(:), 'descend');
            
            theta_peaks = zeros(1, num_peaks);
            phi_peaks = zeros(1, num_peaks);
            peak_vals = zeros(1, num_peaks);
            
            for i = 1:num_peaks
                [theta_idx, phi_idx] = ind2sub(size(spectrum), sort_idx(i));
                theta_peaks(i) = search_grid.theta(theta_idx);
                phi_peaks(i) = search_grid.phi(phi_idx);
                peak_vals(i) = sorted_vals(i);
            end
        end
        
        function beamwidth = estimate_beamwidth(obj, spectrum, angle_axis)
            % 估计3dB主瓣宽度
            spec_db = 10*log10(spectrum / max(spectrum));
            [~, peak_idx] = max(spec_db);
            
            left_idx = find(spec_db(1:peak_idx) < -3, 1, 'last');
            if isempty(left_idx), left_idx = 1; end
            
            right_idx = peak_idx + find(spec_db(peak_idx:end) < -3, 1, 'first') - 1;
            if isempty(right_idx), right_idx = length(angle_axis); end
            
            beamwidth = angle_axis(right_idx) - angle_axis(left_idx);
            if beamwidth <= 0
                beamwidth = angle_axis(2) - angle_axis(1);
            end
        end
    end
end

%% 辅助函数
function val = get_opt(options, field, default)
    if isfield(options, field)
        val = options.(field);
    else
        val = default;
    end
end

