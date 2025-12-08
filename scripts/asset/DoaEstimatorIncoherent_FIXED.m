classdef DoaEstimatorIncoherent_FIXED
    % DoaEstimatorIncoherent_FIXED: 修复后的非相干MUSIC算法
    % 
    % 修复说明：
    % - 原版Bug: 对单个快拍做外积x*x'，得到秩1矩阵，噪声子空间无意义
    % - 修复方案: 将快拍分段，每段用多个快拍计算协方差矩阵，然后非相干平均各段的MUSIC谱
    % 
    % 原理：
    % - 非相干合成孔径MUSIC (Incoherent Synthetic Aperture MUSIC)
    % - 将观测时间分成多个短时段
    % - 每段内：阵列位置变化不大，可以近似为固定阵列
    % - 每段独立计算MUSIC谱（使用该段内多个快拍的协方差矩阵）
    % - 最后非相干平均各段的谱
    
    properties
        array_platform % An instance of the ArrayPlatform class
        radar_params   % Struct with radar parameters
        lambda         % Wavelength
        use_gpu        % Flag to enable GPU acceleration
    end
    
    methods
        function obj = DoaEstimatorIncoherent_FIXED(array_platform, radar_params)
            % Constructor
            if nargin > 0
                obj.array_platform = array_platform;
                obj.radar_params = radar_params;
                obj.lambda = physconst('LightSpeed') / radar_params.fc;
                
                % GPU check
                obj.use_gpu = (gpuDeviceCount > 0);
                if obj.use_gpu
                    persistent has_shown_gpu_msg;
                    if isempty(has_shown_gpu_msg)
                        fprintf('GPU检测到。非相干MUSIC谱计算将加速（修复版）。\n');
                        has_shown_gpu_msg = true;
                    end
                end
            end
        end
        
        function [spectrum, weights] = estimate_incoherent_music(obj, snapshots, t_axis, num_targets, search_grid, options)
            % 非相干MUSIC算法（修复版）
            %
            % 输入：
            %   snapshots: [num_virtual_elements x num_snapshots] 接收信号
            %   t_axis: [1 x num_snapshots] 时间轴
            %   num_targets: 目标数量
            %   search_grid: 搜索网格 struct with .theta and .phi
            %   options: (可选) 处理选项
            %
            % 输出：
            %   spectrum: [num_theta x num_phi] MUSIC谱
            %   weights: [1 x num_segments] 每段的权重
            
            if nargin < 6
                options = struct();
            end
            
            % 默认选项
            if ~isfield(options, 'weighting'), options.weighting = 'uniform'; end
            if ~isfield(options, 'verbose'), options.verbose = false; end
            if ~isfield(options, 'snapshots_per_segment'), options.snapshots_per_segment = 8; end
            
            [num_virtual_elements, num_snapshots] = size(snapshots);
            
            theta_search = search_grid.theta;
            phi_search = search_grid.phi;
            num_theta = length(theta_search);
            num_phi = length(phi_search);
            
            % 计算分段参数
            snapshots_per_segment = min(options.snapshots_per_segment, num_snapshots);
            num_segments = floor(num_snapshots / snapshots_per_segment);
            
            if num_segments == 0
                error('快拍数不足，至少需要%d个快拍', snapshots_per_segment);
            end
            
            % 初始化累积谱
            spectrum_accumulated = zeros(num_theta, num_phi);
            weights = ones(1, num_segments);  % 默认均匀权重
            
            % 对每段独立处理
            fprintf('  处理 %d 段（每段%d快拍）: ', num_segments, snapshots_per_segment);
            
            for seg = 1:num_segments
                % 显示进度
                if mod(seg, 10) == 0 || seg == num_segments
                    fprintf(' %d/%d', seg, num_segments);
                end
                
                % 提取该段的快拍
                idx_start = (seg-1)*snapshots_per_segment + 1;
                idx_end = seg*snapshots_per_segment;
                X_seg = snapshots(:, idx_start:idx_end);
                t_seg = t_axis(idx_start:idx_end);
                t_center = mean(t_seg);  % 该段的中心时刻
                
                % 计算该段的协方差矩阵（多个快拍平均）
                Rxx_seg = (X_seg * X_seg') / snapshots_per_segment;
                
                % 特征分解
                [eigenvectors, eigenvalues] = eig(Rxx_seg);
                [eigenvalues_sorted, sort_idx] = sort(diag(eigenvalues), 'descend');
                
                % 提取噪声子空间
                noise_indices = sort_idx(num_targets+1:end);
                Qn_seg = eigenvectors(:, noise_indices);
                
                % 计算该段的权重（基于信号强度）
                if strcmp(options.weighting, 'snr')
                    signal_power = sum(eigenvalues_sorted(1:num_targets));
                    noise_power = mean(eigenvalues_sorted(num_targets+1:end));
                    weights(seg) = signal_power / noise_power;
                end
                
                % 对该段计算MUSIC谱（使用该段中心时刻的阵列配置）
                spectrum_seg = zeros(num_theta, num_phi);
                
                for phi_idx = 1:num_phi
                    phi = phi_search(phi_idx);
                    for theta_idx = 1:num_theta
                        theta = theta_search(theta_idx);
                        
                        % 构建该方向的导向矢量（在该段中心时刻）
                        u = [sind(theta)*cosd(phi); sind(theta)*sind(phi); cosd(theta)];
                        a_seg = obj.build_steering_vector(t_center, u);
                        
                        % MUSIC谱
                        denominator = a_seg' * (Qn_seg * Qn_seg') * a_seg;
                        spectrum_seg(theta_idx, phi_idx) = 1 / abs(denominator);
                    end
                end
                
                % 加权累加
                spectrum_accumulated = spectrum_accumulated + weights(seg) * spectrum_seg;
            end
            
            fprintf('\n  分段处理完成！\n');
            
            % 归一化权重
            weights = weights / sum(weights);
            
            % 平均谱
            spectrum = spectrum_accumulated / sum(weights);
        end
        
        function a = build_steering_vector(obj, t, u)
            % 构建单个时刻的导向矢量
            %
            % 输入：
            %   t: 标量，时间
            %   u: [3x1] 目标方向单位向量
            %
            % 输出：
            %   a: [num_virtual_elements x 1] 导向矢量
            
            % 获取该时刻的虚拟阵元位置
            positions = obj.array_platform.get_mimo_virtual_positions(t);
            
            % 计算相位
            lambda = obj.radar_params.lambda;
            phase = 4 * pi / lambda * (positions * u);
            
            % 导向矢量
            a = exp(1j * phase);
        end
    end
    
    methods (Static)
        function [theta_peaks, phi_peaks, peak_vals] = find_peaks(spectrum, grid, num_peaks)
            % 峰值查找（与标准MUSIC相同）
            [sorted_vals, sort_idx] = sort(spectrum(:), 'descend');
            
            theta_peaks = zeros(1, num_peaks);
            phi_peaks = zeros(1, num_peaks);
            peak_vals = zeros(1, num_peaks);
            
            for i = 1:num_peaks
                peak_idx = sort_idx(i);
                peak_vals(i) = sorted_vals(i);
                [theta_idx, phi_idx] = ind2sub(size(spectrum), peak_idx);
                theta_peaks(i) = grid.theta(theta_idx);
                phi_peaks(i) = grid.phi(phi_idx);
            end
        end
    end
end



