classdef DoaEstimatorIncoherent
    % DoaEstimatorIncoherent: 非相干MUSIC算法，适用于运动/旋转阵列
    % 
    % 与标准GMUSIC的区别：
    % - GMUSIC: 使用所有快拍的协方差矩阵（相干积累）
    % - 本类: 每个快拍独立计算MUSIC谱，然后非相干平均
    % 
    % 适用场景：
    % - 旋转/运动阵列的合成孔径DOA估算
    % - 大角度旋转（>30度）
    % - 目标方向在全局坐标系固定
    
    properties
        array_platform % An instance of the ArrayPlatform class
        radar_params   % Struct with radar parameters
        lambda         % Wavelength
        use_gpu        % Flag to enable GPU acceleration
    end
    
    methods
        function obj = DoaEstimatorIncoherent(array_platform, radar_params)
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
                        fprintf('GPU检测到。非相干MUSIC谱计算将加速。\n');
                        has_shown_gpu_msg = true;
                    end
                end
            end
        end
        
        function [spectrum, weights] = estimate_incoherent_music(obj, snapshots, t_axis, num_targets, search_grid, options)
            % 非相干MUSIC算法
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
            %   weights: [1 x num_snapshots] 每个快拍的权重
            
            if nargin < 6
                options = struct();
            end
            
            % 默认选项
            if ~isfield(options, 'weighting'), options.weighting = 'uniform'; end
            if ~isfield(options, 'verbose'), options.verbose = false; end
            
            [num_virtual_elements, num_snapshots] = size(snapshots);
            
            theta_search = search_grid.theta;
            phi_search = search_grid.phi;
            num_theta = length(theta_search);
            num_phi = length(phi_search);
            
            % 初始化累积谱
            spectrum_accumulated = zeros(num_theta, num_phi);
            weights = ones(1, num_snapshots);  % 默认均匀权重
            
            % 对每个快拍独立处理
            for k = 1:num_snapshots
                if options.verbose && mod(k, 10) == 0
                    fprintf('  处理快拍 %d/%d\n', k, num_snapshots);
                end
                
                % 提取单个快拍
                x_k = snapshots(:, k);
                
                % 计算单快拍的"协方差矩阵"（外积）
                Rxx_k = x_k * x_k';
                
                % 特征分解
                [eigenvectors, eigenvalues] = eig(Rxx_k);
                [eigenvalues_sorted, sort_idx] = sort(diag(eigenvalues), 'descend');
                
                % 提取噪声子空间
                signal_indices = sort_idx(1:num_targets);
                noise_indices = sort_idx(num_targets+1:end);
                Qn_k = eigenvectors(:, noise_indices);
                
                % 计算该快拍的权重（基于信号强度）
                if strcmp(options.weighting, 'snr')
                    signal_power = sum(eigenvalues_sorted(1:num_targets));
                    noise_power = mean(eigenvalues_sorted(num_targets+1:end));
                    weights(k) = signal_power / noise_power;
                end
                
                % 对该快拍计算MUSIC谱
                spectrum_k = zeros(num_theta, num_phi);
                
                for phi_idx = 1:num_phi
                    phi = phi_search(phi_idx);
                    for theta_idx = 1:num_theta
                        theta = theta_search(theta_idx);
                        
                        % 构建该方向的导向矢量（只在时刻k）
                        u = [sind(theta)*cosd(phi); sind(theta)*sind(phi); cosd(theta)];
                        a_k = obj.build_steering_vector(t_axis(k), u);
                        
                        % MUSIC谱
                        denominator = a_k' * (Qn_k * Qn_k') * a_k;
                        spectrum_k(theta_idx, phi_idx) = 1 / abs(denominator);
                    end
                end
                
                % 加权累加
                spectrum_accumulated = spectrum_accumulated + weights(k) * spectrum_k;
            end
            
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

