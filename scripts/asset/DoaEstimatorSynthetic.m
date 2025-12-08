classdef DoaEstimatorSynthetic
    % DoaEstimatorSynthetic: 合成孔径DOA估计器
    %
    % 支持两种方法：
    %   1. 相干合成孔径波束形成 (CSA-BF) - 默认
    %      将M×K快拍展开为M×K虚拟阵列，使用匹配滤波
    %      优势：充分利用合成孔径，分辨率最高
    %
    %   2. 非相干合成孔径MUSIC (ISA-MUSIC)
    %      分段处理，每段独立MUSIC，谱非相干累加
    %      优势：对相位误差鲁棒
    %
    % 用法示例：
    %   estimator = DoaEstimatorSynthetic(array_platform, radar_params);
    %   [spectrum, peaks, info] = estimator.estimate(snapshots, t_axis, search_grid, num_targets);
    %
    %   % 使用ISA-MUSIC模式
    %   options.method = 'isa-music';
    %   [spectrum, peaks, info] = estimator.estimate(..., options);
    
    properties
        array_platform  % ArrayPlatform实例
        radar_params    % 雷达参数结构体
        lambda          % 波长
        
        % ISA-MUSIC分段参数
        segment_size    % 每段快拍数（默认8）
        segment_overlap % 段间重叠率（默认0.5）
    end
    
    methods
        function obj = DoaEstimatorSynthetic(array_platform, radar_params, options)
            % 构造函数
            
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
                
                % 默认分段参数（用于ISA-MUSIC模式）
                % 注意：segment_size要小，保证段内阵列位置近似不变
                % 对于5m/s运动，每快拍(7.8ms)移动约4cm=0.4λ
                % segment_size=4时，段内移动约1.6λ，可接受
                obj.segment_size = 4;
                obj.segment_overlap = 0.5;
                
                if nargin > 2 && ~isempty(options)
                    if isfield(options, 'segment_size')
                        obj.segment_size = options.segment_size;
                    end
                    if isfield(options, 'segment_overlap')
                        obj.segment_overlap = options.segment_overlap;
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
            %   search_grid  - 搜索网格 (struct with .phi，可选 .theta)
            %   num_targets  - 目标数量
            %   options      - (可选) 选项
            %     .method       - 'csa-bf' (相干波束形成，默认) 或 'isa-music' (非相干MUSIC)
            %     .search_mode  - '1d' 或 '2d' 或 'auto'
            %     .use_cfar     - 是否使用CFAR检测
            %
            % 输出:
            %   spectrum - 角度谱
            %   peaks    - 峰值位置结构体 (.phi, .theta, .vals)
            %   info     - 附加信息
            
            if nargin < 6
                options = struct();
            end
            
            method = get_opt(options, 'method', 'csa-bf');
            search_mode = get_opt(options, 'search_mode', 'auto');
            use_cfar = get_opt(options, 'use_cfar', false);
            
            % 自动检测搜索模式
            if strcmp(search_mode, 'auto')
                if isfield(search_grid, 'theta') && ~isempty(search_grid.theta)
                    search_mode = '2d';
                else
                    search_mode = '1d';
                end
            end
            
            % 根据方法选择算法
            switch lower(method)
                case 'csa-bf'
                    % 相干合成孔径波束形成（匹配滤波）
                    [spectrum, info] = obj.csa_beamforming(snapshots, t_axis, search_grid, search_mode);
                    
                case 'isa-music'
                    % 非相干合成孔径MUSIC
                    [spectrum, info] = obj.isa_music(snapshots, t_axis, search_grid, num_targets, search_mode);
                    
                case 'csa-music'
                    % 相干合成孔径MUSIC（时间平滑）
                    [spectrum, info] = obj.csa_music(snapshots, t_axis, search_grid, num_targets, search_mode);
                    
                otherwise
                    error('未知方法: %s。支持 "csa-bf", "isa-music" 或 "csa-music"', method);
            end
            
            % 峰值检测
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
            
            info.method = method;
            info.search_mode = search_mode;
        end
        
        %% ═══════════════════════════════════════════════════════════════════
        %  方法1: 相干合成孔径波束形成 (CSA-BF)
        %  原理：M阵元×K快拍 → M×K虚拟阵列，使用匹配滤波
        %% ═══════════════════════════════════════════════════════════════════
        function [spectrum, info] = csa_beamforming(obj, snapshots, t_axis, search_grid, search_mode)
            % 相干合成孔径波束形成
            %
            % 核心公式：
            %   P(θ) = |a(θ)' × x_virtual|² / |a(θ)|²
            %
            % 其中：
            %   x_virtual ∈ C^(M×K) - 虚拟阵列信号向量
            %   a(θ) ∈ C^(M×K) - 虚拟阵列导向矢量
            
            [num_elements, num_snapshots] = size(snapshots);
            num_virtual = num_elements * num_snapshots;
            
            % 1. 构建虚拟阵列位置和信号
            virtual_positions = zeros(num_virtual, 3);
            virtual_signals = zeros(num_virtual, 1);
            
            for k = 1:num_snapshots
                t_k = t_axis(k);
                positions_k = obj.array_platform.get_mimo_virtual_positions(t_k);
                
                idx_start = (k-1)*num_elements + 1;
                idx_end = k*num_elements;
                virtual_positions(idx_start:idx_end, :) = positions_k;
                virtual_signals(idx_start:idx_end) = snapshots(:, k);
            end
            
            % 2. 计算合成孔径
            aperture = obj.calc_aperture(virtual_positions);
            
            % 3. 波束形成搜索
            if strcmp(search_mode, '1d')
                spectrum = obj.beamforming_1d(virtual_positions, virtual_signals, search_grid.phi);
            else
                spectrum = obj.beamforming_2d(virtual_positions, virtual_signals, search_grid);
            end
            
            % 输出信息
            info.virtual_positions = virtual_positions;
            info.synthetic_aperture = aperture;
            info.num_virtual = num_virtual;
        end
        
        function spectrum = beamforming_1d(obj, positions, signals, phi_search)
            % 1D波束形成
            % P(φ) = |a(φ)' * x|² / |a(φ)|²
            
            num_phi = length(phi_search);
            spectrum = zeros(1, num_phi);
            
            for phi_idx = 1:num_phi
                phi = phi_search(phi_idx);
                u = [cosd(phi); sind(phi); 0];
                a = obj.build_steering_vector(positions, u);
                
                % 匹配滤波输出
                spectrum(phi_idx) = abs(a' * signals)^2 / (a' * a);
            end
        end
        
        function spectrum = beamforming_2d(obj, positions, signals, search_grid)
            % 2D波束形成
            
            theta_search = search_grid.theta;
            phi_search = search_grid.phi;
            num_theta = length(theta_search);
            num_phi = length(phi_search);
            
            spectrum = zeros(num_theta, num_phi);
            
            for phi_idx = 1:num_phi
                phi = phi_search(phi_idx);
                for theta_idx = 1:num_theta
                    theta = theta_search(theta_idx);
                    
                    u = [sind(theta)*cosd(phi); sind(theta)*sind(phi); cosd(theta)];
                    a = obj.build_steering_vector(positions, u);
                    
                    spectrum(theta_idx, phi_idx) = abs(a' * signals)^2 / real(a' * a);
                end
            end
        end
        
        %% ═══════════════════════════════════════════════════════════════════
        %  方法2: 非相干合成孔径MUSIC (ISA-MUSIC)
        %  原理：分段处理，每段独立MUSIC，谱非相干累加
        %% ═══════════════════════════════════════════════════════════════════
        function [spectrum, info] = isa_music(obj, snapshots, t_axis, search_grid, num_targets, search_mode)
            % 非相干合成孔径MUSIC（改进版：带段内相位补偿）
            %
            % 改进：对每个快拍根据其实际阵列位置进行相位补偿，
            % 补偿到段中心时刻的位置，使段内协方差矩阵更准确
            
            [num_elements, num_snapshots] = size(snapshots);
            
            % 1. 分段参数
            seg_size = min(obj.segment_size, num_snapshots);
            seg_step = max(1, round(seg_size * (1 - obj.segment_overlap)));
            seg_starts = 1:seg_step:(num_snapshots - seg_size + 1);
            num_segments = length(seg_starts);
            
            if num_segments == 0
                seg_starts = 1;
                seg_size = num_snapshots;
                num_segments = 1;
            end
            
            % 2. 初始化累积谱
            if strcmp(search_mode, '1d')
                accumulated_spectrum = zeros(1, length(search_grid.phi));
            else
                accumulated_spectrum = zeros(length(search_grid.theta), length(search_grid.phi));
            end
            
            % 3. 收集虚拟阵列位置
            all_virtual_positions = [];
            
            % 4. 分段处理
            for seg_idx = 1:num_segments
                seg_start = seg_starts(seg_idx);
                seg_end = min(seg_start + seg_size - 1, num_snapshots);
                seg_indices = seg_start:seg_end;
                num_seg_snapshots = length(seg_indices);
                
                seg_t_axis = t_axis(seg_indices);
                
                % 段中心时刻位置（参考位置）
                t_center = mean(seg_t_axis);
                positions_ref = obj.array_platform.get_mimo_virtual_positions(t_center);
                all_virtual_positions = [all_virtual_positions; positions_ref];
                
                % 段内相位补偿：将每个快拍补偿到参考位置
                seg_snapshots_compensated = zeros(num_elements, num_seg_snapshots);
                for k = 1:num_seg_snapshots
                    t_k = seg_t_axis(k);
                    positions_k = obj.array_platform.get_mimo_virtual_positions(t_k);
                    
                    % 位置差
                    delta_pos = positions_k - positions_ref;
                    
                    % 对于宽带方向估计，需要知道大致方向才能补偿
                    % 这里使用简化方法：不补偿（假设段内移动足够小）
                    % 或者可以用迭代方法先粗估方向再补偿
                    seg_snapshots_compensated(:, k) = snapshots(:, seg_indices(k));
                end
                
                % 协方差矩阵（满秩）
                Rxx = (seg_snapshots_compensated * seg_snapshots_compensated') / num_seg_snapshots;
                
                % 特征分解
                [V, D] = eig(Rxx);
                [~, idx] = sort(diag(D), 'descend');
                V = V(:, idx);
                
                % 噪声子空间
                noise_dim = max(1, num_elements - num_targets);
                Qn = V(:, (num_targets+1):end);
                
                % MUSIC谱 - 使用参考位置
                if strcmp(search_mode, '1d')
                    seg_spectrum = obj.music_spectrum_1d(positions_ref, Qn, search_grid.phi);
                else
                    seg_spectrum = obj.music_spectrum_2d(positions_ref, Qn, search_grid);
                end
                
                accumulated_spectrum = accumulated_spectrum + seg_spectrum;
            end
            
            spectrum = accumulated_spectrum / num_segments;
            
            info.virtual_positions = all_virtual_positions;
            info.synthetic_aperture = obj.calc_aperture(all_virtual_positions);
            info.num_segments = num_segments;
        end
        
        function spectrum = music_spectrum_1d(obj, positions, Qn, phi_search)
            % 1D MUSIC谱
            
            num_phi = length(phi_search);
            spectrum = zeros(1, num_phi);
            Qn_proj = Qn * Qn';
            
            for phi_idx = 1:num_phi
                phi = phi_search(phi_idx);
                u = [cosd(phi); sind(phi); 0];
                a = obj.build_steering_vector(positions, u);
                
                denom = real(a' * Qn_proj * a);
                spectrum(phi_idx) = 1 / max(denom, 1e-12);
            end
        end
        
        function spectrum = music_spectrum_2d(obj, positions, Qn, search_grid)
            % 2D MUSIC谱
            
            theta_search = search_grid.theta;
            phi_search = search_grid.phi;
            spectrum = zeros(length(theta_search), length(phi_search));
            Qn_proj = Qn * Qn';
            
            for phi_idx = 1:length(phi_search)
                phi = phi_search(phi_idx);
                for theta_idx = 1:length(theta_search)
                    theta = theta_search(theta_idx);
                    u = [sind(theta)*cosd(phi); sind(theta)*sind(phi); cosd(theta)];
                    a = obj.build_steering_vector(positions, u);
                    
                    denom = real(a' * Qn_proj * a);
                    spectrum(theta_idx, phi_idx) = 1 / max(denom, 1e-12);
                end
            end
        end
        
        %% ═══════════════════════════════════════════════════════════════════
        %  方法3: 相干合成孔径MUSIC (CSA-MUSIC) - 时间平滑
        %  原理：使用时间平滑构造满秩协方差矩阵，在虚拟阵列上做MUSIC
        %% ═══════════════════════════════════════════════════════════════════
        function [spectrum, info] = csa_music(obj, snapshots, t_axis, search_grid, num_targets, search_mode)
            % 相干合成孔径MUSIC（时间平滑方法）
            %
            % 原理：
            %   1. 构建虚拟阵列（M元 × K快拍 → L个子阵列）
            %   2. 使用时间平滑技术构造满秩协方差矩阵
            %   3. 在虚拟阵列上执行MUSIC
            %
            % 时间平滑：将K个快拍分成重叠的子阵列
            %   子阵列大小 = K - L + 1
            %   子阵列数量 = L（用于协方差估计）
            
            [num_elements, num_snapshots] = size(snapshots);
            
            % 时间平滑参数
            % L = 子阵列数量（用于协方差矩阵估计）
            % 子阵列大小 = num_snapshots - L + 1
            L = min(num_snapshots - num_elements, floor(num_snapshots / 2));
            L = max(L, num_targets + 1);  % 至少需要L > num_targets
            subarray_size = num_snapshots - L + 1;
            
            % 虚拟阵列维度 = M × subarray_size
            num_virtual = num_elements * subarray_size;
            
            % 1. 构建虚拟阵列位置
            % 取中间时刻的子阵列位置作为参考
            ref_start = floor(L / 2) + 1;
            ref_indices = ref_start:(ref_start + subarray_size - 1);
            
            virtual_positions = zeros(num_virtual, 3);
            for k = 1:subarray_size
                t_k = t_axis(ref_indices(k));
                positions_k = obj.array_platform.get_mimo_virtual_positions(t_k);
                idx_start = (k-1)*num_elements + 1;
                idx_end = k*num_elements;
                virtual_positions(idx_start:idx_end, :) = positions_k;
            end
            
            % 2. 时间平滑协方差矩阵估计
            Rxx = zeros(num_virtual, num_virtual);
            
            for l = 1:L
                % 第l个子阵列的快拍索引
                sub_indices = l:(l + subarray_size - 1);
                
                % 构建该子阵列的虚拟信号向量
                x_virtual = zeros(num_virtual, 1);
                for k = 1:subarray_size
                    snapshot_idx = sub_indices(k);
                    idx_start = (k-1)*num_elements + 1;
                    idx_end = k*num_elements;
                    x_virtual(idx_start:idx_end) = snapshots(:, snapshot_idx);
                end
                
                % 累加协方差矩阵
                Rxx = Rxx + x_virtual * x_virtual';
            end
            Rxx = Rxx / L;
            
            % 3. 特征分解
            [V, D] = eig(Rxx);
            [eigenvalues, idx] = sort(diag(D), 'descend');
            V = V(:, idx);
            
            % 噪声子空间
            noise_dim = max(1, num_virtual - num_targets);
            Qn = V(:, (num_targets+1):end);
            
            % 4. MUSIC谱搜索
            if strcmp(search_mode, '1d')
                spectrum = obj.music_spectrum_1d_virtual(virtual_positions, Qn, search_grid.phi);
            else
                spectrum = obj.music_spectrum_2d_virtual(virtual_positions, Qn, search_grid);
            end
            
            % 输出信息
            info.virtual_positions = virtual_positions;
            info.synthetic_aperture = obj.calc_aperture(virtual_positions);
            info.num_virtual = num_virtual;
            info.subarray_size = subarray_size;
            info.num_subarrays = L;
            info.eigenvalues = eigenvalues;
        end
        
        function spectrum = music_spectrum_1d_virtual(obj, positions, Qn, phi_search)
            % 虚拟阵列1D MUSIC谱
            num_phi = length(phi_search);
            spectrum = zeros(1, num_phi);
            Qn_proj = Qn * Qn';
            
            for phi_idx = 1:num_phi
                phi = phi_search(phi_idx);
                u = [cosd(phi); sind(phi); 0];
                a = obj.build_steering_vector(positions, u);
                
                denom = real(a' * Qn_proj * a);
                spectrum(phi_idx) = 1 / max(denom, 1e-12);
            end
        end
        
        function spectrum = music_spectrum_2d_virtual(obj, positions, Qn, search_grid)
            % 虚拟阵列2D MUSIC谱
            theta_search = search_grid.theta;
            phi_search = search_grid.phi;
            spectrum = zeros(length(theta_search), length(phi_search));
            Qn_proj = Qn * Qn';
            
            for phi_idx = 1:length(phi_search)
                phi = phi_search(phi_idx);
                for theta_idx = 1:length(theta_search)
                    theta = theta_search(theta_idx);
                    u = [sind(theta)*cosd(phi); sind(theta)*sind(phi); cosd(theta)];
                    a = obj.build_steering_vector(positions, u);
                    
                    denom = real(a' * Qn_proj * a);
                    spectrum(theta_idx, phi_idx) = 1 / max(denom, 1e-12);
                end
            end
        end
        
        %% ═══════════════════════════════════════════════════════════════════
        %  辅助函数
        %% ═══════════════════════════════════════════════════════════════════
        function a = build_steering_vector(obj, positions, u)
            % 构建导向矢量（平面波模型）
            %
            % 相位 = 4π/λ × (位置·方向)，FMCW雷达双程传播
            % 
            % 注意：必须与信号生成器使用相同的模型！
            % SignalGeneratorSimple 应该也使用平面波近似
            
            phase = 4 * pi / obj.lambda * (positions * u);
            a = exp(1j * phase);
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
        
        function [theta_peaks, phi_peaks, peak_vals] = find_peaks_2d(obj, spectrum, search_grid, num_peaks)
            % 2D峰值查找
            [sorted_vals, sort_idx] = sort(spectrum(:), 'descend');
            
            theta_peaks = zeros(1, num_peaks);
            phi_peaks = zeros(1, num_peaks);
            peak_vals = zeros(1, num_peaks);
            
            for i = 1:min(num_peaks, length(sorted_vals))
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
