classdef DoaEstimator_DEBUG
    % DoaEstimator_DEBUG: 调试版本，使用标准MUSIC算法
    % 关键修改：导向矢量归一化 + 单快拍处理
    
    properties
        array_platform
        radar_params
        lambda
        use_gpu
    end
    
    methods
        function obj = DoaEstimator_DEBUG(array_platform, radar_params)
            if nargin > 0
                obj.array_platform = array_platform;
                obj.radar_params = radar_params;
                obj.lambda = physconst('LightSpeed') / radar_params.fc;
                obj.use_gpu = false;  % 暂时禁用GPU以便调试
            end
        end
        
        function spectrum = estimate_gmusic(obj, snapshots, t_axis, num_targets, search_grid, positions_override)
            % 标准MUSIC算法（静态阵列版本）
            % 假设：阵列静止 → 所有快拍的导向矢量相同
            % positions_override: 可选，直接指定阵列位置（绕过get_mimo_virtual_positions）
            
            [num_elements, num_snapshots] = size(snapshots);
            
            fprintf('[DEBUG] 快拍维度: %d × %d\n', num_elements, num_snapshots);
            
            % 1. 计算协方差矩阵
            Rxx = (snapshots * snapshots') / num_snapshots;
            
            fprintf('[DEBUG] 协方差矩阵迹: %.2e\n', trace(Rxx));
            
            % 2. 特征分解
            [V, D] = eig(Rxx);
            [eigenvalues, idx] = sort(diag(D), 'descend');
            V = V(:, idx);
            
            fprintf('[DEBUG] 特征值（前3个）: %.2e, %.2e, %.2e\n', eigenvalues(1:3));
            fprintf('[DEBUG] 信号/噪声比: %.2f\n', eigenvalues(1) / eigenvalues(end));
            
            % 3. 噪声子空间
            noise_indices = (num_targets+1):num_elements;
            Qn = V(:, noise_indices);
            
            fprintf('[DEBUG] 噪声子空间维度: %d × %d\n', size(Qn, 1), size(Qn, 2));
            
            % 4. 搜索角度空间
            theta_search = search_grid.theta;
            phi_search = search_grid.phi;
            spectrum = zeros(length(theta_search), length(phi_search));
            
            % 获取阵元位置
            if nargin >= 6 && ~isempty(positions_override)
                positions = positions_override;  % 使用指定位置
                fprintf('[DEBUG] 使用指定阵列位置（%d阵元）\n', size(positions, 1));
            else
                positions = obj.array_platform.get_mimo_virtual_positions(t_axis(1));  % 默认方式
            end
            
            % 5. MUSIC谱计算
            for phi_idx = 1:numel(phi_search)
                phi = phi_search(phi_idx);
                for theta_idx = 1:numel(theta_search)
                    theta = theta_search(theta_idx);
                    
                    % 方向矢量
                    u = [sind(theta)*cosd(phi);
                         sind(theta)*sind(phi);
                         cosd(theta)];
                    
                    % 导向矢量（静态阵列，只需计算一次）
                    % ⚠️ 关键修改：雷达是双程！虚拟阵元=(Tx+Rx)/2，相位系数是4π
                    phase = 4 * pi / obj.lambda * (positions * u);
                    a = exp(1j * phase);
                    
                    % ⚠️ 关键修改：归一化导向矢量
                    a = a / norm(a);
                    
                    % MUSIC谱
                    denominator = a' * (Qn * Qn') * a;
                    spectrum(theta_idx, phi_idx) = 1 / abs(denominator);
                end
            end
            
            fprintf('[DEBUG] 谱范围: [%.2e, %.2e]\n', min(spectrum(:)), max(spectrum(:)));
            fprintf('[DEBUG] 谱动态范围: %.2f\n', max(spectrum(:)) / min(spectrum(:)));
        end
    end
end

