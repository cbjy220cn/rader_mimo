function [theta_peaks, phi_peaks, peak_vals, peak_mask] = find_peaks_cfar(spectrum, search_grid, num_peaks, options)
% FIND_PEAKS_CFAR 使用CA-CFAR算法的增强峰值检测
%
% 输入:
%   spectrum     - MUSIC谱 (theta × phi)
%   search_grid  - 搜索网格结构体 (.theta, .phi)
%   num_peaks    - 期望找到的峰值数量
%   options      - 可选参数结构体
%                  .numGuard: 保护单元数 (默认2)
%                  .numTrain: 训练单元数 (默认4)
%                  .P_fa: 虚警概率 (默认1e-4)
%                  .SNR_offset_dB: SNR偏移量dB (默认-10)
%                  .min_separation: 峰值最小间隔（度，默认3）
%
% 输出:
%   theta_peaks  - 检测到的theta角度
%   phi_peaks    - 检测到的phi角度
%   peak_vals    - 峰值幅度
%   peak_mask    - CFAR检测掩码

    % 默认参数
    if nargin < 4
        options = struct();
    end
    
    numGuard = get_option(options, 'numGuard', 2);
    numTrain = get_option(options, 'numTrain', 4);
    P_fa = get_option(options, 'P_fa', 1e-4);
    SNR_offset_dB = get_option(options, 'SNR_offset_dB', -10);
    min_separation = get_option(options, 'min_separation', 3);  % 度
    
    %% 1. 将谱转换为dB
    spectrum_linear = spectrum;
    spectrum_dB = 10*log10(spectrum / max(spectrum(:)));
    
    %% 2. CA-CFAR检测
    numTrain2D = (2*numTrain+1)^2 - (2*numGuard+1)^2;
    peak_mask = zeros(size(spectrum_dB));
    
    [num_theta, num_phi] = size(spectrum_dB);
    margin = numTrain + numGuard;
    
    for i = margin+1 : num_theta - margin
        for j = margin+1 : num_phi - margin
            % 计算训练单元的平均噪声功率
            % 整个窗口
            window_all = spectrum_dB(i-margin:i+margin, j-margin:j+margin);
            % 保护单元+CUT
            guard_cut = spectrum_dB(i-numGuard:i+numGuard, j-numGuard:j+numGuard);
            
            % 噪声功率估计（训练单元平均）
            Pn = (sum(window_all(:)) - sum(guard_cut(:))) / numTrain2D;
            
            % 阈值因子
            alpha = numTrain2D * (P_fa^(-1/numTrain2D) - 1);
            threshold = alpha * Pn;
            
            % 判决
            if (spectrum_dB(i,j) > threshold) && (spectrum_dB(i,j) > SNR_offset_dB)
                peak_mask(i,j) = 1;
            end
        end
    end
    
    %% 3. 查找CFAR检测到的峰值位置
    [theta_idx_all, phi_idx_all] = find(peak_mask);
    
    if isempty(theta_idx_all)
        % 如果CFAR没检测到任何峰值，回退到传统方法
        warning('CA-CFAR未检测到峰值，使用传统方法');
        [theta_peaks, phi_peaks, peak_vals] = find_peaks_traditional(spectrum_linear, search_grid, num_peaks);
        return;
    end
    
    %% 4. 聚类：合并距离很近的峰值
    % 将CFAR检测到的点按幅度排序
    peak_values = zeros(length(theta_idx_all), 1);
    for k = 1:length(theta_idx_all)
        peak_values(k) = spectrum_linear(theta_idx_all(k), phi_idx_all(k));
    end
    
    [~, sort_idx] = sort(peak_values, 'descend');
    theta_idx_sorted = theta_idx_all(sort_idx);
    phi_idx_sorted = phi_idx_all(sort_idx);
    
    % 贪心聚类：从最强峰值开始，移除附近的弱峰值
    theta_angles = search_grid.theta;
    phi_angles = search_grid.phi;
    dtheta = theta_angles(2) - theta_angles(1);
    dphi = phi_angles(2) - phi_angles(1);
    
    % 转换最小间隔为索引数
    min_sep_idx_theta = ceil(min_separation / dtheta);
    min_sep_idx_phi = ceil(min_separation / dphi);
    
    selected_peaks = [];
    used_mask = false(size(theta_idx_sorted));
    
    for k = 1:length(theta_idx_sorted)
        if used_mask(k)
            continue;
        end
        
        % 选择当前峰值
        selected_peaks = [selected_peaks; k];
        
        % 标记附近的点为已使用
        for m = k+1:length(theta_idx_sorted)
            if used_mask(m)
                continue;
            end
            
            % 计算角度距离
            dtheta_idx = abs(theta_idx_sorted(k) - theta_idx_sorted(m));
            dphi_idx = abs(phi_idx_sorted(k) - phi_idx_sorted(m));
            
            % 如果太近，标记为已使用
            if (dtheta_idx < min_sep_idx_theta) && (dphi_idx < min_sep_idx_phi)
                used_mask(m) = true;
            end
        end
        
        % 如果已经找到足够多的峰值，停止
        if length(selected_peaks) >= num_peaks
            break;
        end
    end
    
    %% 5. 输出结果
    num_found = min(length(selected_peaks), num_peaks);
    theta_peaks = zeros(1, num_found);
    phi_peaks = zeros(1, num_found);
    peak_vals = zeros(1, num_found);
    
    for k = 1:num_found
        idx = selected_peaks(k);
        theta_peaks(k) = theta_angles(theta_idx_sorted(idx));
        phi_peaks(k) = phi_angles(phi_idx_sorted(idx));
        peak_vals(k) = spectrum_linear(theta_idx_sorted(idx), phi_idx_sorted(idx));
    end
end

%% 辅助函数：获取选项参数
function val = get_option(options, field, default)
    if isfield(options, field)
        val = options.(field);
    else
        val = default;
    end
end

%% 辅助函数：传统峰值检测（回退方案）
function [theta_peaks, phi_peaks, peak_vals] = find_peaks_traditional(spectrum, search_grid, num_peaks)
    % 简单的全局最大值搜索
    [num_theta, num_phi] = size(spectrum);
    spectrum_flat = spectrum(:);
    [sorted_vals, sorted_idx] = sort(spectrum_flat, 'descend');
    
    theta_peaks = zeros(1, num_peaks);
    phi_peaks = zeros(1, num_peaks);
    peak_vals = zeros(1, num_peaks);
    
    for k = 1:num_peaks
        if k > length(sorted_idx)
            break;
        end
        
        [theta_idx, phi_idx] = ind2sub([num_theta, num_phi], sorted_idx(k));
        theta_peaks(k) = search_grid.theta(theta_idx);
        phi_peaks(k) = search_grid.phi(phi_idx);
        peak_vals(k) = sorted_vals(k);
    end
end

