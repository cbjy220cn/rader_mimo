classdef SignalGeneratorSimple
    % SignalGeneratorSimple: 简化的信号生成器
    % 直接生成空间相位信号，不经过FMCW调制/解调
    % 用于DOA算法验证
    
    properties
        radar_params   % 雷达参数（只用fc和lambda）
        array_platform % ArrayPlatform实例
        targets        % Target列表
    end
    
    methods
        function obj = SignalGeneratorSimple(radar_params, array_platform, targets)
            % 构造函数
            if nargin > 0
                obj.radar_params = radar_params;
                obj.array_platform = array_platform;
                obj.targets = targets;
            end
        end
        
        function snapshots = generate_snapshots(obj, t_axis, snr_db)
            % 生成快拍数据
            % 直接计算空间相位，不经过FMCW调制
            %
            % 输入:
            %   t_axis - 时间轴 [1 x num_snapshots]
            %   snr_db - 信噪比(dB)
            %
            % 输出:
            %   snapshots - [num_virtual_elements x num_snapshots]
            
            num_snapshots = length(t_axis);
            num_targets = length(obj.targets);
            num_virtual_elements = obj.array_platform.get_num_virtual_elements();
            lambda = obj.radar_params.lambda;
            
            % 注意：不再固定随机种子
            % 随机种子应由调用者在外部控制，以保证蒙特卡洛试验的有效性
            % 旧代码: rng(0); 这会导致所有试验的目标幅度相同，严重影响统计
            
            % 初始化输出
            snapshots = zeros(num_virtual_elements, num_snapshots);
            
            % 为每个目标生成部分相干的随机复数幅度序列
            % 使用AR(1)模型模拟目标散射的时间相关性
            %
            % 物理背景：
            %   - 真实目标在短时间内散射特性近似恒定（相干时间）
            %   - 但存在缓慢的随机波动（目标微动、大气扰动等）
            %   - AR(1)模型：α(k) = ρ×α(k-1) + √(1-ρ²)×w(k)
            %
            % 参数 rho 的意义：
            %   - rho = 1：完全相干，所有快拍幅度相同
            %   - rho = 0：完全独立，快拍间无相关性
            %   - rho ≈ 0.95：部分相干，相邻快拍高度相关，远距快拍逐渐独立
            %
            % 对时间平滑MUSIC的影响：
            %   - 需要部分相干才能正常工作
            %   - rho太小会破坏合成孔径的空间一致性
            %   - rho太大会导致协方差矩阵秩不足
            
            rho = 0.95;  % 相邻快拍相关系数（可调）
            
            target_amplitudes = zeros(num_targets, num_snapshots);
            for i = 1:num_targets
                % AR(1)过程生成部分相干序列
                alpha_prev = (randn + 1j*randn) / sqrt(2);
                for k = 1:num_snapshots
                    innovation = (randn + 1j*randn) / sqrt(2);
                    alpha_curr = rho * alpha_prev + sqrt(1 - rho^2) * innovation;
                    target_amplitudes(i, k) = alpha_curr;
                    alpha_prev = alpha_curr;
                end
            end
            
            % 遍历每个快拍
            for k = 1:num_snapshots
                t = t_axis(k);
                
                % 获取该时刻的虚拟阵元位置（全局坐标系）
                % 关键：位置包含运动偏移，这正是合成孔径的核心！
                virtual_positions = obj.array_platform.get_mimo_virtual_positions(t);
                
                % 遍历每个目标
                for i = 1:num_targets
                    target_obj = obj.targets{i};
                    target_pos = target_obj.get_position_at(t);
                    target_rcs = target_obj.rcs;
                    
                    % 目标方向单位矢量（从原点指向目标）
                    % 使用原点作为固定参考点，这样运动效应才能体现！
                    target_range = norm(target_pos);
                    target_dir = target_pos / target_range;
                    
                    % 计算每个阵元的相位
                    for v = 1:num_virtual_elements
                        virt_pos = virtual_positions(v, :);
                        
                        % 相位 = 阵元位置在信号来向的投影 × 4π/λ
                        % 远场近似：信号为平面波
                        % 位置包含运动偏移，这样不同快拍的相位会不同
                        phase = 4 * pi / lambda * dot(virt_pos, target_dir);
                        
                        % 信号 = 幅度(k) × RCS × exp(-j×相位)
                        signal = target_amplitudes(i, k) * sqrt(target_rcs) * exp(-1j * phase);
                        
                        snapshots(v, k) = snapshots(v, k) + signal;
                    end
                end
            end
            
            % 添加噪声
            if isfinite(snr_db)
                signal_power = mean(abs(snapshots(:)).^2);
                noise_power = signal_power / (10^(snr_db/10));
                noise = sqrt(noise_power/2) * (randn(size(snapshots)) + 1j * randn(size(snapshots)));
                snapshots = snapshots + noise;
            end
        end
    end
end




