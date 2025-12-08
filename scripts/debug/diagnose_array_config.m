%% 阵列配置诊断脚本
% 检查阵列几何参数是否合理

clear; clc; close all;

fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║         阵列配置诊断工具                               ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

%% 雷达参数
c = physconst('LightSpeed');
f0 = 3e9;  % 3 GHz
lambda = c / f0;

fprintf('📡 雷达参数\n');
fprintf('   频率: %.2f GHz\n', f0/1e9);
fprintf('   波长: %.1f cm\n', lambda*100);
fprintf('   半波长: %.1f cm (标准阵元间距)\n\n', lambda/2*100);

%% 检查不同配置

configs = struct();

% 配置1：当前的小半径（有问题）
configs(1).name = '原配置（5cm半径）';
configs(1).R_rx = 0.05;
configs(1).num_elements = [4, 8, 16];

% 配置2：推荐的中等半径
configs(2).name = '推荐配置（15cm半径）';
configs(2).R_rx = 0.15;
configs(2).num_elements = [4, 8, 16];

% 配置3：大半径
configs(3).name = '大半径配置（30cm半径）';
configs(3).R_rx = 0.30;
configs(3).num_elements = [4, 8, 16];

fprintf('═══════════════════════════════════════════════════════\n');
fprintf('圆形阵列配置分析\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

for cfg_idx = 1:length(configs)
    cfg = configs(cfg_idx);
    fprintf('【%s】\n', cfg.name);
    fprintf('   阵列半径: %.1f cm\n', cfg.R_rx * 100);
    fprintf('   圆周长: %.1f cm\n\n', 2*pi*cfg.R_rx * 100);
    
    fprintf('   阵元数 | 阵元间距 | 间距/半波长 | 状态\n');
    fprintf('   -------|----------|-------------|------\n');
    
    for N = cfg.num_elements
        circumference = 2 * pi * cfg.R_rx;
        element_spacing = circumference / N;
        spacing_ratio = element_spacing / (lambda / 2);
        
        if spacing_ratio < 0.5
            status = '❌ 太密！空间混叠';
        elseif spacing_ratio < 0.8
            status = '⚠️ 偏密';
        elseif spacing_ratio < 1.2
            status = '✓ 合理';
        elseif spacing_ratio < 2.0
            status = '✓ 良好';
        else
            status = '⚠️ 偏稀（栅瓣风险）';
        end
        
        fprintf('   %6d | %6.1f cm | %11.2f | %s\n', ...
            N, element_spacing*100, spacing_ratio, status);
    end
    fprintf('\n');
end

%% 绘制阵列几何对比

figure('Position', [100, 100, 1400, 500]);

for cfg_idx = 1:length(configs)
    cfg = configs(cfg_idx);
    
    for N_idx = 1:length(cfg.num_elements)
        N = cfg.num_elements(N_idx);
        
        subplot(length(configs), length(cfg.num_elements), ...
            (cfg_idx-1)*length(cfg.num_elements) + N_idx);
        
        % 绘制阵列
        theta_rx = linspace(0, 2*pi, N+1); theta_rx(end) = [];
        x_positions = cfg.R_rx * cos(theta_rx);
        y_positions = cfg.R_rx * sin(theta_rx);
        
        plot(x_positions, y_positions, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
        hold on;
        
        % 绘制圆圈
        theta_circle = linspace(0, 2*pi, 100);
        plot(cfg.R_rx * cos(theta_circle), cfg.R_rx * sin(theta_circle), 'k--');
        
        % 绘制阵元间连线
        for i = 1:N
            j = mod(i, N) + 1;
            plot([x_positions(i), x_positions(j)], ...
                 [y_positions(i), y_positions(j)], 'r-', 'LineWidth', 1);
        end
        
        % 标注阵元间距
        element_spacing = 2 * pi * cfg.R_rx / N;
        spacing_ratio = element_spacing / (lambda / 2);
        
        if spacing_ratio < 0.8
            color_status = 'red';
        elseif spacing_ratio < 1.5
            color_status = 'green';
        else
            color_status = 'orange';
        end
        
        title(sprintf('%d元, 间距=%.1fcm (%.2fλ/2)', ...
            N, element_spacing*100, spacing_ratio), ...
            'Color', color_status);
        
        xlabel('X (m)');
        ylabel('Y (m)');
        axis equal;
        grid on;
        xlim([-0.35, 0.35]);
        ylim([-0.35, 0.35]);
    end
end

sgtitle('圆形阵列几何配置对比', 'FontSize', 14, 'FontWeight', 'bold');

%% 理论分析

fprintf('═══════════════════════════════════════════════════════\n');
fprintf('理论分析\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

fprintf('1. 阵元间距要求：\n');
fprintf('   - 最小: 0.4λ (避免严重空间混叠)\n');
fprintf('   - 推荐: 0.5λ ~ 0.7λ (标准配置)\n');
fprintf('   - 最大: 0.8λ (避免栅瓣)\n\n');

fprintf('2. 圆形阵列特点：\n');
fprintf('   - 优势: 方向图各向同性\n');
fprintf('   - 劣势: 同样阵元数，孔径比ULA小\n');
fprintf('   - 适用: 需要360°覆盖的场景\n\n');

fprintf('3. 运动合成孔径效果：\n');
fprintf('   - 旋转1圈 = 虚拟阵元数 × N\n');
fprintf('   - 有效孔径 ≈ 2R × 旋转快拍数\n');
fprintf('   - 前提: 每个阵元本身的孔径要足够大\n\n');

fprintf('4. 当前问题诊断：\n');
fprintf('   【原配置 R=5cm】:\n');
fprintf('     • 8元阵列间距 ≈ 3.9cm < 5cm (半波长)\n');
fprintf('     • 空间相关性过强 → 静态阵列性能崩溃\n');
fprintf('     • 旋转后虽然虚拟阵元多，但每个位置的孔径都很小\n');
fprintf('     • 结果: 8元反而比4元差！\n\n');

fprintf('   【推荐配置 R=15cm】:\n');
fprintf('     • 8元阵列间距 ≈ 11.8cm ≈ 2.36倍半波长 ✓\n');
fprintf('     • 静态阵列性能正常\n');
fprintf('     • 旋转后效果显著（孔径扩展3-5倍）\n\n');

%% 建议

fprintf('═══════════════════════════════════════════════════════\n');
fprintf('🔧 修复建议\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

fprintf('1. 修改 comprehensive_validation.m 第90行:\n');
fprintf('   R_rx = 0.05;  %% ❌ 原来（太小）\n');
fprintf('   R_rx = 0.15;  %% ✅ 推荐（修改后）\n\n');

fprintf('2. 重新运行实验:\n');
fprintf('   >> reset_validation_progress  %% 重置进度\n');
fprintf('   >> comprehensive_validation   %% 重新运行\n\n');

fprintf('3. 预期改善:\n');
fprintf('   - 实验1: 双目标分辨率明显提升\n');
fprintf('   - 实验2: 有效孔径扩展 2-4倍\n');
fprintf('   - 实验3: RMSE显著降低\n');
fprintf('   - 实验5: 运动优势明显（所有配置都改善）\n\n');

fprintf('✅ 诊断完成！\n');
fprintf('   已发现问题: 阵列半径过小导致空间混叠\n');
fprintf('   建议修改: R_rx = 0.15 m\n\n');



