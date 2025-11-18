%% 调试脚本：验证虚拟阵元排序的一致性
clear; clc;

% 简单的雷达参数
radar_params.fc = 77e9;
radar_params.c = physconst('LightSpeed');
radar_params.lambda = radar_params.c / radar_params.fc;
radar_params.fs = 20e6;
radar_params.T_chirp = 50e-6;
radar_params.slope = 100e12;
radar_params.BW = radar_params.slope * radar_params.T_chirp;
radar_params.num_samples = radar_params.T_chirp * radar_params.fs;
radar_params.range_res = radar_params.c / (2 * radar_params.BW);

% 创建一个简单的2x2阵列来测试
lambda = radar_params.lambda;
spacing = lambda / 2;

% 定义4个物理阵元位置
elements = [
    0,       0,       0;
    spacing, 0,       0;
    0,       spacing, 0;
    spacing, spacing, 0
];

% 情况1：所有阵元既是Tx也是Rx（4x4=16个虚拟元素）
% 简化：使用前2个作为Tx，前2个作为Rx（2x2=4个虚拟元素）
tx_indices = [1, 2];
rx_indices = [1, 2];

% 创建静态阵列
array_platform = ArrayPlatform(elements, tx_indices, rx_indices);
array_platform = array_platform.set_trajectory(@(t) struct('position', [0,0,0], 'orientation', [0,0,0]));

% 创建一个简单目标
target_pos = [10, 10, 10];  % 任意位置
target = Target(target_pos, [0,0,0], 1);

% 生成信号
sig_gen = SignalGenerator(radar_params, array_platform, {target});
t_axis = 0;  % 只使用一个快拍
snapshots = sig_gen.generate_snapshots(t_axis, inf);  % 无噪声

fprintf('=== 虚拟阵元排序验证 ===\n\n');
fprintf('物理阵元定义:\n');
for i = 1:size(elements, 1)
    fprintf('  元素%d: [%.4f, %.4f, %.4f]\n', i, elements(i,1), elements(i,2), elements(i,3));
end
fprintf('\n');

fprintf('Tx索引: %s\n', mat2str(tx_indices));
fprintf('Rx索引: %s\n\n', mat2str(rx_indices));

% 从ArrayPlatform获取虚拟阵元位置
virtual_pos = array_platform.get_mimo_virtual_positions(0);
fprintf('ArrayPlatform计算的虚拟阵元顺序:\n');
for i = 1:size(virtual_pos, 1)
    fprintf('  虚拟元素%d: [%.4f, %.4f, %.4f]\n', i, virtual_pos(i,1), virtual_pos(i,2), virtual_pos(i,3));
end
fprintf('\n');

% 手动计算SignalGenerator应该产生的顺序
fprintf('SignalGenerator应产生的虚拟阵元顺序（基于RDC reshape）:\n');
fprintf('  RDC维度: [samples x chirps x rx x tx]\n');
fprintf('  permute([1,2,4,3])后: [samples x chirps x tx x rx]\n');
fprintf('  reshape后虚拟元素顺序（Tx内层，Rx外层）:\n');
idx = 1;
for rx_idx = 1:length(rx_indices)
    for tx_idx = 1:length(tx_indices)
        tx_pos = elements(tx_indices(tx_idx), :);
        rx_pos = elements(rx_indices(rx_idx), :);
        virtual_pos_manual = (tx_pos + rx_pos) / 2;
        fprintf('  虚拟元素%d (Tx%d+Rx%d): [%.4f, %.4f, %.4f]\n', ...
            idx, tx_indices(tx_idx), rx_indices(rx_idx), ...
            virtual_pos_manual(1), virtual_pos_manual(2), virtual_pos_manual(3));
        idx = idx + 1;
    end
end
fprintf('\n');

% 比较两者是否一致
fprintf('=== 一致性检查 ===\n');
is_consistent = true;
idx = 1;
for rx_idx = 1:length(rx_indices)
    for tx_idx = 1:length(tx_indices)
        tx_pos = elements(tx_indices(tx_idx), :);
        rx_pos = elements(rx_indices(rx_idx), :);
        expected_pos = (tx_pos + rx_pos) / 2;
        actual_pos = virtual_pos(idx, :);
        error = norm(expected_pos - actual_pos);
        
        if error > 1e-10
            fprintf('❌ 虚拟元素%d 不一致！ 期望[%.4f,%.4f,%.4f], 实际[%.4f,%.4f,%.4f]\n', ...
                idx, expected_pos(1), expected_pos(2), expected_pos(3), ...
                actual_pos(1), actual_pos(2), actual_pos(3));
            is_consistent = false;
        else
            fprintf('✓ 虚拟元素%d 一致\n', idx);
        end
        idx = idx + 1;
    end
end

if is_consistent
    fprintf('\n✅ 虚拟阵元排序完全一致！\n');
else
    fprintf('\n❌ 虚拟阵元排序不一致，需要修复！\n');
end

