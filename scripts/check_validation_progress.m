%% 检查验证进度的辅助脚本
% 用于查看当前验证的进度状态

clear; clc;
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('验证进度查看工具\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

output_dir = 'validation_results';
progress_file = fullfile(output_dir, 'progress.mat');
exp3_progress_file = fullfile(output_dir, 'exp3_rmse_progress.mat');

%% 主进度文件
if exist(progress_file, 'file')
    load(progress_file);
    fprintf('📊 总体进度:\n');
    fprintf('   最后完成的实验: %d\n', progress.last_completed_experiment);
    
    if progress.last_completed_experiment >= 1
        fprintf('   ✅ 实验1: 角度分辨率测试 - 已完成\n');
    else
        fprintf('   ⏸️  实验1: 角度分辨率测试 - 未开始\n');
    end
    
    if progress.last_completed_experiment >= 2
        fprintf('   ✅ 实验2: 有效孔径扩展 - 已完成\n');
    else
        fprintf('   ⏸️  实验2: 有效孔径扩展 - 未开始\n');
    end
    
    if progress.last_completed_experiment >= 3
        fprintf('   ✅ 实验3: RMSE vs SNR - 已完成\n');
    else
        fprintf('   ⏸️  实验3: RMSE vs SNR - 未开始或进行中\n');
    end
    
    if progress.last_completed_experiment >= 4
        fprintf('   ✅ 图表生成 - 已完成\n');
        if isfield(progress, 'completion_time')
            fprintf('   完成时间: %s\n', progress.completion_time);
        end
    else
        fprintf('   ⏸️  图表生成 - 未完成\n');
    end
    fprintf('\n');
else
    fprintf('❌ 未找到进度文件，验证尚未开始\n\n');
end

%% 实验3详细进度
if exist(exp3_progress_file, 'file')
    load(exp3_progress_file);
    fprintf('🔬 实验3详细进度:\n');
    fprintf('   完成的SNR点: %d / %d\n', exp3_progress.last_snr_idx, length(exp3_progress.snr_range));
    fprintf('   SNR范围: [%s] dB\n', sprintf('%+d ', exp3_progress.snr_range));
    fprintf('   已完成的SNR点: [%s] dB\n', sprintf('%+d ', exp3_progress.snr_range(1:exp3_progress.last_snr_idx)));
    
    if exp3_progress.last_snr_idx < length(exp3_progress.snr_range)
        fprintf('   ⏭️  下一个: SNR = %+d dB\n', exp3_progress.snr_range(exp3_progress.last_snr_idx + 1));
    end
    
    % 显示已完成的RMSE结果
    fprintf('\n   当前RMSE结果:\n');
    for i = 1:exp3_progress.last_snr_idx
        fprintf('      SNR=%+3d dB: 静态=%.2f°, 旋转=%.2f°\n', ...
            exp3_progress.snr_range(i), ...
            exp3_progress.rmse_static(i), ...
            exp3_progress.rmse_rotating(i));
    end
    fprintf('\n');
else
    fprintf('📝 实验3尚未开始或无中间进度\n\n');
end

%% 保存的文件列表
fprintf('💾 已保存的结果文件:\n');
if exist(output_dir, 'dir')
    files = dir(fullfile(output_dir, '*.mat'));
    if ~isempty(files)
        for i = 1:length(files)
            file_info = dir(fullfile(output_dir, files(i).name));
            size_kb = file_info.bytes / 1024;
            fprintf('   - %s (%.1f KB)\n', files(i).name, size_kb);
        end
    else
        fprintf('   (无)\n');
    end
    
    % 图片文件
    fprintf('\n🖼️  已生成的图像:\n');
    images = dir(fullfile(output_dir, '*.png'));
    if ~isempty(images)
        for i = 1:length(images)
            fprintf('   - %s\n', images(i).name);
        end
    else
        fprintf('   (无)\n');
    end
else
    fprintf('   ❌ 输出目录不存在\n');
end

fprintf('\n');
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('操作选项:\n');
fprintf('  1. 继续运行: 直接执行 comprehensive_validation\n');
fprintf('  2. 重新开始: 删除 %s\n', progress_file);
fprintf('  3. 清理实验3进度: 删除 %s\n', exp3_progress_file);
fprintf('  4. 清理所有: 删除 %s 文件夹\n', output_dir);
fprintf('═══════════════════════════════════════════════════════\n\n');

