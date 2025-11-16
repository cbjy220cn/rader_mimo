%% 重置验证进度的辅助脚本
% 用于清理进度文件，从头开始运行

clear; clc;
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('验证进度重置工具\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

output_dir = 'validation_results';
progress_file = fullfile(output_dir, 'progress.mat');
exp3_progress_file = fullfile(output_dir, 'exp3_rmse_progress.mat');

fprintf('⚠️  警告: 此操作将删除所有中间进度！\n\n');
fprintf('请选择重置选项:\n');
fprintf('  1 - 仅重置实验3的进度（保留实验1和2）\n');
fprintf('  2 - 重置所有实验进度（保留数据文件）\n');
fprintf('  3 - 删除所有文件（包括结果图片）\n');
fprintf('  0 - 取消\n\n');

choice = input('请输入选项 [0]: ', 's');
if isempty(choice)
    choice = '0';
end

switch choice
    case '1'
        if exist(exp3_progress_file, 'file')
            delete(exp3_progress_file);
            fprintf('✓ 已删除实验3进度文件\n');
            % 同时更新主进度
            if exist(progress_file, 'file')
                load(progress_file);
                if progress.last_completed_experiment >= 3
                    progress.last_completed_experiment = 2;
                    save(progress_file, 'progress');
                    fprintf('✓ 已更新主进度（退回到实验2）\n');
                end
            end
        else
            fprintf('❌ 实验3进度文件不存在\n');
        end
        
    case '2'
        if exist(progress_file, 'file')
            delete(progress_file);
            fprintf('✓ 已删除主进度文件\n');
        end
        if exist(exp3_progress_file, 'file')
            delete(exp3_progress_file);
            fprintf('✓ 已删除实验3进度文件\n');
        end
        fprintf('✓ 所有进度已重置（数据文件保留）\n');
        
    case '3'
        if exist(output_dir, 'dir')
            % 询问确认
            confirm = input(sprintf('确认删除整个 %s 文件夹？ (yes/no) [no]: ', output_dir), 's');
            if strcmpi(confirm, 'yes')
                rmdir(output_dir, 's');
                fprintf('✓ 已删除 %s 文件夹及所有内容\n', output_dir);
            else
                fprintf('❌ 操作已取消\n');
            end
        else
            fprintf('❌ 目录不存在\n');
        end
        
    case '0'
        fprintf('❌ 操作已取消\n');
        
    otherwise
        fprintf('❌ 无效选项\n');
end

fprintf('\n═══════════════════════════════════════════════════════\n\n');

