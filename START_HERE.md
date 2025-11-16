# 🚀 快速开始指南

## 步骤1：快速验证（5分钟）⭐

**目的**：验证代码是否正确，功能是否正常

```matlab
clear; clc; close all
quick_validation_test
```

**检查项**：
- ✅ 智能搜索：单目标DOA估计
- ✅ CA-CFAR：双目标峰值分离
- ✅ 多阵列：ULA、矩形阵测试
- ✅ 数据保存：断点续传功能

**如果所有测试通过**：继续步骤2  
**如果有测试失败**：检查错误信息或联系支持

---

## 步骤2：完整实验（20-40分钟）⭐⭐⭐

```matlab
clear classes; clear all; clc
comprehensive_validation
```

### 实验内容

| 实验 | 内容 | 预计时间 | 可中断 |
|-----|------|---------|--------|
| **1** | 双目标角度分辨率（CA-CFAR） | 5-8分钟 | ✓ |
| **2** | 有效孔径扩展（3种阵元数） | 3-5分钟 | ✓ |
| **3** | RMSE vs SNR（6个SNR点） | 6-12分钟 | ✓✓ |
| **4** | 最优轨迹-阵列组合（12组） | 5-10分钟 | ✓ |
| **绘图** | 生成10张对比图表 | 1-2分钟 | - |

### 输出结果

```
validation_results/
├── exp1_resolution_results.mat       # 实验1数据
├── exp2_aperture_results.mat         # 实验2数据  
├── exp3_rmse_results.mat             # 实验3数据
├── exp3_rmse_progress.mat            # 实验3进度（每个SNR点）
├── exp4_trajectory_array_results.mat # 实验4数据
├── progress.mat                      # 总进度
├── progress_backup.mat               # 进度备份
├── 1A_resolution_normalized.png      # 图1A
├── 1B_resolution_dB.png             # 图1B
├── 1C_resolution_1D_slices.png      # 图1C ⭐
├── 2_aperture_extension.png          # 图2
├── 2B_beam_pattern_slices.png       # 图2B
├── 3_rmse_vs_snr.png                # 图3
├── 4_performance_summary.png         # 图4
├── 5_trajectory_array_analysis.png   # 图5
└── README_visualization.md           # 说明文档
```

---

## 断点续传功能 🔄

### 特性

- ✅ **自动保存**：每完成一个实验/SNR点立即保存
- ✅ **双重备份**：progress.mat + progress_backup.mat
- ✅ **时间戳**：记录每次保存时间
- ✅ **容错机制**：文件损坏自动恢复
- ✅ **进程安全**：即使MATLAB崩溃也能恢复

### 中断恢复

**中断方式**：
- Ctrl+C 手动中断
- MATLAB进程崩溃
- 电脑断电/重启
- 远程连接断开

**恢复运行**：
```matlab
comprehensive_validation
```

系统会自动：
1. 检测进度文件
2. 显示上次运行时间和完成状态
3. 询问是否继续（默认继续）
4. 跳过已完成部分，从断点继续

**示例输出**：
```
🔄 检测到之前的进度文件
   上次运行: 16-Nov-2025 10:30:45
   已完成: 实验2
   是否继续？ (1=继续, 0=重新开始) [1]: 
```

---

## 进度管理工具

### 查看进度

```matlab
check_validation_progress
```

显示：
- 总体进度（完成了哪些实验）
- 实验3详细进度（完成了哪些SNR点）
- 已保存文件列表
- 操作建议

### 重置进度

```matlab
reset_validation_progress
```

选项：
- 选项1：仅重置实验3
- 选项2：重置所有进度
- 选项3：删除所有文件

---

## 配置选项

### 禁用实验4（加快速度）

编辑 `comprehensive_validation.m` 第63行：

```matlab
RUN_TRAJECTORY_ARRAY_TEST = false;  % 禁用实验4，节省5-10分钟
```

### 禁用CA-CFAR（回退传统方法）

编辑 `comprehensive_validation.m` 第53行：

```matlab
USE_CFAR_EXP1 = false;  % 实验1使用传统峰值检测
```

### 调整快拍数（速度vs精度）

编辑 `comprehensive_validation.m` 第66行：

```matlab
num_snapshots_base = 32;  % 从64改为32，速度快2倍，精度略降
```

### 调整蒙特卡洛试验次数

编辑 `comprehensive_validation.m` 第60行：

```matlab
num_trials_mc = 10;  % 从20改为10，实验3快2倍
```

---

## 常见问题 ❓

### Q1: GPU相关错误？

**A**: 在 `DoaEstimatorIncoherent.m` 第15行禁用GPU：
```matlab
obj.use_gpu = false;
```

### Q2: 内存不足？

**A**: 
1. 减少快拍数（第66行）
2. 降低搜索网格分辨率（第35行）

### Q3: 实验3太慢？

**A**: 
- 已支持断点续跑，每个SNR点自动保存
- 可以每天跑1-2个SNR点
- 或减少蒙特卡洛次数

### Q4: 进度文件损坏？

**A**: 自动使用备份文件 `progress_backup.mat`

### Q5: 如何只运行部分实验？

**A**: 修改 `progress.mat`：
```matlab
load('validation_results/progress.mat');
progress.last_completed_experiment = 1;  % 从实验2开始
save('validation_results/progress.mat', 'progress');
```

---

## 远程服务器运行 🖥️

### Linux/Mac

```bash
# 后台运行，记录日志
nohup matlab -batch "comprehensive_validation" > validation.log 2>&1 &

# 查看实时输出
tail -f validation.log

# 查看进度（另一个SSH会话）
matlab -batch "check_validation_progress"
```

### Windows PowerShell

```powershell
# 后台任务
Start-Job -ScriptBlock { matlab -batch "comprehensive_validation" }

# 查看输出
Receive-Job -Id 1

# 或使用 diary
matlab -batch "diary('log.txt'); diary on; comprehensive_validation; diary off"
```

---

## 性能优化 ⚡

### 当前优化

- ✅ 智能搜索：速度提升30-50倍
- ✅ GPU加速：自动检测并启用
- ✅ 断点续跑：零时间损失
- ✅ 实时保存：防止数据丢失

### 预计时间（默认配置）

- **最快**：禁用实验4，32快拍，10次试验 → 10-15分钟
- **标准**：启用实验4，64快拍，20次试验 → 20-40分钟
- **完整**：保持默认，所有功能 → 30-50分钟

---

## 技术支持 📞

### 检查日志

```matlab
% 启用详细日志
diary('validation_full.log');
diary on;
comprehensive_validation
diary off;
```

### 报告问题

提供以下信息：
1. MATLAB版本：`version`
2. 错误消息：完整的错误堆栈
3. 进度状态：`check_validation_progress`
4. 系统信息：OS、内存、GPU

---

## 🎉 现在开始！

**推荐流程**：

```matlab
% 步骤1: 快速验证（5分钟）
quick_validation_test

% 如果通过，步骤2: 完整实验（20-40分钟）
clear classes; clear all; clc
comprehensive_validation

% 随时查看进度
check_validation_progress
```

**祝实验顺利！** 🚀

---

**提示**：
- 实验运行期间可以随时Ctrl+C中断
- 所有进度都会自动保存
- 重新运行时自动继续
- 最坏情况：只丢失当前正在计算的SNR点

**即使MATLAB进程被杀死，数据也是安全的！** ✅

