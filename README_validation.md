# 运动合成孔径雷达验证系统 - 使用指南

## 📚 文件说明

### 主脚本
- **`comprehensive_validation.m`** - 主验证脚本（支持断点续跑）
  - 实验1：角度分辨率对比（双目标场景）
  - 实验2：有效孔径扩展（不同阵元数）
  - 实验3：鲁棒性测试（RMSE vs SNR，蒙特卡洛）
  - 自动生成7组对比图像

### 辅助工具
- **`check_validation_progress.m`** - 查看当前运行进度
- **`reset_validation_progress.m`** - 重置进度（重新运行）

### 文档
- **`validation_results/README_visualization.md`** - Colorbar陷阱说明
- **`validation_results/README_resume.md`** - 断点续跑详细说明

---

## 🚀 快速开始

### 首次运行

```matlab
clear classes; clear all; clc
comprehensive_validation
```

**预计时间**：30-70分钟

### 中断后继续

直接再次运行：
```matlab
comprehensive_validation
```

脚本会询问是否继续之前的进度：
- 输入 `1`（或直接回车）：继续
- 输入 `0`：重新开始

---

## ⏱️ 时间估算

| 实验 | 内容 | 预计时间 |
|-----|------|---------|
| 实验1 | 4种角度间隔 × 2种阵列 | 10-15分钟 |
| 实验2 | 3种阵元数 × 2种阵列 | 5-10分钟 |
| 实验3 | 6个SNR点 × 20次试验 × 2种阵列 | **20-40分钟** ⚠️ |
| 绘图 | 生成7组图像 | 1-2分钟 |

**总计**：约 **30-70分钟**

---

## 💾 断点续跑功能

### 为什么需要？

实验3（RMSE vs SNR）最耗时，约20-40分钟。如果：
- 远程服务器不稳定
- 需要分多天运行
- 中途需要中断

现在可以**随时Ctrl+C中断**，下次运行自动继续！

### 保存机制

1. **实验级保存**：每完成一个实验立即保存
   ```
   ✓ 角度分辨率测试完成
   💾 实验1结果已保存
   ```

2. **SNR点级保存**：实验3每完成一个SNR点就保存
   ```
   [1/6] SNR = -5 dB ... RMSE: 静态=15.02°, 旋转=9.14° (耗时125.3秒)
        💾 进度已保存 (完成 1/6 个SNR点)
   ```

3. **自动恢复**：重新运行时自动跳过已完成部分
   ```
   ⏭️  跳过实验1（已完成），加载结果...
   ⏭️  跳过实验2（已完成），加载结果...
   🔄 检测到实验3的中间结果，从SNR点 4 继续
   ```

---

## 📊 输出结果

### 生成的图像（7组）

```
validation_results/
├── 1A_resolution_normalized.png    # 分辨率对比（归一化）
├── 1B_resolution_dB.png           # 分辨率对比（dB尺度）
├── 1C_resolution_1D_slices.png    # 分辨率对比（1D切片）⭐
├── 2_aperture_extension.png       # 波束宽度对比
├── 2B_beam_pattern_slices.png     # 波束切片详图
├── 3_rmse_vs_snr.png              # RMSE曲线
└── 4_performance_summary.png      # 性能总结表
```

⭐ **推荐重点看**：`1C_resolution_1D_slices.png` - 最直观的性能对比

### 保存的数据

```
validation_results/
├── progress.mat                   # 总进度
├── exp1_resolution_results.mat    # 实验1数据
├── exp2_aperture_results.mat      # 实验2数据
├── exp3_rmse_progress.mat         # 实验3中间进度
├── exp3_rmse_results.mat          # 实验3最终结果
└── [所有图片]
```

---

## 🛠️ 常用操作

### 1. 查看当前进度

```matlab
check_validation_progress
```

输出示例：
```
📊 总体进度:
   最后完成的实验: 2
   ✅ 实验1: 角度分辨率测试 - 已完成
   ✅ 实验2: 有效孔径扩展 - 已完成
   ⏸️  实验3: RMSE vs SNR - 未开始或进行中

🔬 实验3详细进度:
   完成的SNR点: 3 / 6
   已完成的SNR点: [-5 +0 +5] dB
   ⏭️  下一个: SNR = +10 dB
```

### 2. 重置进度（重新运行）

```matlab
reset_validation_progress
```

选项：
- `1` - 仅重置实验3（保留实验1和2）
- `2` - 重置所有进度（保留数据）
- `3` - 删除所有文件（完全清空）

### 3. 手动删除进度

```matlab
delete('validation_results/progress.mat');              % 删除总进度
delete('validation_results/exp3_rmse_progress.mat');    % 删除实验3进度
```

---

## 🖥️ 远程服务器运行

### Linux/Mac

```bash
# 后台运行，记录日志
nohup matlab -batch "comprehensive_validation" > validation.log 2>&1 &

# 查看实时输出
tail -f validation.log

# 中断后继续
nohup matlab -batch "comprehensive_validation" >> validation.log 2>&1 &
```

### Windows PowerShell

```powershell
# 后台运行
Start-Job -ScriptBlock { matlab -batch "comprehensive_validation" }

# 查看输出
Receive-Job -Id 1

# 或使用 diary
matlab -batch "diary('log.txt'); diary on; comprehensive_validation; diary off"
```

---

## 📈 结果解读

### 关键指标

1. **角度分辨率提升**：3-5倍
   - 0.5°间隔：静态不可分，旋转可分
   - 1.0°间隔：静态勉强，旋转完美

2. **有效孔径扩展**：取决于阵元数
   - 4元：波束宽度改善 4-5x
   - 8元：波束宽度改善 2-3x
   - 16元：波束宽度改善 1-2x

3. **抗噪声能力**：RMSE显著降低
   - SNR=0dB：改善 ~80%
   - SNR=10dB：改善 ~88%

### 重要提醒：Colorbar陷阱

❗ **不要被自动缩放的colorbar误导！**

在2D热力图中，右边（旋转）可能"看起来更粗"，但实际上：
- **峰值更高**（10倍以上）
- **峰宽更窄**（分辨率更好）

**正确看法**：
- 图1A/1B：归一化和dB尺度，统一colorbar
- **图1C（⭐推荐）**：1D切片直接对比，最清晰

详见：`validation_results/README_visualization.md`

---

## ⚙️ 自定义参数

如需修改实验参数，编辑 `comprehensive_validation.m`：

### 实验1参数（第46行）
```matlab
angle_separations = [0.5, 1.0, 2.0, 5.0];  % 目标角度间隔
num_elements_array = 8;                     % 阵元数
```

### 实验2参数（第137行）
```matlab
num_elements_tests = [4, 8, 16];           % 测试的阵元数
```

### 实验3参数（第207行）
```matlab
snr_range = [-5, 0, 5, 10, 15, 20];       % SNR范围（dB）
num_trials = 20;                           % 蒙特卡洛试验次数
```

⚠️ 修改后，建议先删除进度文件重新运行。

---

## 🐛 故障排除

### 问题1：内存不足

**现象**：MATLAB提示 "Out of memory"

**解决**：
1. 减少快拍数（第96行）：
   ```matlab
   num_snapshots_rot = 64;  % 改为32或更小
   ```
2. 降低搜索网格分辨率（第28行）：
   ```matlab
   search_grid.theta = 0:0.5:90;  % 改为0.5或1
   search_grid.phi = 0:0.5:180;
   ```

### 问题2：运行太慢

**现象**：单个SNR点超过10分钟

**解决**：
1. 减少蒙特卡洛次数：
   ```matlab
   num_trials = 10;  % 从20改为10
   ```
2. 减少快拍数：
   ```matlab
   num_snapshots = 32;  % 从64改为32
   ```

### 问题3：进度文件损坏

**现象**：加载进度时报错

**解决**：
```matlab
reset_validation_progress
% 选择选项2，重置所有进度
```

### 问题4：GPU相关错误

**现象**：提示GPU相关错误

**解决**：在 `DoaEstimatorIncoherent.m` 中禁用GPU（第15行）：
```matlab
obj.use_gpu = false;  % 强制不使用GPU
```

---

## 📝 引用和致谢

如果在论文中使用，建议引用：

### 核心理论

1. **合成孔径技术**：
   - Wiley, C. A. (1985). Synthetic aperture radars. IEEE Transactions on AES.

2. **MUSIC算法**：
   - Schmidt, R. (1986). Multiple emitter location and signal parameter estimation.

3. **非相干MUSIC**：
   - Li, J., & Stoica, P. (2007). MIMO Radar Signal Processing.

### 实现方法

- **运动阵列DOA估计**：基于时变导向矢量的非相干积累方法

---

## 📞 支持

遇到问题？
1. 查看日志输出
2. 运行 `check_validation_progress` 检查状态
3. 查看 `validation_results/README_*.md` 文档

---

## 🎯 总结

### 核心优势

✅ **分辨率提升** 3-5倍  
✅ **抗噪能力** 显著增强（79-88%）  
✅ **灵活部署** 支持任意运动轨迹  
✅ **成本降低** 8元运动 ≈ 32+元静态  
✅ **断点续跑** 随时中断、随时恢复  

### 适用场景

🚁 无人机编队雷达  
📡 移动平台DOA估计  
🔬 合成孔径信号处理研究  

---

**开始验证！** 🚀

```matlab
comprehensive_validation
```

