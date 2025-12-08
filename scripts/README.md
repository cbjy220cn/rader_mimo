# 运动阵列DOA估计系统 - 脚本说明

## 📂 目录结构

```
scripts/
├── asset/                    # 核心类库（必需）
├── validation_results/       # 实验结果输出
├── debug/                    # 调试脚本
├── 参考/                     # 参考代码
├── MIMO_FMCW/               # FMCW参考实现
├── old_need_to_del/         # 待清理旧代码
└── *.m                      # 实验脚本
```

---

## 🔧 核心类库 (asset/)

| 文件 | 说明 | 使用频率 |
|------|------|---------|
| `DoaEstimatorSynthetic.m` | **主DOA估计器**，合成虚拟阵列MUSIC | ⭐⭐⭐ |
| `SignalGeneratorSimple.m` | **简化信号生成**，跳过FMCW直接计算空间相位 | ⭐⭐⭐ |
| `ArrayPlatform.m` | 阵列平台，管理位置和轨迹 | ⭐⭐⭐ |
| `Target.m` | 目标定义 | ⭐⭐⭐ |
| `find_peaks_cfar.m` | CA-CFAR峰值检测 | ⭐⭐ |
| `smart_doa_search.m` | 智能两步搜索（加速） | ⭐⭐ |
| `DoaEstimator.m` | 相干GMUSIC（静态阵列用） | ⭐ |
| `DoaEstimatorIncoherent.m` | 非相干MUSIC（有rank-1问题） | ⚠️ |
| `SignalGenerator.m` | 完整FMCW信号生成（复杂） | ⭐ |

---

## 📋 实验脚本分类

### ⭐⭐⭐ 主要实验（论文用）

| 脚本 | 功能 | 运行命令 |
|------|------|---------|
| `comprehensive_motion_array_test.m` | **综合性能测试**：8阵列×5运动×8SNR | `run('comprehensive_motion_array_test.m')` |
| `experiment_vibration_robustness.m` | **抗震动测试**：评估平台震动影响 | `run('experiment_vibration_robustness.m')` |
| `experiment_parallel_processing_v2.m` | **实时性测试**：滑动窗口优化版 | `run('experiment_parallel_processing_v2.m')` |
| `experiment_dual_target_resolution.m` | **双目标分辨**：分辨率测试 | `run('experiment_dual_target_resolution.m')` |

### ⭐⭐ 辅助脚本

| 脚本 | 功能 | 备注 |
|------|------|------|
| `generate_paper_figures.m` | 生成论文级图表 | 可独立运行 |
| `experiment_parallel_processing.m` | 实时性测试v1 | 已被v2替代 |

### ⭐ 旧版实验（参考）

| 脚本 | 功能 | 状态 |
|------|------|------|
| `comprehensive_validation_FIXED.m` | 旧版综合验证（2D搜索） | 被主测试替代 |
| `comprehensive_validation.m` | 更早版本 | 参考 |
| `comprehensive_experiment.m` | 阵列×轨迹优化 | 功能整合到主测试 |
| `motion_vs_static_comparison.m` | 静态vs运动对比 | 功能整合到主测试 |

### 🔧 工具脚本

| 脚本 | 功能 | 使用场景 |
|------|------|---------|
| `check_validation_progress.m` | 查看断点续传进度 | 长时间实验中断后 |
| `reset_validation_progress.m` | 重置实验进度 | 需要重新开始时 |
| `verify_tools_reliability.m` | 验证工具类正确性 | 代码修改后 |
| `verify_rotation.m` | 验证旋转功能 | 调试用 |

### 🧪 快速测试脚本

| 脚本 | 功能 | 运行时间 |
|------|------|---------|
| `quick_validation_test.m` | 快速功能验证 | ~5分钟 |
| `quick_validation_experiment.m` | 快速性能验证 | ~10分钟 |
| `quick_test_motion_modes.m` | 运动模式快速对比 | ~5分钟 |

### 📐 单项测试脚本 (run_*.m)

| 脚本 | 功能 |
|------|------|
| `run_trajectory_array_experiment.m` | 轨迹-阵列组合测试 |
| `run_rotation_vs_circular_test.m` | 旋转vs圆周运动对比 |
| `run_rotation_experiment.m` | 旋转实验 |
| `run_baseline_ura_test.m` | URA基线测试 |
| `run_rmse_vs_snr_test.m` | RMSE-SNR曲线 |
| `run_resolution_test.m` | 分辨率测试 |
| `run_verification.m` | 通用验证 |

---

## 🚀 快速开始

### 1. 运行主实验
```matlab
cd scripts
run('comprehensive_motion_array_test.m')
% 输出: validation_results/comprehensive_motion_array_test_<时间戳>/
```

### 2. 运行补充实验
```matlab
run('experiment_vibration_robustness.m')      % 抗震动
run('experiment_parallel_processing_v2.m')    % 实时性
run('experiment_dual_target_resolution.m')    % 双目标
```

### 3. 自定义实验
```matlab
addpath('asset');

% 创建阵列
elements = [...];  % 阵元位置
array = ArrayPlatform(elements, 1, 1:8);
array.set_trajectory(@(t) struct('position', [v*t,0,0], 'orientation', [0,0,0]));

% 生成信号
targets = {Target([500,300,0], [0,0,0], 1)};
sig_gen = SignalGeneratorSimple(radar_params, array, targets);
snapshots = sig_gen.generate_snapshots(t_axis, snr_db);

% DOA估计
estimator = DoaEstimatorSynthetic(array, radar_params);
[spectrum, peaks, info] = estimator.estimate(snapshots, t_axis, search_grid, 1);
```

---

## 📁 debug/ 目录说明

调试和单元测试脚本，用于问题排查：

| 类别 | 脚本 |
|------|------|
| **诊断** | `diagnose_*.m` - 问题诊断 |
| **测试** | `test_*.m` - 功能测试 |
| **调试** | `debug_*.m` - 调试辅助 |

常用：
- `test_incoherent_vs_coherent.m` - 相干vs非相干对比
- `test_complex_motion_modes.m` - 复杂运动测试
- `diagnose_steering_vector.m` - 导向矢量诊断

---

## 📚 文档

| 文档 | 说明 |
|------|------|
| `技术文档_运动阵列DOA系统.md` | **主技术文档**，包含原理、参数、结果解读 |
| `asset/README.md` | 核心类简介 |

---

## ⚠️ 注意事项

1. **运行前**: 确保 `addpath('asset')` 
2. **长时间实验**: 支持断点续传，Ctrl+C中断后重新运行即可继续
3. **结果位置**: 所有输出在 `validation_results/` 下带时间戳的文件夹中
4. **中文支持**: 图表标题使用中文，需确保MATLAB字体设置正确

---

*最后更新: 2025-12-07*
