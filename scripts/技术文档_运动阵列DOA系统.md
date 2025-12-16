# 运动阵列DOA估计系统 - 技术文档

**版本**: v5.0  
**更新日期**: 2025-12-10  
**适用代码**: `comprehensive_motion_array_test.m` 及 `asset/` 目录

---

## 一、系统概述

### 1.1 研究目标

利用阵列平台的**运动**形成合成孔径，扩展有效孔径，提升DOA估计精度和角度分辨率。

### 1.2 FMCW雷达系统背景

本系统基于**FMCW（调频连续波）雷达**设计，关键时间参数说明：

| 参数 | 典型值 | 说明 |
|------|--------|------|
| **Chirp周期** | **50 ms** | 一个完整调频周期，包含上升沿和处理时间 |
| 调频带宽 | 200 MHz | 决定距离分辨率 |
| 采样率 | 10 MHz | ADC采样率 |
| 快拍间隔 | 50 ms | 每个Chirp产生一个DOA快拍 |

**50ms Chirp周期的物理意义**：
- FMCW雷达需要完成：调频发射 → 接收混频 → FFT处理 → 距离检测
- 50ms是中等PRF（脉冲重复频率=20Hz）的典型配置
- 适用于中低速目标跟踪（最大不模糊速度 = λ/(4×T_chirp) ≈ 0.5 m/s）
- 平台运动速度5m/s时，每个Chirp期间平移0.25m

### 1.2 核心发现

| 发现 | 结论 |
|------|------|
| 纯旋转 | ❌ **无法扩展孔径**（最大基线不变） |
| 平移运动 | ✅ **有效扩展孔径**（合成孔径=物理+平移距离） |
| 单快拍问题 | ⚠️ 合成阵列协方差矩阵秩-1，标准MUSIC失效 |
| **时间平滑MUSIC** | ✅ **最优方案**（解决秩-1问题，恢复超分辨能力） |

### 1.3 v4.0 更新内容

- ✅ **时间平滑MUSIC**：解决合成孔径单快拍秩-1问题
- ✅ **2D DOA估计**：支持θ(俯仰)+φ(方位)联合估计
- ✅ **四层智能搜索**：0.01°精度，显著加速
- ✅ **细网格主瓣测量**：准确测量窄主瓣宽度
- ✅ **对数坐标图表**：更直观展示性能差异

### 1.4 代码架构

```
scripts/
├── asset/                              # 核心类
│   ├── ArrayPlatform.m                 # 阵列平台
│   ├── DoaEstimatorSynthetic.m         # DOA估计器 ⭐ (时间平滑MUSIC)
│   ├── SignalGeneratorSimple.m         # 信号生成 (相位一致性修复)
│   ├── Target.m                        # 目标定义
│   ├── find_peaks_cfar.m               # CFAR峰值检测
│   └── smart_doa_search.m              # 智能搜索
│
├── 主实验脚本
│   ├── comprehensive_motion_array_test.m  # 综合测试 ⭐ v2.3
│   ├── experiment_parallel_processing_v2.m # 实时性测试 v3.0 (2D)
│   ├── experiment_vibration_robustness.m   # 抗震动测试 v3.0 (2D)
│   ├── experiment_dual_target_resolution.m # 双目标分辨 v3.0 (2D)
│   └── experiment_dual_target_advantage.m  # 双目标优势展示 v2.0
│
├── 后处理脚本
│   └── generate_compressed_figures.m      # 生成对数坐标图表
│
├── validation_results/                 # 实验结果
└── 技术文档_运动阵列DOA系统.md         # 本文档
```

---

## 二、算法原理

### 2.1 核心问题：合成孔径的秩-1困境

#### 问题描述

合成虚拟阵列方法将 M阵元×K快拍 展开为 M×K 个虚拟阵元：

```matlab
% 虚拟阵列构建
for k = 1:K
    virtual_positions(k) = array.get_positions(t_k)
    virtual_signals(k) = snapshots(:, k)
end
% 虚拟阵列维度: [M×K, 1]，如[512, 1]
```

**问题**：`virtual_signals` 是一个列向量（单快拍），其协方差矩阵：
$$\mathbf{R} = \mathbf{x}\mathbf{x}^H$$

是**秩-1矩阵**！MUSIC算法需要估计噪声子空间，但秩-1矩阵只有1个非零特征值，无法正确分离信号/噪声子空间。

#### 解决方案：时间平滑MUSIC

**核心思想**：将时间轴划分为多个重叠子阵列，每个子阵列产生一个独立样本，累加形成满秩协方差矩阵。

```matlab
% 时间平滑配置
subarray_len = floor(num_snapshots / 2);  % 子阵列长度
num_subarrays = num_snapshots - subarray_len + 1;

% 构建平滑协方差矩阵
R_smoothed = zeros(subarray_virtual_size);
for i = 1:num_subarrays
    current_snapshots = snapshots(:, i : i + subarray_len - 1);
    [positions, signals, ~] = build_virtual_array(current_snapshots, t_axis(i:...));
    R_sub = signals * signals';
    R_smoothed = R_smoothed + R_sub;
end
R_smoothed = R_smoothed / num_subarrays;  % 满秩！
```

**数学证明**：
- 原始：rank(R) = 1
- 平滑后：rank(R_smoothed) = min(num_subarrays, subarray_virtual_size)
- 当 num_subarrays ≥ num_targets + 1 时，可正确估计噪声子空间

### 2.2 2D DOA估计

#### 坐标系定义

```
      z (θ=0°)
      |
      |  θ=75°目标
      | /
      |/_____ y (φ=90°)
     /
    /
   x (φ=0°)
```

- **θ (俯仰角)**: 与z轴夹角，θ=90°为水平面
- **φ (方位角)**: xy平面投影与x轴夹角

#### 方向矢量

```matlab
u = [sind(theta)*cosd(phi); sind(theta)*sind(phi); cosd(theta)];
```

#### 导向矢量

```matlab
% 相位 = 4π/λ × (阵元位置 · 方向)
phase = 4 * pi / lambda * (positions * u);
a = exp(-1j * phase);  % 注意：负号与信号模型一致！
```

### 2.3 四层智能搜索

为实现0.01°精度同时保持计算效率：

| 层次 | 分辨率 | 搜索范围 | 用途 |
|------|--------|----------|------|
| 第1层 | 2.0° | 全范围 | 粗定位 |
| 第2层 | 0.5° | 峰值±5° | 缩小范围 |
| 第3层 | 0.1° | 峰值±2° | 精细化 |
| 第4层 | 0.01° | 峰值±0.5° | 最终精度 |

**加速效果**：相比均匀0.01°网格，计算量减少约100倍。

---

## 三、关键参数配置

### 3.1 FMCW雷达参数（统一标准）

| 参数 | 值 | 说明 |
|------|-----|------|
| 载频 | 3 GHz | S波段雷达 |
| 波长 | ~10 cm | λ = c/fc |
| 阵元间距 | 0.5λ | 避免栅瓣 |
| **Chirp周期** | **50 ms** | FMCW标准配置，PRF=20Hz |
| 观测时间 | 0.8-2.0 s | 16-40个Chirp |
| 快拍数 | 16-40 | 每Chirp一个快拍 |
| 平移速度 | 5 m/s | 每Chirp平移0.25m |

### 3.2 合成孔径计算

| 窗口配置 | 快拍数 | 时间 | 合成孔径 |
|----------|--------|------|----------|
| 小窗口 | 16 | 0.8 s | 4m = 40λ |
| 中窗口 | 32 | 1.6 s | 8m = 80λ |
| 大窗口 | 64 | 3.2 s | 16m = 160λ |

**公式**：合成孔径 = 平移速度 × 窗口时间 = 5 m/s × (快拍数 × 50ms)

### 3.2 2D搜索配置

```matlab
% 智能搜索参数
smart_2d.theta_range = [60, 90];    % θ搜索范围
smart_2d.phi_range = [0, 90];       % φ搜索范围
smart_2d.coarse_res = 2.0;          % 第1层: 2°
smart_2d.medium_res = 0.5;          % 第2层: 0.5°
smart_2d.fine_res = 0.1;            % 第3层: 0.1°
smart_2d.ultra_res = 0.01;          % 第4层: 0.01° ← 最终精度
```

### 3.3 阵列类型推荐

| 阵列类型 | 2D能力 | 推荐运动 | 备注 |
|----------|--------|----------|------|
| ULA (线阵) | ❌ 1D | x/y平移 | 仅φ方向 |
| URA (面阵) | ✅ 2D | y平移 | 推荐3×3 |
| 圆阵 | ✅ 2D | 平移 | 全向均匀 |
| L/T/十字阵 | ✅ 2D | 平移 | 稀疏阵列 |

---

## 四、实验结果

### 4.1 主实验结果 (v2.3)

#### RMSE性能（2D模式）

| 阵列 | 运动 | 孔径 | 主瓣 | -10dB | 0dB | 20dB |
|------|------|------|------|-------|-----|------|
| ULA-8 | 静态 | 1.8λ | 0.4° | 11.4° | 8.3° | 9.9° |
| ULA-8 | y平移 | 25.1λ | 0.1° | **2.8°** | **0.87°** | **0.08°** |
| URA-3×3 | 静态 | 0.7λ | 0.3° | 11.2° | 2.5° | 0.32° |
| URA-3×3 | y平移 | 25.5λ | 0.1° | **5.8°** | **2.6°** | **0.21°** |

**关键发现**：
1. y平移（合成孔径25λ）显著优于静态
2. 低SNR下改善可达4-10倍
3. 高SNR下运动阵列可达0.1°级精度

#### 主瓣宽度

| 配置 | 静态 | 平移 | 改善 |
|------|------|------|------|
| ULA-8 | 0.4° | 0.1° | 4× |
| URA-3×3 | 0.3° | 0.1° | 3× |

### 4.2 双目标分辨实验

| 间隔 | 静态阵列 | 运动阵列 | 优势 |
|------|----------|----------|------|
| 1° | ❌ | ✅ | 运动 |
| 2° | ❌ | ✅ | 运动 |
| 3° | ❌ | ✅ | 运动 |
| 5° | ❌/✅ | ✅ | 运动 |

**结论**：运动阵列可分辨1-2°间隔的双目标，静态阵列需要5°以上。

### 4.3 实时处理性能 (v5.0)

| 配置 | Chirp周期 | 窗口 | 搜索网格 | 实时率 | 精度 |
|------|-----------|------|----------|--------|------|
| 1D | 10 ms | 32 | 0.5° | 100% | φ误差<1° |
| **2D** | **50 ms** | **16** | **1°** | **100%** | **φ,θ误差<1°** |

**2D实时处理验证结果**（50ms Chirp周期）：
- φ误差: **0.00°**
- θ误差: **0.03°**
- 平均处理时间: 17.7 ms
- 实时达成率: **100%**

### 4.4 抗震动能力

| 震动 | RMSE_φ | RMSE_θ | 可用性 |
|------|--------|--------|--------|
| 0.05λ | ~0.5° | ~0.5° | ✅ |
| 0.1λ | ~1° | ~1° | ✅ |
| 0.2λ | ~5° | ~5° | ⚠️ |

**结论**：容许约0.1λ（1cm）振动。

---

## 五、快速开始

### 5.1 运行主实验

```matlab
cd scripts
run('comprehensive_motion_array_test.m')
```

**输出**: `validation_results/comprehensive_motion_array_test_<时间戳>/`
- 6张论文级图表（PNG + EPS）
- 实验日志
- 结果数据（.mat）

### 5.2 生成对数坐标图表

```matlab
run('generate_compressed_figures.m')
```

生成对数Y轴版本的图表，更直观展示性能差异。

### 5.3 运行补充实验

```matlab
% 双目标分辨实验 (2D)
run('experiment_dual_target_resolution.m')

% 实时性验证 (2D滑动窗口)
run('experiment_parallel_processing_v2.m')

% 抗震动测试 (2D)
run('experiment_vibration_robustness.m')

% 双目标优势展示 (简化版)
run('experiment_dual_target_advantage.m')
```

### 5.4 自定义实验

```matlab
% 1. 创建阵列
elements = create_ura(3, 3, lambda/2);  % 3×3 URA
array = ArrayPlatform(elements, 1, 1:9);
array.set_trajectory(@(t) struct('position', [0, v*t, 0], 'orientation', [0,0,0]));

% 2. 生成信号
target_pos = 500 * [sind(75)*cosd(30), sind(75)*sind(30), cosd(75)];
targets = {Target(target_pos, [0,0,0], 1)};
sig_gen = SignalGeneratorSimple(radar_params, array, targets);
snapshots = sig_gen.generate_snapshots(t_axis, snr_db);

% 3. 2D DOA估计
estimator = DoaEstimatorSynthetic(array, radar_params);
search_grid.phi = 0:0.5:90;
search_grid.theta = 60:0.5:90;
est_options.search_mode = '2d';
[spectrum, peaks, info] = estimator.estimate(snapshots, t_axis, search_grid, 1, est_options);
```

---

## 六、论文写作建议

### 6.1 核心图表

1. **fig1_静态vs运动阵列_对数** - RMSE对比（对数坐标）
2. **fig2_运动模式对比_对数** - 不同运动的效果
3. **fig3_ULA核心对比_对数** - ULA详细对比
4. **fig4_论文综合图** - 改善倍数统计
5. **fig5_主瓣宽度对比** - 主瓣窄化
6. **fig6_孔径扩展对比** - 孔径扩展

### 6.2 建议章节结构

```
第3章 运动合成孔径DOA估计算法
  3.1 问题描述与系统模型
  3.2 传统方法局限性分析
      3.2.1 合成孔径的秩-1问题
      3.2.2 标准MUSIC失效原因
  3.3 时间平滑MUSIC方法
      3.3.1 时间平滑原理
      3.3.2 子阵列划分策略
      3.3.3 协方差矩阵重构
  3.4 2D DOA估计扩展
  3.5 智能搜索加速策略

第4章 仿真实验与分析
  4.1 实验参数设置
  4.2 静态vs运动阵列对比
  4.3 不同运动模式分析
  4.4 双目标分辨能力
  4.5 实时性与鲁棒性验证
```

### 6.3 关键结论

1. **时间平滑MUSIC**解决了合成孔径单快拍秩-1问题
2. **y平移运动**可实现**14倍孔径扩展**，主瓣宽度从0.4°降至0.1°
3. 低SNR下RMSE改善可达**4-10倍**
4. 双目标分辨能力从5°提升至1-2°
5. 滑动窗口优化实现**90%+实时达成率**
6. 算法容许**1cm以内**的平台震动

---

## 七、常见问题

### Q1: 为什么使用时间平滑而不是空间平滑？

合成虚拟阵列的阵元位置不均匀（随平台运动变化），无法直接应用传统空间平滑。时间平滑利用快拍时间维度创建多个子阵列样本，适用于任意阵列几何。

### Q2: 子阵列长度如何选择？

- 子阵列长度 = floor(快拍数 / 2)
- 子阵列数 = 快拍数 - 子阵列长度 + 1
- 需保证：子阵列数 ≥ 目标数 + 1

### Q3: 为什么纯旋转无法扩展孔径？

旋转只改变阵元方向，不改变最大基线长度。圆形阵列绕中心旋转后，孔径 = 2R 恒定不变。

### Q4: 2D搜索为什么慢？

2D搜索点数 = θ点数 × φ点数。使用四层智能搜索可显著加速，从均匀网格的 ~10000点 减少到 ~500点。

### Q5: 如何验证时间平滑MUSIC有效？

运行 `quick_test_synthetic_aperture.m`，比较主瓣宽度：
- 静态MUSIC: ~0.5°
- 时间平滑MUSIC: ~0.2° (2.5倍改善)

---

## 八、版本历史

| 版本 | 日期 | 主要更新 |
|------|------|----------|
| v1.0 | 2025-12-06 | 基础框架 |
| v2.0 | 2025-12-07 | 合成虚拟阵列方法 |
| v3.0 | 2025-12-08 | 智能搜索、CFAR |
| v4.0 | 2025-12-09 | 时间平滑MUSIC、2D DOA、对数图表 |
| **v5.0** | **2025-12-10** | **FMCW参数统一(50ms Chirp)、实时2D精度优化、信号相干性修复** |

---

*文档版本: v5.0 | 最后更新: 2025-12-10*
