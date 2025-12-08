# 运动阵列DOA估计系统 - 技术文档

**版本**: v3.0  
**更新日期**: 2025-12-07  
**适用代码**: `comprehensive_motion_array_test.m` 及 `asset/` 目录

---

## 一、系统概述

### 1.1 研究目标

利用阵列平台的**运动**形成合成孔径，扩展有效孔径，提升DOA估计精度和角度分辨率。

### 1.2 核心发现

| 发现 | 结论 |
|------|------|
| 纯旋转 | ❌ **无法扩展孔径**（最大基线不变） |
| 平移运动 | ✅ **有效扩展孔径**（合成孔径=物理+平移距离） |
| 非相干MUSIC | ⚠️ 存在rank-1矩阵问题和SNR损失 |
| 合成虚拟阵列 | ✅ **最优方案**（时间展开为空间） |

### 1.3 代码架构

```
scripts/
├── asset/                              # 核心类
│   ├── ArrayPlatform.m                 # 阵列平台
│   ├── DoaEstimatorSynthetic.m         # DOA估计器 ⭐
│   ├── SignalGeneratorSimple.m         # 信号生成
│   ├── Target.m                        # 目标定义
│   ├── find_peaks_cfar.m               # CFAR峰值检测
│   └── smart_doa_search.m              # 智能搜索
│
├── 主实验脚本
│   ├── comprehensive_motion_array_test.m  # 综合测试 ⭐
│   ├── experiment_parallel_processing_v2.m # 实时性测试
│   ├── experiment_vibration_robustness.m   # 抗震动测试
│   └── experiment_dual_target_resolution.m # 双目标分辨
│
├── validation_results/                 # 实验结果
└── debug/                              # 调试脚本
```

---

## 二、算法原理

### 2.1 三种DOA方法对比

#### 方法1: 相干GMUSIC（静态阵列）

$$\mathbf{R}_{xx} = \frac{1}{K} \sum_{k=1}^{K} \mathbf{x}_k \mathbf{x}_k^H$$

- **矩阵维度**: M×M（如8×8）
- **适用**: 静态阵列
- **问题**: 运动时导向矢量失配

#### 方法2: 非相干MUSIC（有缺陷）

$$P_{avg}(\theta, \phi) = \frac{1}{K} \sum_{k=1}^{K} P_k(\theta, \phi)$$

- **致命问题**: 单快拍协方差矩阵 $\mathbf{R}_k = \mathbf{x}_k \mathbf{x}_k^H$ 是**秩-1矩阵**
- **后果**: 噪声子空间估计不准确，SNR损失约6dB

#### 方法3a: 相干合成孔径波束形成 (CSA-BF) ⭐（默认）

$$\text{M阵元} \times \text{K快拍} \rightarrow \text{M×K 虚拟阵列}$$

```matlab
% 核心思想：时间展开为空间，使用匹配滤波
for k = 1:K
    virtual_positions(k) = array.get_positions(t_k)
    virtual_signals(k) = snapshots(:, k)
end
% 虚拟阵列: [M×K, 3]，如[512, 3]
% 波束形成: P(θ) = |a(θ)' × x_virtual|² / |a(θ)|²
```

- **不用协方差矩阵**：直接使用信号向量进行匹配滤波
- **优势**: 完全相干利用合成孔径，分辨率最高
- **注意**: 对相位误差敏感

#### 方法3b: 非相干合成孔径MUSIC (ISA-MUSIC)

$$P_{ISA}(\theta) = \sum_{seg} P_{MUSIC}^{(seg)}(\theta)$$

```matlab
% 核心思想：分段MUSIC + 谱累加
for seg = 1:num_segments
    seg_snapshots = snapshots(:, seg_indices);
    Rxx = seg_snapshots * seg_snapshots' / K_seg;  % M×M 满秩！
    [V,D] = eig(Rxx); Qn = V(:, targets+1:end);
    spectrum += music(positions_seg, Qn, θ);
end
```

- **协方差矩阵**: 每段 M×M，满秩（解决秩1问题）
- **优势**: 对相位误差鲁棒，有超分辨能力
- **孔径利用**: 通过段间位置差异间接利用

### 2.2 为什么纯旋转无法扩展孔径？

**数学证明**:

圆形阵列绕中心旋转，阵元i在t时刻位置:
$$\mathbf{r}_i(t) = R \begin{bmatrix} \cos(\theta_i + \omega t) \\ \sin(\theta_i + \omega t) \\ 0 \end{bmatrix}$$

最大基线长度:
$$B_{max}(t) = \max_{i,j} \|\mathbf{r}_i(t) - \mathbf{r}_j(t)\| = 2R \text{ (恒定不变!)}$$

**物理解释**:
```
旋转前：      旋转后：
  ●            ●
 ● ●          ●   ●
●   ●   →    ●     ●  (整体转动，孔径不变)
 ● ●          ●   ●
  ●            ●
```

**平移有效的原因**:
$$D_{synthetic} = D_{physical} + v \cdot T_{obs}$$

例：物理孔径3.5λ + 平移2.5m(0.5s×5m/s) = 28.5λ，扩展8倍！

---

## 三、关键参数配置

### 3.1 孔径与分辨率关系

```
角度分辨率 Δθ ≈ λ / D_effective

要分辨间隔Δ的双目标，需要：
D_effective > λ / Δ
```

### 3.2 参数配置表

| 场景 | 快拍数 | 速度 | 观测时间 | 平移距离 | 有效孔径 | 分辨率 |
|------|--------|------|----------|----------|----------|--------|
| 快速测试 | 32 | 15m/s | 0.32s | 4.8m | 5.1m | 1.1° |
| **标准配置** | 64 | 12m/s | 0.64s | 7.7m | 8.0m | **0.7°** |
| 高性能 | 128 | 15m/s | 1.28s | 19.2m | 19.5m | 0.3° |

### 3.3 雷达参数

```matlab
radar_params.fc = 3e9;              % 载频 3GHz
radar_params.lambda = 0.1;          % 波长 10cm
radar_params.T_chirp = 10e-3;       % Chirp周期 10ms
radar_params.fs = 100e6;            % 采样率 100MHz（避免混叠）
```

### 3.4 阵列几何注意事项

**目标方向与阵列基线需匹配**:

| 阵列类型 | 适用目标方向 | 推荐θ |
|----------|-------------|-------|
| ULA(x轴) | x分量大 | 70-90° |
| L型/十字型 | x或y分量大 | 70-90° |
| 圆阵/方阵 | 水平面内 | 60-90° |

---

## 四、快速开始

### 4.1 运行主实验

```matlab
cd scripts
run('comprehensive_motion_array_test.m')
```

**输出**: `validation_results/comprehensive_motion_array_test_<时间戳>/`
- 6张论文级图表（PNG + EPS）
- 实验日志
- 结果数据

### 4.2 运行补充实验

```matlab
% 实时性验证（滑动窗口优化版）
run('experiment_parallel_processing_v2.m')

% 抗震动测试
run('experiment_vibration_robustness.m')

% 双目标分辨率
run('experiment_dual_target_resolution.m')
```

### 4.3 自定义实验

```matlab
% 1. 创建阵列
elements = create_ula(8, lambda/2);  % 8元ULA
array = ArrayPlatform(elements, 1, 1:8);
array.set_trajectory(@(t) struct('position', [v*t, 0, 0], 'orientation', [0,0,0]));

% 2. 生成信号
targets = {Target([500, 300, 0], [0,0,0], 1)};
sig_gen = SignalGeneratorSimple(radar_params, array, targets);
snapshots = sig_gen.generate_snapshots(t_axis, snr_db);

% 3. DOA估计
estimator = DoaEstimatorSynthetic(array, radar_params);
[spectrum, peaks, info] = estimator.estimate(snapshots, t_axis, search_grid, 1);
```

---

## 五、实验结果解读

### 5.1 主要性能指标

| 指标 | 静态阵列 | 运动阵列(平移) | 改善倍数 |
|------|----------|---------------|----------|
| 孔径 | 1.8λ | 25-26λ | **14×** |
| 低SNR(-15dB) RMSE | 48.8° | 0.3° | **160×** |
| 高SNR(20dB) RMSE | 0.04° | 0.08° | 持平 |
| 主瓣宽度 | 0.4° | 0.3° | 1.3× |

### 5.2 运动模式对比

| 运动模式 | 孔径扩展 | RMSE改善 | 推荐度 |
|----------|---------|----------|--------|
| 静态 | 1× | 基准 | - |
| 纯旋转 | 1.4× | 有限 | ⭐ |
| x平移 | **14×** | **显著** | ⭐⭐⭐ |
| 平移+旋转 | **14×** | **显著** | ⭐⭐⭐ |

### 5.3 同步运算能力

| 版本 | 实时达成率 | 关键优化 |
|------|-----------|----------|
| v1（累积全部） | 36% | - |
| **v2（滑动窗口）** | **90%+** | 固定窗口16快拍 |

### 5.4 抗震动能力

| 震动标准差 | RMSE | 可用性 |
|-----------|------|--------|
| 0.1λ (1cm) | 0.19° | ✅ 可用 |
| 0.2λ (2cm) | 8.63° | ❌ 需要抑制 |

---

## 六、论文写作建议

### 6.1 核心图表（已生成）

1. **fig1_静态vs运动阵列** - RMSE对比
2. **fig2_运动模式对比** - 不同运动的效果
3. **fig3_孔径扩展对比** - 孔径与性能关系
4. **fig4_改善倍数** - 量化改善
5. **fig5_阵列形状** - 不同阵列几何
6. **fig6_虚拟阵列轨迹** - 合成孔径可视化

### 6.2 建议章节结构

```
第3章 运动合成孔径DOA估计算法
  3.1 问题描述与系统模型
  3.2 传统方法局限性分析
      3.2.1 相干GMUSIC的旋转失效
      3.2.2 非相干MUSIC的rank-1问题
  3.3 合成虚拟阵列方法
  3.4 智能搜索加速策略

第4章 仿真实验与分析
  4.1 实验参数设置
  4.2 静态vs运动阵列对比
  4.3 不同运动模式分析
  4.4 抗噪性能评估
  4.5 实时性与鲁棒性验证
```

### 6.3 关键结论

1. **运动合成孔径必须包含平移分量**，纯旋转无效
2. **合成虚拟阵列方法**优于非相干MUSIC
3. 平移运动可实现**14倍孔径扩展**，低SNR改善**160倍**
4. 滑动窗口优化可实现**90%+实时达成率**
5. 算法容许**1cm以内**的平台震动

---

## 七、常见问题

### Q1: 为什么静态阵列在低SNR下误差很大？

静态阵列孔径小（~2λ），主瓣宽，噪声容易导致峰值偏移。运动阵列扩展孔径后主瓣变窄，抗噪性增强。

### Q2: 为什么用1D搜索而不是2D？

对于ULA等线阵，目标在特定方向（如θ=90°水平面），只需搜索φ角，速度提升450倍。

### Q3: 实时处理为什么超时？

原因：累积所有快拍导致矩阵维度线性增长。解决：使用滑动窗口（固定16快拍），保持O(1)复杂度。

### Q4: 如何选择快拍数？

- **16-32**: 快速测试
- **64**: 标准配置（推荐）
- **128**: 高精度需求

---

## 八、文件清单

### 保留文件

| 类别 | 文件 | 说明 |
|------|------|------|
| 核心类 | `asset/*.m` | 6个类文件 |
| 主实验 | `comprehensive_motion_array_test.m` | 综合测试 |
| 补充实验 | `experiment_*.m` | 3个实验脚本 |
| 结果 | `validation_results/` | 实验输出 |
| 文档 | **本文档** | 唯一技术文档 |

### 可删除文件

- 所有其他 `.md` 文档（已整合到本文档）
- `quick_*.m`, `run_*.m` 等旧版脚本
- `comprehensive_validation*.m` 旧版实验
- scripts目录下的 `.png` 图片

---

*文档版本: v3.0 | 最后更新: 2025-12-07*

