# 现代雷达信号处理仿真框架 - 架构文档

## 1. 框架概述

本框架提供了一个现代化的、面向对象的MATLAB仿真平台，用于处理各类天线阵列在不同运动状态下的目标测向问题。

其核心设计思想为 **“位置中心化 (Position-Centric)”**。与传统方法中针对特定运动（如旋转、平移）设计复杂“运动补偿”算法不同，本框架将所有问题都统一到三维空间中的**瞬时位置**上来。无论是阵列运动还是目标运动，都被视为一系列离散时间点上的天线/目标位置变化。

这种方法从根本上简化了问题，带来了无与伦比的**灵活性**和**可扩展性**。

**主要优势:**
- **统一性**: 无需为直线、圆周或任意复杂运动编写不同的补偿代码。所有运动轨迹都通过一个简单的轨迹函数 `trajectory_func` 来定义。
- **模块化**: 将复杂的雷达系统分解为四个独立、低耦合的模块，逻辑清晰，易于维护和扩展。
- **高可靠性**: 通过 `run_verification.m` 中严格的四步验证法，证明了其在多种条件下的正确性和鲁棒性。

---

## 2. 核心概念：广义导向矢量 (Generalized Steering Vector)

对于运动平台，不同时刻的快照（snapshot）对应着不同的阵列空间位置和姿态。传统的MUSIC算法假设阵列不动（导向矢量不变），因此无法直接使用。

本框架的核心是 **广义MUSIC算法**，它使用一个 **时变阵列流形 (Time-Varying Array Manifold)** 或 **广义导向矢量 (GSV)** 来匹配信号模型。

- **传统导向矢量 `a(u)`**: 一个 `[N x 1]` 的向量，描述一个方向 `u` 的信号在 `N` 个静止天线上的相位分布。
- **广义导向矢量 `A(u)`**: 一个 `[N x M]` 的**矩阵**，描述一个方向 `u` 的信号，在 `M` 个不同时刻（对应 `M` 个不同的阵列位置）的 `N` 个天线上的完整时空相位分布。

`DoaEstimator` 通过构建这个 `A(u)` 矩阵，就可以直接处理由于平台运动导致的时变信号，将运动信息自然地融入到了算法内部，从而无需任何额外的补偿步骤。

---

## 3. 文件结构与模块解析

本框架由四个核心MATLAB类 (`.m` 文件) 和一个主运行脚本构成。

### `ArrayPlatform.m` - 阵列平台类
- **职责**: 定义雷达天线阵的“物理属性”和“运动属性”。
- **核心属性**:
    - `physical_elements`: `[N x 3]` 矩阵，定义了N个物理天线的**相对位置**。这是阵列的“形状”。
    - `tx_indices`, `rx_indices`: 定义了哪些天线用于发射，哪些用于接收，以形成MIMO虚拟阵列。
    - `trajectory_func`: 一个函数句柄，输入时间 `t`，输出平台此时的 `{位置, 姿态}`。这是阵列的“运动轨迹”。
- **核心方法**:
    - `get_mimo_virtual_positions(t)`: **框架的心脏**。它根据阵列的物理布局和在 `t` 时刻的运动状态，计算出所有MIMO虚拟阵元的**绝对三维空间坐标**。

### `Target.m` - 目标类
- **职责**: 定义被探测目标的物理属性。
- **核心属性**:
    - `initial_position`: `[1 x 3]` 向量，目标的初始位置。
    - `velocity`: `[1 x 3]` 向量，目标的速度。
- **核心方法**:
    - `get_position_at(t)`: 计算目标在 `t` 时刻的绝对三维空间坐标。

### `SignalGenerator.m` - 信号发生器类
- **职责**: 模拟物理世界的电磁波传播，生成雷达接收到的回波信号。
- **工作流程**:
    1. 接收一个 `ArrayPlatform` 对象和一个 `Target` 对象列表。
    2. 在每个时间点 `t`：
    3. 调用 `array.get_mimo_virtual_positions(t)` 获取天线位置。
    4. 调用 `target.get_position_at(t)` 获取目标位置。
    5. 基于两者之间的几何关系，计算出信号的**往返路径延迟**，并转换为基带复信号的**相位**。
    6. 最终生成一个 `[N_virt x M]` 的快照矩阵，其中 `N_virt` 是虚拟阵元数，`M` 是快照数（采样点数）。

### `DoaEstimator.m` - 角度估计算法类
- **职责**: 实现核心的信号处理算法，从接收信号中估计出目标方向。
- **工作流程**:
    1. 接收信号快照矩阵 `snapshots` 和对应的 `ArrayPlatform` 对象。
    2. 计算信号的协方差矩阵 `Rxx`。
    3. 对 `Rxx` 进行特征值分解，得到噪声子空间 `Qn`。
    4. **遍历**所有可能的目标方向 `(theta, phi)`：
    5. 对于每一个方向 `u`，利用 `array.get_mimo_virtual_positions(t)` 在所有 `M` 个时刻的位置，构建出对应的**广义导向矢量矩阵 `A(u)`**。
    6. 计算MUSIC空间谱：`P(u) = 1 / ( A(u)' * Qn * Qn' * A(u) )`。
    7. 寻找谱峰，谱峰对应的 `(theta, phi)` 即为目标方向估计值。

### `run_verification.m` - 验证与示例脚本
- **职责**: 作为框架的“单元测试”和“使用说明书”。
- **内容**:
    - **第一步：环境设置**: 定义了一个L型阵列和一个已知方向的目标，作为后续测试的“标准答案”。
    - **第二步：理想验证**: 在无噪声情况下，分别测试了**静态**和**高速动态**两种场景，验证了算法的核心功能。
    - **第三步：统计验证**: 在有噪声的情况下，进行100次蒙特卡洛仿真，检验算法的稳定性和精度。
    - **第四步：可视化**: 将上述所有结果绘制成图，直观地证明框架的正确性。

---

## 4. 如何使用

要仿真一个新的场景，主要步骤如下 (可完全参考 `run_verification.m`):

1.  **定义雷达参数**: 设置载频 `fc` 等。
2.  **创建阵列平台**:
    ```matlab
    % 定义天线物理位置 (例如一个4单元方阵)
    spacing = lambda / 2;
    elements = [ -spacing/2, -spacing/2, 0;
                  spacing/2, -spacing/2, 0;
                 -spacing/2,  spacing/2, 0;
                  spacing/2,  spacing/2, 0 ];
    tx = 1; % 第1根发射
    rx = 1:4; % 所有都接收
    array = ArrayPlatform(elements, tx, rx);

    % 定义运动轨迹 (例如绕Z轴匀速转动)
    rotation_speed_dps = 10; % 10 deg/sec
    traj_func = @(t) struct('position', [0,0,0], 'orientation', [0, 0, rotation_speed_dps * t]);
    array = array.set_trajectory(traj_func);
    ```
3.  **创建目标**:
    ```matlab
    target = Target([1000, 500, 300], [10, 5, 0], 1);
    ```
4.  **生成信号**:
    ```matlab
    sig_gen = SignalGenerator(radar_params, array, target);
    snapshots = sig_gen.generate_snapshots(t_axis, snr_db);
    ```
5.  **进行估计**:
    ```matlab
    estimator = DoaEstimator(array, radar_params);
    [spectrum, grid] = estimator.estimate_gmusic(snapshots, t_axis, 1, search_grid);
    ```
---

## 5. 数学基础

- **MIMO虚拟阵元位置**:
  \[ \mathbf{p}_v = \mathbf{p}_t + \mathbf{p}_r \]
- **信号模型 (远场近似)**:
  \[ x(t) \propto \exp\left(j \frac{2\pi}{\lambda} (\mathbf{p}_v(t) \cdot \mathbf{u})\right) \]
- **广义MUSIC谱**:
  \[ P_{MUSIC}(\mathbf{u}) = \frac{1}{\mathrm{Tr}\left( \mathbf{A}(\mathbf{u})^H \mathbf{Q}_n \mathbf{Q}_n^H \mathbf{A}(\mathbf{u}) \right)} \]
  其中 `Tr` 代表矩阵的迹。
