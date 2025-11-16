# 🚀 运行实验指南

## 📋 实验清单

### 完整验证实验 ⭐⭐⭐⭐⭐
**脚本**：`comprehensive_validation.m`  
**目标**：一站式完整验证系统，包含4个实验  
**预计时间**：20-40分钟（智能搜索加速）  
**输出**：10张对比图表 + 性能总结

**包含的实验**：
1. **实验1**：角度分辨率（双目标，CA-CFAR增强）
2. **实验2**：有效孔径扩展
3. **实验3**：鲁棒性测试（RMSE vs SNR）
4. **实验4**：最优轨迹-阵列组合探索（新增）

### 单独测试脚本
- `test_cfar_peak_detection.m` - CA-CFAR性能验证
- `test_smart_search_single_target.m` - 智能搜索精度验证
- `test_optimal_trajectory_array.m` - 完整轨迹-阵列测试（24组合）

---

## 🎯 实验1：综合性能验证

### 运行命令

**方式A：MATLAB命令窗口**
```matlab
clear classes; clear all; clc
comprehensive_validation
```

**方式B：批处理运行（后台）**
```matlab
matlab -batch "comprehensive_validation"
```

---

### 实验内容

#### 实验1-1：角度分辨率测试（双目标）
- **测试场景**：4种角度间隔（0.5°, 1.0°, 2.0°, 5.0°）
- **对比对象**：8元静态圆阵 vs 8元旋转合成孔径
- **预计时间**：5-8分钟
- **输出图表**：
  - `1A_resolution_normalized.png` - 归一化对比
  - `1B_resolution_dB.png` - dB尺度对比
  - `1C_resolution_1D_slices.png` - 1D切片对比 ⭐推荐

#### 实验1-2：有效孔径扩展测试
- **测试场景**：3种阵元数（4元, 8元, 16元）
- **评估指标**：波束宽度（3dB）
- **预计时间**：3-5分钟
- **输出图表**：
  - `2_aperture_extension.png` - 波束宽度对比
  - `2B_beam_pattern_slices.png` - 波束切片详图

#### 实验1-3：鲁棒性测试（RMSE vs SNR）
- **测试场景**：6个SNR点 × 20次蒙特卡洛试验
- **对比指标**：DOA估计精度
- **预计时间**：6-12分钟（支持断点续跑）
- **输出图表**：
  - `3_rmse_vs_snr.png` - RMSE曲线

#### 实验1-4：性能总结
- **输出图表**：
  - `4_performance_summary.png` - 文字总结表

---

### 断点续跑支持

如果实验中断（Ctrl+C或程序崩溃）：

**查看进度**：
```matlab
check_validation_progress
```

**继续运行**：
```matlab
comprehensive_validation
% 会自动询问是否继续之前的进度
```

**重新开始**：
```matlab
reset_validation_progress
% 选择选项2：重置所有进度
```

---

### 预期结果

**关键指标**：
- ✅ 角度分辨率：提升 **3-5倍**
- ✅ 抗噪性能：RMSE改善 **79-88%**
- ✅ 有效孔径：4元阵改善 **4-5倍**

**输出目录**：`validation_results/`
- 7张PNG图片
- 5个MAT数据文件
- 2个README文档

---

## 🔬 实验2：最优轨迹-阵列组合探索

### 运行命令

```matlab
clear; clc; close all
test_optimal_trajectory_array
```

---

### 实验内容

**测试矩阵**：4种阵列 × 6种轨迹 = **24组组合**

#### 阵列配置（8元）
1. 均匀线阵(ULA)
2. 圆形阵列
3. 矩形阵列(URA)
4. L型阵列

#### 运动轨迹
1. 静止（基准）
2. 直线平移
3. 圆周旋转
4. 螺旋运动
5. 8字轨迹
6. 圆形平移

#### 评估指标
- 波束宽度（分辨率）
- 空间采样点数（孔径扩展）
- DOA估计误差（精度）

---

### 预期结果

**性能排名**：
```
🥇 最优组合: 圆形阵列 + 圆周旋转
   - 分辨率提升: 3-4倍
   - 空间覆盖: 512点
   
🥈 次优组合: ULA + 螺旋运动
   - 分辨率提升: 2.5-3倍
```

**输出图表**：
- 3张热力图（波束宽度、空间覆盖、DOA误差）
- 4张柱状图（各阵列的轨迹性能对比）

---

## ⚙️ 实验参数调整

### 如需加快速度

**修改快拍数**（`comprehensive_validation.m` 第147行）：
```matlab
num_snapshots_base = 32;  % 从64改为32，速度快2倍
```

**修改蒙特卡洛试验次数**（第327行）：
```matlab
num_trials_mc = 10;  % 从20改为10
```

### 如需提高精度

**修改细搜索分辨率**（第35行）：
```matlab
smart_grid.fine_res = 0.1;  % 从0.2改为0.1
```

---

## 📊 结果使用建议

### 论文图表
- **首选**：`1C_resolution_1D_slices.png` - 最直观
- **补充**：`3_rmse_vs_snr.png` - 展示鲁棒性
- **总结**：`4_performance_summary.png` - 文字总结

### 设计指导
- 运行实验2，找到最优阵列-轨迹组合
- 根据应用场景（分辨率 vs 精度 vs 覆盖）选择配置

### 性能验证
- 对比传统方法的改善倍数
- 量化SNR容限和分辨率极限

---

## 🆘 常见问题

### Q1: GPU加速失败？
**A**: 在`DoaEstimatorIncoherent.m`第15行设置：
```matlab
obj.use_gpu = false;
```

### Q2: 内存不足？
**A**: 减少快拍数或降低搜索网格分辨率

### Q3: 实验3太慢？
**A**: 已支持断点续跑，可以随时中断

### Q4: 如何只运行部分实验？
**A**: 修改`progress.mat`，设置已完成的实验编号

---

## 📈 时间估算

| 实验 | 预计时间 | 可中断 |
|-----|---------|--------|
| 实验1-1 | 5-8分钟 | ✓ |
| 实验1-2 | 3-5分钟 | ✓ |
| 实验1-3 | 6-12分钟 | ✓✓ 每个SNR点 |
| 实验1-4 | 1-2分钟 | - |
| **总计** | **15-30分钟** | |
| 实验2 | 20-40分钟 | ✗ |

---

## 🎉 开始实验！

**推荐运行顺序**：

```matlab
% 1. 综合验证（必做）
clear classes; clear all; clc
comprehensive_validation

% 2. 最优组合探索（可选）
clear; clc; close all
test_optimal_trajectory_array
```

祝实验顺利！ 🚀

