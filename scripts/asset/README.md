# Asset - 核心类文件

本目录包含运动阵列DOA估计系统的核心类文件。

## 类列表

| 文件名 | 描述 |
|--------|------|
| `ArrayPlatform.m` | 阵列平台类，管理阵元位置、MIMO配置和运动轨迹 |
| `Target.m` | 目标类，描述目标位置、速度和RCS |
| `SignalGenerator.m` | FMCW信号生成器（完整模式） |
| `SignalGeneratorSimple.m` | 简化信号生成器（直接空间相位） |
| `DoaEstimator.m` | 标准GMUSIC DOA估计器 |
| `DoaEstimatorIncoherent.m` | 非相干MUSIC估计器 |
| `DoaEstimatorSynthetic.m` | **核心** - 合成虚拟阵列MUSIC估计器 |
| `find_peaks_cfar.m` | CA-CFAR峰值检测 |
| `smart_doa_search.m` | 多层智能搜索加速 |

## 使用方法

在MATLAB脚本中添加路径：
```matlab
addpath('asset');
```

## 核心类关系

```
ArrayPlatform (阵列几何与运动)
      ↓
SignalGeneratorSimple (生成快拍数据)
      ↓
DoaEstimatorSynthetic (合成孔径DOA估计)
      ↓
smart_doa_search + find_peaks_cfar (加速与多目标检测)
```


