# V16: 微多普勒特征融合方案

## 概述

微多普勒特征是区分鸟类和无人机的**物理本质特征**：
- **鸟类**：翅膀拍动产生2-15Hz的周期性多普勒调制
- **无人机**：螺旋桨产生50-200Hz的高频微多普勒
- **空飘球**：几乎无微多普勒，频谱平坦

## 特征说明（12维）

| 序号 | 特征名 | 说明 | 鸟类特点 |
|------|--------|------|----------|
| 1 | doppler_bandwidth | 多普勒带宽 | 通常较宽 |
| 2 | doppler_centroid_std | 多普勒质心时变标准差 | 波动大 |
| 3 | spectral_entropy | 频谱熵 | 较高（能量分散） |
| 4 | periodicity_score | 周期性强度 | **显著**（翅膀拍动） |
| 5 | dominant_period | 主周期 | 2-15Hz对应 |
| 6 | modulation_index | 调制指数 | 较高 |
| 7 | svd_ratio | SVD主成分比例 | 较低 |
| 8 | time_variance | 时间维方差 | 较高 |
| 9 | doppler_asymmetry | 多普勒不对称性 | 视飞行方向 |
| 10 | peak_sidelobe_ratio | 主峰旁瓣比 | 较低 |
| 11 | spectral_flatness | 频谱平坦度 | 较高 |
| 12 | micro_doppler_strength | 微多普勒强度 | **显著** |

## 实施步骤

### 步骤1：提取微多普勒特征（MATLAB）

```matlab
% 1. 打开MATLAB，运行extract_micro_doppler_v1.m

% 2. 处理训练集
batch_extract_micro_doppler(...
    './dataset/train_cleandata/train', ...
    './dataset/micro_doppler_features/train');

% 3. 处理验证集
batch_extract_micro_doppler(...
    './dataset/train_cleandata/val', ...
    './dataset/micro_doppler_features/val');

% 4. （可选）分析特征差异
analyze_micro_doppler_difference('./dataset/train_cleandata/val');
```

**预计时间**：约10-30分钟（取决于数据量）

**输出目录结构**：
```
dataset/micro_doppler_features/
├── train/
│   ├── 0/
│   │   ├── Track123_xxx_microdoppler.mat
│   │   └── ...
│   ├── 1/
│   ├── 2/
│   └── 3/
└── val/
    ├── 0/
    ├── 1/
    ├── 2/
    └── 3/
```

### 步骤2：训练V16模型（Python）

```bash
python train_fusion_v16.py
```

**配置**：
- 统计特征维度：32（原始20 + 微多普勒12）
- 训练轮数：60
- 学习率：1e-3（余弦退火）

### 步骤3：测试模型

```bash
python train_fusion_v16.py test
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `extract_micro_doppler_v1.m` | MATLAB：提取微多普勒特征 |
| `data_loader_fusion_v5.py` | Python：加载RD+航迹+微多普勒 |
| `train_fusion_v16.py` | Python：训练V16模型 |

## 网络结构

```
V16 TrackNet:
├── temporal_net (Conv1D)     → [B, 256]
│   └── 12维时序 → 64 → 128 → 256
├── stats_net (MLP)           → [B, 128]
│   └── 20维统计 → 64 → 128
├── micro_doppler_net (MLP)   → [B, 64]   ← 新增
│   └── 12维微多普勒 → 32 → 64 → 64
└── classifier
    └── [256+128+64=448] → 128 → 6
```

## 预期效果

| 指标 | V14 (20维) | V16 (32维) | 改进 |
|------|------------|------------|------|
| 鸟类全部准确率 | 91% | **93-95%** | +2-4% |
| 鸟类高置信度 | 97.5% | **98-99%** | +0.5-1.5% |
| 鸟类覆盖率 | 80% | **85-90%** | +5-10% |

**关键改进**：微多普勒的周期性特征（periodicity_score）是鸟类独有的，应该能显著提升鸟类识别。

## 常见问题

### Q1: MATLAB提取特征报错

确保RD数据是`.mat`格式，且包含变量`data`或`rangePower1`。

### Q2: Python找不到微多普勒特征

检查目录结构是否正确：
```bash
ls ./dataset/micro_doppler_features/val/2/
# 应该看到 xxx_microdoppler.mat 文件
```

### Q3: 特征维度不匹配

确保：
1. MATLAB提取的是12维特征
2. Python数据加载器使用`FusionDataLoaderV5`
3. 模型使用`TrackNetV16(stats_dim=32)`

## 下一步改进方向

1. **STFT增强**：对RD数据做短时傅里叶变换，提取更细粒度的时频特征
2. **深度微多普勒网络**：用CNN直接从RD图像学习微多普勒模式
3. **多尺度分析**：在不同时间窗口提取微多普勒，捕捉不同频率的拍动