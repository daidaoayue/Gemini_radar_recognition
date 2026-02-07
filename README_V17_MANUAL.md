# V17 雷达目标分类系统 - 操作手册

## 目录

1. [系统概述](#1-系统概述)
2. [环境配置](#2-环境配置)
3. [目录结构](#3-目录结构)
4. [快速开始](#4-快速开始)
5. [训练流程](#5-训练流程)
6. [推理部署](#6-推理部署)
7. [常见问题](#7-常见问题)
8. [附录](#8-附录)

---

## 1. 系统概述

### 1.1 系统功能

本系统用于低空雷达目标分类，能够识别以下6类目标：

| 类别ID | 类别名称 | 说明 |
|--------|----------|------|
| 0 | 轻型无人机 | 小型消费级无人机 |
| 1 | 小型无人机 | 工业级无人机 |
| 2 | 鸟类 | 各类飞鸟 |
| 3 | 空飘球 | 气球、风筝等 |
| 4 | 杂波 | 地杂波、气象杂波等 |
| 5 | 其他 | 其他目标 |

### 1.2 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    V17 级联分类系统                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   输入数据                                                   │
│   ├── RD谱图 [2, 128, 128]                                  │
│   └── 航迹特征                                               │
│       ├── 时序特征 [12, 16]                                 │
│       └── 统计特征 [46]                                     │
│                                                             │
│                      ↓                                      │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                   特征融合模块                        │   │
│   │  ┌─────────────┐        ┌─────────────────────┐     │   │
│   │  │  RD分支     │        │  Track分支           │     │   │
│   │  │  DRSN网络   │        │  V17 MultiScale+SE  │     │   │
│   │  │  准确率95%  │        │  多尺度卷积+注意力    │     │   │
│   │  └──────┬──────┘        └──────────┬──────────┘     │   │
│   │         │                          │                │   │
│   │         └────────┬─────────────────┘                │   │
│   │                  │                                  │   │
│   │         加权融合 (RD:50% + Track:50%)               │   │
│   └──────────────────┬──────────────────────────────────┘   │
│                      │                                      │
│                      ↓                                      │
│                                                             │
│              置信度 ≥ 0.5?                                  │
│              /          \                                   │
│            是            否                                  │
│             │              │                                │
│             ↓              ↓                                │
│       ┌──────────┐  ┌──────────────┐                       │
│       │  输出     │  │  级联分类器   │                       │
│       │  主模型   │  │  专门处理     │                       │
│       │  结果     │  │  困难样本     │                       │
│       └──────────┘  └──────────────┘                       │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  性能指标:                                                   │
│    • 整体准确率: 99.56%                                      │
│    • 高置信度准确率: 99.52%                                  │
│    • 低置信度准确率: 100% (级联分类器)                        │
│    • 高置信度覆盖率: 91.2%                                   │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 核心特点

- **双流融合架构**: RD谱图 + 航迹特征
- **多尺度特征提取**: 1×1, 3×3, 5×5多尺度卷积
- **SE注意力机制**: 自适应特征权重
- **级联分类策略**: 对低置信度样本专门处理
- **46维增强特征**: 28维航迹特征 + 18维RCS/Doppler特征

---

## 2. 环境配置

### 2.1 硬件要求

| 配置项 | 最低要求 | 推荐配置 |
|--------|----------|----------|
| GPU | GTX 1060 6GB | RTX 3080 10GB |
| 内存 | 8GB | 16GB |
| 硬盘 | 20GB | 50GB SSD |

### 2.2 软件环境

```bash
# Python版本
Python 3.8+

# 核心依赖
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.20.0
scipy>=1.7.0
tqdm>=4.62.0
```

### 2.3 安装步骤

```bash
# 1. 创建conda环境
conda create -n drsn_env python=3.8
conda activate drsn_env

# 2. 安装PyTorch (根据CUDA版本选择)
# CUDA 11.3
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# 3. 安装其他依赖
pip install numpy scipy tqdm matplotlib

# 4. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 3. 目录结构

```
project/
├── dataset/                          # 数据目录
│   ├── train_cleandata/              # RD谱图数据
│   │   ├── train/                    # 训练集
│   │   │   ├── 0/                    # 轻型无人机
│   │   │   ├── 1/                    # 小型无人机
│   │   │   ├── 2/                    # 鸟类
│   │   │   ├── 3/                    # 空飘球
│   │   │   ├── 4/                    # 杂波
│   │   │   └── 5/                    # 其他
│   │   ├── val/                      # 验证集
│   │   └── test/                     # 测试集
│   │
│   └── track_enhanced_v5_cleandata/  # 46维航迹特征
│       ├── train/
│       ├── val/
│       └── test/
│
├── checkpoint/                       # 模型权重
│   ├── ckpt_best_8_95.07.pth        # RD模型
│   ├── fusion_v17_multiscale_se/    # Track模型
│   │   └── ckpt_best_ep*.pth
│   └── fusion_v17_cascade/          # 级联分类器
│       └── cascade_classifier_*.pth
│
├── code/                             # 代码目录
│   ├── drsncww.py                   # RD网络定义
│   ├── data_loader_fusion_v4.py     # 数据加载器
│   ├── train_fusion_v17_multiscale_se.py  # 训练脚本
│   ├── train_cascade_classifier.py  # 级联分类器训练
│   ├── inference.py                 # 推理脚本
│   └── evaluate_test_set.py         # 测试集评估
│
└── docs/                             # 文档
    ├── README_V17_MANUAL.md         # 操作手册
    └── PROJECT_REPORT.md            # 项目报告
```

---

## 4. 快速开始

### 4.1 推理单个样本

```python
from inference import V17InferenceSystem

# 初始化系统
system = V17InferenceSystem()

# 推理
result = system.predict_from_files(
    rd_file='./data/sample_rd.mat',
    track_file='./data/sample_track.mat'
)

print(f"预测类别: {result['class_name']}")
print(f"置信度: {result['confidence']:.4f}")
```

### 4.2 批量推理

```bash
python inference.py --rd_dir ./data/rd --track_dir ./data/track --output results.csv
```

### 4.3 评估测试集

```bash
python evaluate_test_set.py
```

---

## 5. 训练流程

### 5.1 完整训练流程

```
步骤1: 特征提取 (MATLAB)
        ↓
步骤2: 训练主模型 (Python)
        ↓
步骤3: 训练级联分类器 (Python)
        ↓
步骤4: 验证与评估 (Python)
```

### 5.2 步骤1: 特征提取

使用MATLAB提取46维航迹特征:

```matlab
% 运行特征提取脚本
run('extract_features_v5_with_rcs.m')
```

输出目录: `./dataset/track_enhanced_v5_cleandata/`

### 5.3 步骤2: 训练主模型

```bash
# 训练V17 MultiScale+SE模型
python train_fusion_v17_multiscale_se.py
```

**训练参数:**

| 参数 | 值 | 说明 |
|------|-----|------|
| Epochs | 60 | 训练轮数 |
| Batch Size | 32 | 批次大小 |
| Learning Rate | 1e-3 | 初始学习率 |
| Optimizer | AdamW | 优化器 |
| Scheduler | CosineAnnealing | 学习率调度 |
| Loss | Focal Loss | 损失函数 |
| Bird Weight | 2.0 | 鸟类权重 |

**预期输出:**
```
训练完成!
  最佳Epoch: 12
  准确率: 99.37%
  鸟类: 98.9%
  覆盖率: 93.6%
```

### 5.4 步骤3: 训练级联分类器

```bash
# 训练级联分类器
python train_cascade_classifier.py
```

**预期输出:**
```
级联分类器最佳准确率: 100.00%
低置信度样本提升: +36.97%
```

### 5.5 步骤4: 验证评估

```bash
# 验证集评估
python evaluate_v17.py

# 测试集评估
python evaluate_test_set.py
```

---

## 6. 推理部署

### 6.1 命令行推理

```bash
# 单文件推理
python inference.py \
    --rd_file ./sample_rd.mat \
    --track_file ./sample_track.mat

# 批量推理
python inference.py \
    --rd_dir ./dataset/test/rd \
    --track_dir ./dataset/test/track \
    --output results.csv

# 指定参数
python inference.py \
    --rd_dir ./data \
    --track_dir ./data \
    --conf_thresh 0.5 \
    --track_weight 0.5 \
    --device cuda:0 \
    --output results.csv
```

### 6.2 Python API调用

```python
from inference import V17InferenceSystem
import numpy as np

# 初始化系统
system = V17InferenceSystem(
    conf_threshold=0.5,
    track_weight=0.5,
    device='cuda:0'
)

# 方式1: 从文件推理
result = system.predict_from_files(rd_file, track_file)

# 方式2: 从数据推理
rd_data = np.random.randn(2, 128, 128)        # RD谱图
track_temporal = np.random.randn(12, 16)       # 时序特征
track_stats = np.random.randn(46)              # 统计特征

result = system.predict_single(rd_data, track_temporal, track_stats)

# 方式3: 批量推理
results = system.predict_batch(rd_batch, temporal_batch, stats_batch)

# 方式4: 文件夹批量处理
results = system.predict_folder(rd_dir, track_dir, output_file='results.csv')
```

### 6.3 输出格式

```python
result = {
    'class_id': 2,                    # 类别ID
    'class_name': '鸟类',              # 类别名称
    'confidence': 0.8234,             # 置信度
    'is_high_conf': True,             # 是否高置信度
    'method': 'main',                 # 分类方法 (main/cascade)
    'all_probs': [0.05, 0.02, 0.82, 0.08, 0.02, 0.01]  # 各类概率
}
```

### 6.4 CSV输出格式

| 文件名 | 预测类别 | 类别ID | 置信度 | 高置信度 | 分类方法 |
|--------|----------|--------|--------|----------|----------|
| sample1.mat | 鸟类 | 2 | 0.8234 | 是 | main |
| sample2.mat | 轻型无人机 | 0 | 0.4521 | 否 | cascade |

---

## 7. 常见问题

### Q1: CUDA内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案:**
```bash
# 减小批次大小
python inference.py --batch_size 16

# 或使用CPU
python inference.py --device cpu
```

### Q2: 找不到模型文件

```
FileNotFoundError: 未找到Track模型!
```

**解决方案:**
1. 确认checkpoint目录结构正确
2. 指定模型路径:
```bash
python inference.py --track_model ./path/to/model.pth
```

### Q3: 数据格式不匹配

```
ValueError: 无法在xxx.mat中找到RD数据
```

**解决方案:**
确保MAT文件包含以下字段:
- RD文件: `rd` 或 `data` 字段, 形状 [128, 128] 或 [2, 128, 128]
- Track文件: `track_temporal` [12, 16] 和 `track_stats` [46]

### Q4: 低置信度样本过多

**可能原因:**
1. 输入数据质量差
2. 目标特征不明显
3. 存在新类型目标

**解决方案:**
1. 检查输入数据
2. 调整置信度阈值: `--conf_thresh 0.4`
3. 收集更多样本重新训练

---

## 8. 附录

### 8.1 特征说明

**46维特征组成:**

| 序号 | 特征名 | 说明 |
|------|--------|------|
| 1-5 | 速度特征 | 均值、标准差、最大值、变化率等 |
| 6-10 | 加速度特征 | 切向、法向加速度统计 |
| 11-15 | 航向特征 | 航向角、变化率、稳定性 |
| 16-20 | 位置特征 | 高度、距离、轨迹长度 |
| 21-28 | 运动稳定性 | 稳定性得分、曲率、变异系数 |
| 29-37 | RCS特征 | RCS均值、标准差、变化率 |
| 38-46 | Doppler特征 | 谱宽、带宽、能量集中度 |

### 8.2 性能基准

| 测试环境 | 批次大小 | 推理速度 |
|----------|----------|----------|
| RTX 3080 | 32 | ~200 样本/秒 |
| RTX 2080 | 32 | ~150 样本/秒 |
| GTX 1080 | 16 | ~80 样本/秒 |
| CPU (i7) | 8 | ~20 样本/秒 |

### 8.3 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| V14 | - | 基础版本, 94.62%准确率 |
| V15 | - | 28维特征, 99.53%准确率 |
| V16 | - | 46维特征, ~99.1%准确率 |
| V17 | - | MultiScale+SE, 99.37%准确率 |
| V17+级联 | 当前 | 完整系统, 99.56%准确率 |

---

## 技术支持

如有问题，请检查:
1. 环境配置是否正确
2. 数据格式是否符合要求
3. 模型文件是否完整

---

*文档版本: 1.0*  
*最后更新: 2024年*
