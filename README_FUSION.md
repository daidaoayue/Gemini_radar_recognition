# 双流融合目标识别系统

## 最终结果

| 指标 | 数值 |
|------|------|
| RD单流准确率 | 95.24% |
| 航迹单流准确率 | 95.16% |
| **融合准确率** | **97.20%** |
| 理论上限 | 97.71% |
| 目标 | 98% |

## 最佳配置

```python
track_weight = 0.40  # 航迹权重
rd_weight = 0.60     # RD权重
conf_thresh = 0.5    # 置信度阈值
```

## 文件结构

```
double_fusion/
├── 数据加载器
│   ├── data_loader_fusion_v2.py    # 基础版（6通道航迹特征）
│   └── data_loader_fusion_v3.py    # 增强版（12通道+20维统计特征）★推荐
│
├── 训练脚本
│   ├── train_track_only_v3.py      # 航迹模型单独训练
│   └── train_fusion_v13.py         # 三阶段融合训练★最终版
│
├── 测试脚本
│   ├── test_fusion_final.py        # 最终测试脚本★推荐
│   ├── recalculate_accuracy.py     # 重新计算准确率（只统计0-3类）
│   └── analyze_fusion_v14.py       # 详细分析脚本
│
├── MATLAB特征提取
│   ├── Extract_Enhanced_Features_v3.m  # 增强航迹特征提取★推荐
│   └── Extract_Aligned_Track_Features.m # 对齐航迹特征提取
│
├── 预训练权重
│   ├── checkpoint/ckpt_best_1_93.17.pth          # RD模型
│   └── checkpoint/fusion_v13/ckpt_best_96.02.pth # 航迹模型
│
└── 数据目录
    ├── dataset/train/2026-1-14/train/  # RD训练数据
    ├── dataset/train/2026-1-14/val/    # RD验证数据
    └── dataset/track_enhanced/         # 增强航迹特征
        ├── train/
        └── val/
```

## 使用流程

### 1. 提取航迹特征（MATLAB）

```matlab
run('Extract_Enhanced_Features_v3.m')
```

选择目录：
- RD数据根目录
- 航迹txt文件夹（Tracks_xxx.txt）
- 点迹txt文件夹（PointTracks_xxx.txt）
- 输出目录：`./dataset/track_enhanced`

### 2. 训练融合模型（Python）

```bash
python train_fusion_v13.py
```

这会依次执行：
1. 训练航迹模型（30 epochs）
2. 测试固定权重融合
3. 可学习融合微调

### 3. 测试

```bash
python test_fusion_final.py
```

## 类别说明

| 代码类别 | 目标类型 | 是否统计 |
|---------|---------|---------|
| 0 | 轻型旋翼无人机 | ✓ |
| 1 | 小型旋翼无人机 | ✓ |
| 2 | 鸟类 | ✓ |
| 3 | 空飘球 | ✓ |
| 4 | 杂波 | ✗（仅训练用） |
| 5 | 其它 | ✗（仅训练用） |

## 逐类别准确率

| 类别 | RD | 航迹 | 融合 |
|------|-----|------|------|
| 轻型无人机 | 96.53% | 97.03% | 97.77% |
| 小型无人机 | 96.65% | 98.51% | 98.88% |
| 鸟类 | 85.80% | 80.86% | **87.04%** |
| 空飘球 | 97.08% | 97.08% | **100%** |

## 待优化方向

要达到98%，需要提升**鸟类**的识别率（当前87.04%）：

1. 增加鸟类训练样本
2. 针对鸟类做数据增强
3. 分析鸟类与其他类别的混淆原因
4. 考虑使用更复杂的RD特征提取网络

## 关键改进历程

| 版本 | 改进 | 准确率 |
|------|------|--------|
| 单流RD | 基线 | 93.17% |
| V10 | 修复预处理bug | 94.34% |
| V11 | 增强航迹特征 | 94.61% |
| V12 | 置信度自适应 | 94.39% |
| **V13** | **三阶段冻结训练** | **97.20%** |
