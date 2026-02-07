"""
温度缩放(Temperature Scaling)优化脚本
=====================================
原理：
  校准后的置信度 = softmax(logits / T)
  - T > 1: 降低置信度（模型变谨慎）
  - T < 1: 提高置信度（模型变自信）

目标：
  在保持准确率的同时，提高覆盖率
  让更多"接近阈值"的正确样本超过阈值

使用方法：
  python temperature_scaling.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import warnings
import glob
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")

from data_loader_fusion_v3 import FusionDataLoaderV3

try:
    from drsncww import rsnet34
except ImportError:
    print("错误: 找不到 drsncww.py")
    exit()


class TrackOnlyNetV3(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.temporal_net = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten()
        )
        self.stats_net = nn.Sequential(
            nn.Linear(20, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 6)
        )
    
    def forward(self, x_temporal, x_stats):
        feat_temporal = self.temporal_net(x_temporal)
        feat_stats = self.stats_net(x_stats)
        return self.classifier(torch.cat([feat_temporal, feat_stats], dim=1))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 配置
    RD_VAL = "./dataset/train_cleandata/val"
    TRACK_VAL = "./dataset/track_enhanced_cleandata/val"
    VALID_CLASSES = [0, 1, 2, 3]
    
    # 查找V14 checkpoint
    v14_pths = glob.glob("./checkpoint/fusion_v14*/ckpt_best*.pth")
    if not v14_pths:
        v14_pths = glob.glob("./checkpoint/fusion_v13*/ckpt_best*.pth")
    
    if not v14_pths:
        print("错误: 找不到checkpoint")
        return
    
    CKPT_PATH = sorted(v14_pths)[-1]
    
    print("="*70)
    print("温度缩放(Temperature Scaling)优化")
    print("="*70)
    print(f"Checkpoint: {CKPT_PATH}")
    
    # 加载数据
    print(f"\n加载数据...")
    val_ds = FusionDataLoaderV3(RD_VAL, TRACK_VAL, val=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # 加载模型
    print("加载模型...")
    
    # RD模型
    rd_model = rsnet34()
    rd_pths = glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth")
    rd_pths = [p for p in rd_pths if 'fusion' not in p]
    if rd_pths:
        rd_ckpt = torch.load(rd_pths[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
    rd_model.to(device).eval()
    
    # Track模型
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    track_weight = ckpt.get('best_fixed_weight', ckpt.get('track_weight', 0.45))
    rd_weight = 1.0 - track_weight
    
    track_model = TrackOnlyNetV3()
    track_model.load_state_dict(ckpt['track_model'])
    track_model.to(device).eval()
    
    print(f"融合权重: RD={rd_weight:.2f}, Track={track_weight:.2f}")
    
    # 收集所有logits
    print("\n收集预测结果...")
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            
            rd_logits = rd_model(x_rd)
            track_logits = track_model(x_track, x_stats)
            
            # 融合logits（在logit空间融合）
            # 注意：原来是在概率空间融合，这里为了温度缩放在logit空间操作
            rd_probs = torch.softmax(rd_logits, dim=1)
            track_probs = torch.softmax(track_logits, dim=1)
            fused_probs = rd_weight * rd_probs + track_weight * track_probs
            
            # 转回logits用于温度缩放
            fused_logits = torch.log(fused_probs + 1e-8)
            
            for i in range(len(y)):
                if y[i].item() in VALID_CLASSES:
                    all_logits.append(fused_logits[i].cpu().numpy())
                    all_labels.append(y[i].item())
    
    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    
    print(f"总样本数: {len(all_labels)}")
    
    # 测试不同温度的效果
    print("\n" + "="*70)
    print("测试不同温度参数")
    print("="*70)
    
    def evaluate_with_temperature(T, conf_thresh=0.5):
        """用温度T评估"""
        scaled_probs = []
        for logits in all_logits:
            scaled = np.exp(logits / T) / np.sum(np.exp(logits / T))
            scaled_probs.append(scaled)
        scaled_probs = np.array(scaled_probs)
        
        preds = np.argmax(scaled_probs, axis=1)
        confs = np.max(scaled_probs, axis=1)
        
        # 高置信度
        high_mask = confs >= conf_thresh
        high_correct = np.sum((preds == all_labels) & high_mask)
        high_total = np.sum(high_mask)
        high_acc = 100 * high_correct / high_total if high_total > 0 else 0
        coverage = 100 * high_total / len(all_labels)
        
        # 全部
        all_correct = np.sum(preds == all_labels)
        all_acc = 100 * all_correct / len(all_labels)
        
        return high_acc, coverage, all_acc, high_total
    
    print(f"\n{'温度T':^10}|{'高置信度准确率':^15}|{'覆盖率':^12}|{'全样本准确率':^15}|{'高置信度样本':^12}")
    print("-" * 70)
    
    best_T = 1.0
    best_score = 0  # 准确率 * 覆盖率的调和平均
    
    temperatures = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
    
    for T in temperatures:
        high_acc, coverage, all_acc, high_total = evaluate_with_temperature(T)
        
        # 计算得分：准确率和覆盖率的调和平均
        if high_acc > 0 and coverage > 0:
            score = 2 * (high_acc/100) * (coverage/100) / ((high_acc/100) + (coverage/100))
        else:
            score = 0
        
        mark = ""
        if score > best_score and high_acc >= 98:  # 保证准确率不低于98%
            best_score = score
            best_T = T
            mark = " *"
        
        print(f"{T:^10.1f}|{high_acc:^15.2f}%|{coverage:^12.1f}%|{all_acc:^15.2f}%|{high_total:^12}{mark}")
    
    # 精细搜索最佳温度
    print(f"\n精细搜索最佳温度（在{best_T-0.2:.1f}到{best_T+0.2:.1f}之间）...")
    
    fine_temps = np.arange(max(0.3, best_T - 0.2), best_T + 0.21, 0.05)
    
    for T in fine_temps:
        high_acc, coverage, all_acc, high_total = evaluate_with_temperature(T)
        
        if high_acc > 0 and coverage > 0:
            score = 2 * (high_acc/100) * (coverage/100) / ((high_acc/100) + (coverage/100))
        else:
            score = 0
        
        if score > best_score and high_acc >= 98:
            best_score = score
            best_T = T
    
    # 显示最佳结果
    print("\n" + "="*70)
    print("最佳温度参数")
    print("="*70)
    
    high_acc, coverage, all_acc, high_total = evaluate_with_temperature(best_T)
    
    # 对比原始（T=1.0）
    orig_acc, orig_cov, orig_all, orig_total = evaluate_with_temperature(1.0)
    
    print(f"""
最佳温度: T = {best_T:.2f}

效果对比:
                    原始(T=1.0)    优化(T={best_T:.2f})     变化
  高置信度准确率:    {orig_acc:.2f}%        {high_acc:.2f}%        {high_acc-orig_acc:+.2f}%
  覆盖率:           {orig_cov:.1f}%         {coverage:.1f}%         {coverage-orig_cov:+.1f}%
  高置信度样本:      {orig_total}           {high_total}            {high_total-orig_total:+d}
  全样本准确率:      {orig_all:.2f}%        {all_acc:.2f}%        {all_acc-orig_all:+.2f}%
""")
    
    # 分析效果
    if best_T < 1.0:
        print(f"分析: T={best_T:.2f} < 1.0，模型置信度被放大")
        print(f"  → 更多样本超过阈值，覆盖率提高")
        if high_acc < orig_acc:
            print(f"  ⚠ 但准确率略有下降，需权衡")
    elif best_T > 1.0:
        print(f"分析: T={best_T:.2f} > 1.0，模型置信度被压缩")
        print(f"  → 模型变得更谨慎")
    else:
        print(f"分析: T=1.0，原始模型已经较优")
    
    # 建议
    print("\n" + "="*70)
    print("使用建议")
    print("="*70)
    
    print(f"""
在推理时使用温度缩放:

```python
# 原始推理
probs = softmax(logits)

# 温度缩放推理
T = {best_T:.2f}
probs = softmax(logits / T)
```

或者直接降低置信度阈值:
  - 阈值0.5: 覆盖率{orig_cov:.1f}%, 准确率{orig_acc:.2f}%
  - 阈值0.4: """)
    
    # 测试阈值0.4的效果
    acc_04, cov_04, _, _ = evaluate_with_temperature(1.0, conf_thresh=0.4)
    print(f"覆盖率{cov_04:.1f}%, 准确率{acc_04:.2f}%")
    
    print(f"""
推荐方案:
  1. 如果追求高准确率: 使用原始阈值0.5
  2. 如果追求高覆盖率: 使用温度T={best_T:.2f}或降低阈值到0.4
  3. 平衡方案: 使用温度T={best_T:.2f} + 阈值0.5
""")


if __name__ == '__main__':
    main()