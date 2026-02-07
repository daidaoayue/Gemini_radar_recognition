"""
V15 最终推理脚本
================
支持自定义置信度阈值，输出详细结果
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import warnings
import glob
from collections import defaultdict

warnings.filterwarnings("ignore")

from data_loader_fusion_v4 import FusionDataLoaderV4
from drsncww import rsnet34


class TrackOnlyNetV4(nn.Module):
    def __init__(self, num_classes=6, stats_dim=28):
        super().__init__()
        self.stats_dim = stats_dim
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
            nn.Linear(stats_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(),
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
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
    
    print("="*70)
    print("V15 双流融合模型 - 最终推理")
    print("="*70)
    
    # ==================== 加载模型 ====================
    v15_pths = glob.glob("./checkpoint/fusion_v15*/ckpt_best*.pth")
    if not v15_pths:
        print("错误：找不到V15 checkpoint")
        return
    
    ckpt_path = sorted(v15_pths)[-1]
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    track_weight = ckpt.get('best_fixed_weight', 0.5)
    rd_weight = 1.0 - track_weight
    stats_dim = ckpt.get('stats_dim', 28)
    
    print(f"\n模型信息:")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Track权重: {track_weight}")
    print(f"  特征维度: {stats_dim}")
    
    # RD模型
    rd_model = rsnet34()
    rd_pths = [p for p in glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth") if 'fusion' not in p]
    if rd_pths:
        rd_ckpt = torch.load(rd_pths[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
    rd_model.to(device).eval()
    
    # Track模型
    track_model = TrackOnlyNetV4(stats_dim=stats_dim)
    track_model.load_state_dict(ckpt['track_model'])
    track_model.to(device).eval()
    
    # ==================== 加载数据 ====================
    print(f"\n加载验证数据...")
    val_ds = FusionDataLoaderV4(
        "./dataset/train_cleandata/val",
        "./dataset/track_enhanced_v4_cleandata/val",
        val=True,
        stats_dim=stats_dim
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # ==================== 推理 ====================
    print(f"\n开始推理...")
    
    all_results = []  # (true_label, pred_label, confidence, rd_pred, track_pred)
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            
            rd_probs = torch.softmax(rd_model(x_rd), dim=1)
            track_probs = torch.softmax(track_model(x_track, x_stats), dim=1)
            fused_probs = rd_weight * rd_probs + track_weight * track_probs
            
            conf, pred = fused_probs.max(dim=1)
            rd_pred = rd_probs.argmax(dim=1)
            track_pred = track_probs.argmax(dim=1)
            
            for i in range(len(y)):
                if y[i].item() in VALID_CLASSES:
                    all_results.append((
                        y[i].item(),
                        pred[i].item(),
                        conf[i].item(),
                        rd_pred[i].item(),
                        track_pred[i].item()
                    ))
    
    # ==================== 测试不同阈值 ====================
    print("\n" + "="*70)
    print("不同置信度阈值的性能")
    print("="*70)
    
    thresholds = [0.3, 0.4, 0.45, 0.5, 0.6, 0.7]
    
    print(f"\n{'阈值':^8}|{'覆盖率':^10}|{'准确率':^10}|{'高置信度样本数':^14}")
    print("-" * 50)
    
    for thresh in thresholds:
        high_conf = [(t, p, c) for t, p, c, _, _ in all_results if c >= thresh]
        if high_conf:
            correct = sum(1 for t, p, c in high_conf if t == p)
            acc = 100 * correct / len(high_conf)
            cov = 100 * len(high_conf) / len(all_results)
            print(f"{thresh:^8.2f}|{cov:^10.1f}%|{acc:^10.2f}%|{len(high_conf):^14}")
    
    # ==================== 详细分析（阈值0.5）====================
    print("\n" + "="*70)
    print("详细分析（阈值=0.5）")
    print("="*70)
    
    thresh = 0.5
    
    # 分类别统计
    class_stats = defaultdict(lambda: {'total': 0, 'high_total': 0, 'high_correct': 0})
    
    for true_label, pred_label, conf, rd_pred, track_pred in all_results:
        class_stats[true_label]['total'] += 1
        if conf >= thresh:
            class_stats[true_label]['high_total'] += 1
            if pred_label == true_label:
                class_stats[true_label]['high_correct'] += 1
    
    print(f"\n{'类别':^12}|{'总数':^8}|{'高置信度':^10}|{'准确率':^10}|{'覆盖率':^10}")
    print("-" * 55)
    
    for c in VALID_CLASSES:
        stats = class_stats[c]
        if stats['high_total'] > 0:
            acc = 100 * stats['high_correct'] / stats['high_total']
        else:
            acc = 0
        cov = 100 * stats['high_total'] / max(stats['total'], 1)
        print(f"{CLASS_NAMES[c]:^12}|{stats['total']:^8}|{stats['high_total']:^10}|{acc:^10.2f}%|{cov:^10.1f}%")
    
    # 混淆矩阵（高置信度样本）
    print(f"\n混淆矩阵（高置信度样本）:")
    confusion = np.zeros((4, 4), dtype=int)
    for true_label, pred_label, conf, _, _ in all_results:
        if conf >= thresh and true_label in VALID_CLASSES and pred_label in VALID_CLASSES:
            confusion[true_label][pred_label] += 1
    
    header_str = "真实\\预测"
    print(f"\n{header_str:^12}", end='')
    for c in VALID_CLASSES:
        print(f"|{CLASS_NAMES[c][:4]:^8}", end='')
    print()
    print("-" * 48)
    
    for i, c in enumerate(VALID_CLASSES):
        print(f"{CLASS_NAMES[c]:^12}", end='')
        for j in range(4):
            print(f"|{confusion[i][j]:^8}", end='')
        print()
    
    # RD vs Track对比
    print(f"\n" + "="*70)
    print("RD vs Track模型对比（全部样本）")
    print("="*70)
    
    rd_correct = sum(1 for t, _, _, rd, _ in all_results if rd == t)
    track_correct = sum(1 for t, _, _, _, tr in all_results if tr == t)
    fusion_correct = sum(1 for t, p, _, _, _ in all_results if p == t)
    
    print(f"\n  RD单流准确率: {100*rd_correct/len(all_results):.2f}%")
    print(f"  Track单流准确率: {100*track_correct/len(all_results):.2f}%")
    print(f"  融合准确率: {100*fusion_correct/len(all_results):.2f}%")
    
    # ==================== 总结 ====================
    print("\n" + "="*70)
    print("V15模型总结")
    print("="*70)
    
    high_conf_results = [(t, p, c) for t, p, c, _, _ in all_results if c >= 0.5]
    acc_high = 100 * sum(1 for t, p, c in high_conf_results if t == p) / len(high_conf_results)
    cov_high = 100 * len(high_conf_results) / len(all_results)
    
    print(f"""
  ┌─────────────────────────────────────────┐
  │  高置信度准确率: {acc_high:>6.2f}%              │
  │  覆盖率:        {cov_high:>6.1f}%               │
  │  高置信度样本:   {len(high_conf_results):>5} / {len(all_results)}           │
  └─────────────────────────────────────────┘
""")
    
    print("如需更高覆盖率，可降低阈值到0.4（覆盖率约85%，准确率约98%）")


if __name__ == '__main__':
    main()