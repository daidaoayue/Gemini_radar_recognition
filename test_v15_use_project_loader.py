"""
V15 测试脚本（使用项目中的data_loader）
======================================
直接导入 data_loader_fusion_v4.py，确保和训练时一致
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

# 直接导入项目中的数据加载器（和训练时完全一样）
from data_loader_fusion_v4 import FusionDataLoaderV4
from drsncww import rsnet34


class TrackOnlyNetV4(nn.Module):
    """V15用的28维版本 - 必须和训练时一致"""
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
    print("V15 测试（使用项目data_loader）")
    print("="*70)
    
    # ==================== 配置 ====================
    RD_VAL = "./dataset/train_cleandata/val"
    
    # 尝试V4目录
    if os.path.exists("./dataset/track_enhanced_v4_cleandata/val"):
        TRACK_VAL = "./dataset/track_enhanced_v4_cleandata/val"
    else:
        TRACK_VAL = "./dataset/track_enhanced_cleandata/val"
    
    print(f"RD目录: {RD_VAL}")
    print(f"Track目录: {TRACK_VAL}")
    
    # ==================== 加载checkpoint ====================
    v15_pths = glob.glob("./checkpoint/fusion_v15*/ckpt_best*.pth")
    if not v15_pths:
        print("错误: 找不到V15 checkpoint!")
        return
    
    v15_ckpt_path = sorted(v15_pths)[-1]
    print(f"\nCheckpoint: {v15_ckpt_path}")
    
    v15_ckpt = torch.load(v15_ckpt_path, map_location='cpu')
    track_weight = v15_ckpt.get('best_fixed_weight', 0.5)
    rd_weight = 1.0 - track_weight
    stats_dim = v15_ckpt.get('stats_dim', 28)
    
    print(f"  Track权重: {track_weight}")
    print(f"  stats_dim: {stats_dim}")
    print(f"  训练时最佳准确率: {v15_ckpt.get('best_acc', 'N/A')}")
    
    # ==================== 加载RD模型 ====================
    rd_model = rsnet34()
    rd_pths = glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth")
    rd_pths = [p for p in rd_pths if 'fusion' not in p]
    if rd_pths:
        rd_ckpt = torch.load(rd_pths[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
        print(f"  RD模型: {rd_pths[0]}")
    rd_model.to(device).eval()
    
    # ==================== 加载Track模型 ====================
    track_model = TrackOnlyNetV4(stats_dim=stats_dim)
    track_model.load_state_dict(v15_ckpt['track_model'])
    track_model.to(device).eval()
    
    # ==================== 加载数据（使用项目中的加载器）====================
    print(f"\n加载数据...")
    val_ds = FusionDataLoaderV4(RD_VAL, TRACK_VAL, val=True, stats_dim=stats_dim)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # ==================== 验证第一个batch ====================
    print(f"\n验证数据加载...")
    for x_rd, x_track, x_stats, y in val_loader:
        print(f"  x_rd shape: {x_rd.shape}")
        print(f"  x_track shape: {x_track.shape}")
        print(f"  x_stats shape: {x_stats.shape}")
        print(f"  x_track sum: {x_track.sum().item():.4f}")
        print(f"  x_stats[0] 前10维: {x_stats[0, :10].numpy()}")
        print(f"  x_stats[0] 新增特征(21-28): {x_stats[0, 20:28].numpy()}")
        print(f"  labels: {y[:8].tolist()}")
        
        if x_track.sum().item() == 0:
            print("\n⚠️ 警告: track数据全零！数据加载可能有问题！")
        break
    
    # ==================== 测试 ====================
    print(f"\n开始测试...")
    
    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'high_correct': 0, 'high_total': 0, 'low_correct': 0, 'low_total': 0})
    all_preds = []
    all_labels = []
    all_confs = []
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            
            rd_probs = torch.softmax(rd_model(x_rd), dim=1)
            track_probs = torch.softmax(track_model(x_track, x_stats), dim=1)
            fused_probs = rd_weight * rd_probs + track_weight * track_probs
            conf, pred = fused_probs.max(dim=1)
            
            for i in range(len(y)):
                true_label = y[i].item()
                if true_label not in VALID_CLASSES:
                    continue
                
                all_preds.append(pred[i].item())
                all_labels.append(true_label)
                all_confs.append(conf[i].item())
                
                is_correct = pred[i].item() == true_label
                is_high_conf = conf[i].item() >= 0.5
                
                class_stats[true_label]['total'] += 1
                if is_correct:
                    class_stats[true_label]['correct'] += 1
                
                if is_high_conf:
                    class_stats[true_label]['high_total'] += 1
                    if is_correct:
                        class_stats[true_label]['high_correct'] += 1
                else:
                    class_stats[true_label]['low_total'] += 1
                    if is_correct:
                        class_stats[true_label]['low_correct'] += 1
    
    # ==================== 输出结果 ====================
    print("\n" + "="*70)
    print("测试结果")
    print("="*70)
    
    total_samples = len(all_preds)
    total_correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    high_conf_total = sum(1 for c in all_confs if c >= 0.5)
    high_conf_correct = sum(1 for p, l, c in zip(all_preds, all_labels, all_confs) if c >= 0.5 and p == l)
    low_conf_total = total_samples - high_conf_total
    low_conf_correct = total_correct - high_conf_correct
    
    print(f"\n总样本数: {total_samples}")
    print(f"总体准确率: {100*total_correct/total_samples:.2f}%")
    print(f"\n高置信度 (conf >= 0.5):")
    print(f"  样本数: {high_conf_total} ({100*high_conf_total/total_samples:.1f}%)")
    print(f"  准确率: {100*high_conf_correct/max(high_conf_total,1):.2f}%")
    print(f"\n低置信度 (conf < 0.5):")
    print(f"  样本数: {low_conf_total} ({100*low_conf_total/total_samples:.1f}%)")
    print(f"  准确率: {100*low_conf_correct/max(low_conf_total,1):.2f}%")
    
    # 分类别
    print(f"\n{'类别':^12}|{'总数':^8}|{'准确率':^10}|{'高置信度数':^12}|{'高置信度准确率':^14}")
    print("-" * 65)
    
    for c in VALID_CLASSES:
        stats = class_stats[c]
        acc = 100 * stats['correct'] / max(stats['total'], 1)
        high_acc = 100 * stats['high_correct'] / max(stats['high_total'], 1)
        print(f"{CLASS_NAMES[c]:^12}|{stats['total']:^8}|{acc:^10.2f}%|{stats['high_total']:^12}|{high_acc:^14.2f}%")
    
    # 预测分布
    print(f"\n预测分布:")
    pred_counts = defaultdict(int)
    for p in all_preds:
        pred_counts[p] += 1
    for c in sorted(pred_counts.keys()):
        name = CLASS_NAMES.get(c, f"类别{c}")
        print(f"  {name}: {pred_counts[c]}个 ({100*pred_counts[c]/total_samples:.1f}%)")
    
    # 置信度分布
    print(f"\n置信度分布:")
    conf_bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    for low, high in conf_bins:
        count = sum(1 for c in all_confs if low <= c < high)
        print(f"  [{low:.1f}, {high:.1f}): {count}个 ({100*count/total_samples:.1f}%)")
    
    # 与V14对比（如果有）
    print(f"\n" + "="*70)
    print("与训练时对比")
    print("="*70)
    print(f"\n训练时记录的最佳准确率: {v15_ckpt.get('best_acc', 'N/A')}")
    print(f"当前测试高置信度准确率: {100*high_conf_correct/max(high_conf_total,1):.2f}%")
    
    if abs(100*high_conf_correct/max(high_conf_total,1) - v15_ckpt.get('best_acc', 0)) < 1.0:
        print("\n✓ 测试结果与训练结果一致！")
    else:
        print("\n⚠️ 测试结果与训练结果有差异，请检查数据加载！")


if __name__ == '__main__':
    main()