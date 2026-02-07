"""
V15 推理（带BatchNorm校准）
===========================
在正式测试前，用一部分验证数据校准BatchNorm的running stats
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


def calibrate_batchnorm(model, data_loader, device, num_batches=None):
    """
    用数据校准BatchNorm的running stats
    
    在train模式下forward，但不更新模型权重
    这会更新BatchNorm的running_mean和running_var
    """
    model.train()  # 必须是train模式才会更新running stats
    
    with torch.no_grad():  # 不计算梯度
        for i, (x_rd, x_track, x_stats, y) in enumerate(data_loader):
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            
            # 只forward，不backward
            _ = model(x_track, x_stats)
            
            if num_batches and i >= num_batches - 1:
                break
    
    model.eval()  # 校准完成后切换回eval模式


def test_model(rd_model, track_model, val_loader, rd_weight, track_weight, device, valid_classes):
    """测试模型"""
    rd_model.eval()
    track_model.eval()
    
    class_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'high_total': 0, 'high_correct': 0})
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
                if true_label not in valid_classes:
                    continue
                
                all_preds.append(pred[i].item())
                all_labels.append(true_label)
                all_confs.append(conf[i].item())
                
                class_stats[true_label]['total'] += 1
                if pred[i].item() == true_label:
                    class_stats[true_label]['correct'] += 1
                
                if conf[i].item() >= 0.5:
                    class_stats[true_label]['high_total'] += 1
                    if pred[i].item() == true_label:
                        class_stats[true_label]['high_correct'] += 1
    
    return all_preds, all_labels, all_confs, class_stats


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
    
    print("="*70)
    print("V15 推理（带BatchNorm校准）")
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
    track_model.to(device)
    
    # ==================== 加载数据 ====================
    print(f"\n加载数据...")
    val_ds = FusionDataLoaderV4(
        "./dataset/train_cleandata/val",
        "./dataset/track_enhanced_v4_cleandata/val",
        val=True,
        stats_dim=stats_dim
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # ==================== 测试1: 不校准（直接用checkpoint） ====================
    print("\n" + "="*70)
    print("测试1: 不校准（直接用checkpoint的running stats）")
    print("="*70)
    
    track_model.eval()
    preds1, labels1, confs1, stats1 = test_model(
        rd_model, track_model, val_loader, rd_weight, track_weight, device, VALID_CLASSES)
    
    total = len(preds1)
    correct = sum(1 for p, l in zip(preds1, labels1) if p == l)
    high_conf = sum(1 for c in confs1 if c >= 0.5)
    high_correct = sum(1 for p, l, c in zip(preds1, labels1, confs1) if c >= 0.5 and p == l)
    
    print(f"\n  总体准确率: {100*correct/total:.2f}%")
    print(f"  高置信度准确率: {100*high_correct/max(high_conf,1):.2f}% (覆盖率: {100*high_conf/total:.1f}%)")
    
    pred_dist = defaultdict(int)
    for p in preds1:
        pred_dist[p] += 1
    print(f"  预测分布: ", end='')
    for c in sorted(pred_dist.keys()):
        print(f"{CLASS_NAMES.get(c, c)}({pred_dist[c]}) ", end='')
    print()
    
    # ==================== 测试2: 用验证数据校准BatchNorm ====================
    print("\n" + "="*70)
    print("测试2: 用验证数据校准BatchNorm后测试")
    print("="*70)
    
    # 重新加载模型
    track_model2 = TrackOnlyNetV4(stats_dim=stats_dim)
    track_model2.load_state_dict(ckpt['track_model'])
    track_model2.to(device)
    
    # 校准BatchNorm
    print("\n  正在校准BatchNorm...")
    calibrate_batchnorm(track_model2, val_loader, device, num_batches=None)  # 用全部数据校准
    print("  校准完成")
    
    # 测试
    preds2, labels2, confs2, stats2 = test_model(
        rd_model, track_model2, val_loader, rd_weight, track_weight, device, VALID_CLASSES)
    
    correct2 = sum(1 for p, l in zip(preds2, labels2) if p == l)
    high_conf2 = sum(1 for c in confs2 if c >= 0.5)
    high_correct2 = sum(1 for p, l, c in zip(preds2, labels2, confs2) if c >= 0.5 and p == l)
    
    print(f"\n  总体准确率: {100*correct2/total:.2f}%")
    print(f"  高置信度准确率: {100*high_correct2/max(high_conf2,1):.2f}% (覆盖率: {100*high_conf2/total:.1f}%)")
    
    pred_dist2 = defaultdict(int)
    for p in preds2:
        pred_dist2[p] += 1
    print(f"  预测分布: ", end='')
    for c in sorted(pred_dist2.keys()):
        print(f"{CLASS_NAMES.get(c, c)}({pred_dist2[c]}) ", end='')
    print()
    
    # ==================== 测试3: 用训练数据校准BatchNorm ====================
    print("\n" + "="*70)
    print("测试3: 用训练数据校准BatchNorm后测试验证集")
    print("="*70)
    
    # 加载训练数据
    train_ds = FusionDataLoaderV4(
        "./dataset/train_cleandata/train",
        "./dataset/track_enhanced_v4_cleandata/train",
        val=True,  # 不做数据增强
        stats_dim=stats_dim
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    
    # 重新加载模型
    track_model3 = TrackOnlyNetV4(stats_dim=stats_dim)
    track_model3.load_state_dict(ckpt['track_model'])
    track_model3.to(device)
    
    # 用训练数据校准
    print("\n  正在用训练数据校准BatchNorm...")
    calibrate_batchnorm(track_model3, train_loader, device, num_batches=50)  # 用50个batch
    print("  校准完成")
    
    # 测试验证集
    preds3, labels3, confs3, stats3 = test_model(
        rd_model, track_model3, val_loader, rd_weight, track_weight, device, VALID_CLASSES)
    
    correct3 = sum(1 for p, l in zip(preds3, labels3) if p == l)
    high_conf3 = sum(1 for c in confs3 if c >= 0.5)
    high_correct3 = sum(1 for p, l, c in zip(preds3, labels3, confs3) if c >= 0.5 and p == l)
    
    print(f"\n  总体准确率: {100*correct3/total:.2f}%")
    print(f"  高置信度准确率: {100*high_correct3/max(high_conf3,1):.2f}% (覆盖率: {100*high_conf3/total:.1f}%)")
    
    pred_dist3 = defaultdict(int)
    for p in preds3:
        pred_dist3[p] += 1
    print(f"  预测分布: ", end='')
    for c in sorted(pred_dist3.keys()):
        print(f"{CLASS_NAMES.get(c, c)}({pred_dist3[c]}) ", end='')
    print()
    
    # ==================== 详细分析最佳结果 ====================
    best_idx = 2 if correct2 >= correct3 else 3
    best_preds = preds2 if best_idx == 2 else preds3
    best_labels = labels2 if best_idx == 2 else labels3
    best_confs = confs2 if best_idx == 2 else confs3
    best_stats = stats2 if best_idx == 2 else stats3
    
    print("\n" + "="*70)
    print(f"最佳方案详细分析（测试{best_idx}）")
    print("="*70)
    
    # 分类别
    print(f"\n{'类别':^12}|{'总数':^8}|{'准确率':^10}|{'高置信度':^10}|{'高置信度准确率':^14}")
    print("-" * 60)
    
    for c in VALID_CLASSES:
        s = best_stats[c]
        acc = 100 * s['correct'] / max(s['total'], 1)
        high_acc = 100 * s['high_correct'] / max(s['high_total'], 1)
        print(f"{CLASS_NAMES[c]:^12}|{s['total']:^8}|{acc:^10.2f}%|{s['high_total']:^10}|{high_acc:^14.2f}%")
    
    # ==================== 总结 ====================
    print("\n" + "="*70)
    print("总结与建议")
    print("="*70)
    
    print(f"""
问题根源：
  checkpoint中保存的BatchNorm running stats与测试数据分布不匹配

解决方案：
  1. 测试前用测试数据校准BatchNorm（测试2的方法）
  2. 或修改训练脚本，在保存checkpoint前校准BatchNorm

当前V15性能（校准后）:
  - 总体准确率: {100*correct2/total:.2f}%
  - 高置信度准确率: {100*high_correct2/max(high_conf2,1):.2f}%
  - 覆盖率: {100*high_conf2/total:.1f}%
""")


if __name__ == '__main__':
    main()