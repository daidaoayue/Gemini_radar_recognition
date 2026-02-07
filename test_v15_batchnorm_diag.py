"""
V15 测试 - 诊断BatchNorm问题
=============================
测试train模式 vs eval模式
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


def test_with_mode(track_model, rd_model, val_loader, rd_weight, track_weight, device, valid_classes, use_train_mode=False):
    """测试模型"""
    rd_model.eval()
    
    if use_train_mode:
        track_model.train()  # 使用train模式（BatchNorm使用batch统计量）
    else:
        track_model.eval()   # 使用eval模式（BatchNorm使用running统计量）
    
    correct_high = 0
    total_high = 0
    total_all = 0
    
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
                if y[i].item() not in valid_classes:
                    continue
                total_all += 1
                if conf[i].item() >= 0.5:
                    total_high += 1
                    if pred[i].item() == y[i].item():
                        correct_high += 1
    
    acc = 100 * correct_high / max(total_high, 1)
    cov = 100 * total_high / total_all
    return acc, cov, total_high


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VALID_CLASSES = [0, 1, 2, 3]
    
    print("="*70)
    print("V15 BatchNorm诊断测试")
    print("="*70)
    
    # 加载模型
    v15_pths = glob.glob("./checkpoint/fusion_v15*/ckpt_best*.pth")
    ckpt = torch.load(sorted(v15_pths)[-1], map_location='cpu')
    track_weight = ckpt.get('best_fixed_weight', 0.5)
    rd_weight = 1.0 - track_weight
    stats_dim = ckpt.get('stats_dim', 28)
    
    print(f"Checkpoint训练准确率: {ckpt.get('best_acc', 'N/A'):.2f}%")
    
    rd_model = rsnet34()
    rd_pths = [p for p in glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth") if 'fusion' not in p]
    if rd_pths:
        rd_ckpt = torch.load(rd_pths[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
    rd_model.to(device).eval()
    
    track_model = TrackOnlyNetV4(stats_dim=stats_dim)
    track_model.load_state_dict(ckpt['track_model'])
    track_model.to(device)
    
    # ==================== 测试1: 用训练集测试 ====================
    print("\n" + "="*70)
    print("测试1: 用训练集数据测试（应该接近训练准确率）")
    print("="*70)
    
    train_ds = FusionDataLoaderV4(
        "./dataset/train_cleandata/train",
        "./dataset/track_enhanced_v4_cleandata/train",
        val=True,  # 不做数据增强
        stats_dim=stats_dim
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
    
    acc_train_eval, cov_train_eval, _ = test_with_mode(
        track_model, rd_model, train_loader, rd_weight, track_weight, device, VALID_CLASSES, use_train_mode=False)
    print(f"训练集 + eval模式: 准确率={acc_train_eval:.2f}%, 覆盖率={cov_train_eval:.1f}%")
    
    acc_train_train, cov_train_train, _ = test_with_mode(
        track_model, rd_model, train_loader, rd_weight, track_weight, device, VALID_CLASSES, use_train_mode=True)
    print(f"训练集 + train模式: 准确率={acc_train_train:.2f}%, 覆盖率={cov_train_train:.1f}%")
    
    # ==================== 测试2: 用验证集测试 ====================
    print("\n" + "="*70)
    print("测试2: 用验证集数据测试")
    print("="*70)
    
    val_ds = FusionDataLoaderV4(
        "./dataset/train_cleandata/val",
        "./dataset/track_enhanced_v4_cleandata/val",
        val=True,
        stats_dim=stats_dim
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    acc_val_eval, cov_val_eval, _ = test_with_mode(
        track_model, rd_model, val_loader, rd_weight, track_weight, device, VALID_CLASSES, use_train_mode=False)
    print(f"验证集 + eval模式: 准确率={acc_val_eval:.2f}%, 覆盖率={cov_val_eval:.1f}%")
    
    acc_val_train, cov_val_train, _ = test_with_mode(
        track_model, rd_model, val_loader, rd_weight, track_weight, device, VALID_CLASSES, use_train_mode=True)
    print(f"验证集 + train模式: 准确率={acc_val_train:.2f}%, 覆盖率={cov_val_train:.1f}%")
    
    # ==================== 诊断BatchNorm统计量 ====================
    print("\n" + "="*70)
    print("BatchNorm统计量诊断")
    print("="*70)
    
    # 检查第一个BatchNorm层的running stats
    for name, module in track_model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            print(f"\n{name}:")
            print(f"  running_mean: min={module.running_mean.min():.4f}, max={module.running_mean.max():.4f}, mean={module.running_mean.mean():.4f}")
            print(f"  running_var:  min={module.running_var.min():.6f}, max={module.running_var.max():.4f}, mean={module.running_var.mean():.4f}")
            
            # 检查是否有极端值
            if module.running_var.min() < 1e-6:
                print(f"  ⚠️ running_var有非常小的值，可能导致数值不稳定！")
            break  # 只看第一个
    
    # ==================== 结论 ====================
    print("\n" + "="*70)
    print("诊断结论")
    print("="*70)
    
    if acc_train_eval < 50:
        print("\n⚠️ 训练集+eval模式准确率很低，说明BatchNorm的running stats有问题！")
        print("   可能原因：train和val目录的数据分布差异很大")
    
    if abs(acc_train_train - ckpt.get('best_acc', 0)) < 5:
        print("\n✓ 训练集+train模式准确率接近checkpoint记录，模型本身没问题")
        print("   问题在于BatchNorm的running stats")
    
    if acc_val_train > acc_val_eval:
        print(f"\n✓ train模式比eval模式好 ({acc_val_train:.2f}% vs {acc_val_eval:.2f}%)")
        print("   建议：保存时也保存数据的均值/标准差，或者在测试前先用一些数据更新running stats")


if __name__ == '__main__':
    main()
    