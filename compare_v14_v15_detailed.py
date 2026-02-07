"""
V14 vs V15 全面对比分析
========================
分析为什么V15覆盖率下降，并找到最佳方案
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
from data_loader_fusion_v3 import FusionDataLoaderV3
from drsncww import rsnet34


class TrackOnlyNetV3(nn.Module):
    """V14用的20维版本"""
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


class TrackOnlyNetV4(nn.Module):
    """V15用的28维版本"""
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


def calibrate_bn(model, data_loader, device, num_batches=50):
    """校准BatchNorm"""
    model.train()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            x_track = batch[1].to(device)
            x_stats = batch[2].to(device)
            _ = model(x_track, x_stats)
            if i >= num_batches - 1:
                break
    model.eval()


def test_model_detailed(rd_model, track_model, val_loader, rd_weight, track_weight, device, valid_classes):
    """详细测试，返回所有预测结果"""
    rd_model.eval()
    track_model.eval()
    
    results = []  # (true_label, pred, conf, rd_conf, track_conf)
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            
            rd_probs = torch.softmax(rd_model(x_rd), dim=1)
            track_probs = torch.softmax(track_model(x_track, x_stats), dim=1)
            fused_probs = rd_weight * rd_probs + track_weight * track_probs
            
            fused_conf, fused_pred = fused_probs.max(dim=1)
            rd_conf, _ = rd_probs.max(dim=1)
            track_conf, _ = track_probs.max(dim=1)
            
            for i in range(len(y)):
                if y[i].item() in valid_classes:
                    results.append({
                        'true': y[i].item(),
                        'pred': fused_pred[i].item(),
                        'conf': fused_conf[i].item(),
                        'rd_conf': rd_conf[i].item(),
                        'track_conf': track_conf[i].item(),
                    })
    
    return results


def analyze_results(results, thresholds=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]):
    """分析不同阈值的效果"""
    total = len(results)
    
    print(f"\n{'阈值':^8}|{'覆盖率':^10}|{'准确率':^10}|{'高置信度样本':^12}|{'错误数':^8}")
    print("-" * 55)
    
    best_f1_thresh = 0.5
    best_f1 = 0
    
    for thresh in thresholds:
        high_conf = [r for r in results if r['conf'] >= thresh]
        n_high = len(high_conf)
        n_correct = sum(1 for r in high_conf if r['pred'] == r['true'])
        n_wrong = n_high - n_correct
        
        if n_high > 0:
            acc = 100 * n_correct / n_high
            cov = 100 * n_high / total
            
            # 计算F1-like分数 (平衡准确率和覆盖率)
            f1 = 2 * (acc/100) * (cov/100) / ((acc/100) + (cov/100) + 1e-6)
            if f1 > best_f1:
                best_f1 = f1
                best_f1_thresh = thresh
            
            print(f"{thresh:^8.2f}|{cov:^10.1f}%|{acc:^10.2f}%|{n_high:^12}|{n_wrong:^8}")
    
    return best_f1_thresh


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
    
    print("="*70)
    print("V14 vs V15 全面对比分析")
    print("="*70)
    
    # 加载RD模型
    rd_model = rsnet34()
    rd_pths = [p for p in glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth") if 'fusion' not in p]
    if rd_pths:
        rd_ckpt = torch.load(rd_pths[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
    rd_model.to(device).eval()
    
    # ==================== 测试V14 ====================
    print("\n" + "="*70)
    print("V14 (20维特征) 分析")
    print("="*70)
    
    v14_pths = glob.glob("./checkpoint/fusion_v14*/ckpt_best*.pth")
    if v14_pths:
        v14_ckpt = torch.load(sorted(v14_pths)[-1], map_location='cpu')
        v14_weight = v14_ckpt.get('best_fixed_weight', 0.45)
        
        track_model_v14 = TrackOnlyNetV3()
        track_model_v14.load_state_dict(v14_ckpt['track_model'])
        track_model_v14.to(device)
        
        # 加载数据
        val_ds_v14 = FusionDataLoaderV3(
            "./dataset/train_cleandata/val",
            "./dataset/track_enhanced_cleandata/val",
            val=True
        )
        val_loader_v14 = DataLoader(val_ds_v14, batch_size=32, shuffle=False, num_workers=0)
        
        # 校准BN
        print("\n校准BatchNorm...")
        calibrate_bn(track_model_v14, val_loader_v14, device)
        
        # 测试
        results_v14 = test_model_detailed(
            rd_model, track_model_v14, val_loader_v14,
            1.0 - v14_weight, v14_weight, device, VALID_CLASSES
        )
        
        print(f"\nV14不同阈值效果 (Track权重={v14_weight}):")
        best_thresh_v14 = analyze_results(results_v14)
        
        # 置信度分布
        confs_v14 = [r['conf'] for r in results_v14]
        print(f"\nV14置信度分布:")
        print(f"  平均: {np.mean(confs_v14):.3f}, 中位数: {np.median(confs_v14):.3f}")
        print(f"  <0.5: {sum(1 for c in confs_v14 if c < 0.5)}个 ({100*sum(1 for c in confs_v14 if c < 0.5)/len(confs_v14):.1f}%)")
        print(f"  >=0.5: {sum(1 for c in confs_v14 if c >= 0.5)}个 ({100*sum(1 for c in confs_v14 if c >= 0.5)/len(confs_v14):.1f}%)")
    else:
        print("未找到V14 checkpoint")
        results_v14 = None
    
    # ==================== 测试V15 ====================
    print("\n" + "="*70)
    print("V15 (28维特征) 分析")
    print("="*70)
    
    v15_pths = glob.glob("./checkpoint/fusion_v15*/ckpt_best*.pth")
    if v15_pths:
        v15_ckpt = torch.load(sorted(v15_pths)[-1], map_location='cpu')
        v15_weight = v15_ckpt.get('best_fixed_weight', 0.5)
        stats_dim = v15_ckpt.get('stats_dim', 28)
        
        track_model_v15 = TrackOnlyNetV4(stats_dim=stats_dim)
        track_model_v15.load_state_dict(v15_ckpt['track_model'])
        track_model_v15.to(device)
        
        # 加载数据
        val_ds_v15 = FusionDataLoaderV4(
            "./dataset/train_cleandata/val",
            "./dataset/track_enhanced_v4_cleandata/val",
            val=True,
            stats_dim=stats_dim
        )
        val_loader_v15 = DataLoader(val_ds_v15, batch_size=32, shuffle=False, num_workers=0)
        
        # 校准BN
        print("\n校准BatchNorm...")
        calibrate_bn(track_model_v15, val_loader_v15, device)
        
        # 测试
        results_v15 = test_model_detailed(
            rd_model, track_model_v15, val_loader_v15,
            1.0 - v15_weight, v15_weight, device, VALID_CLASSES
        )
        
        print(f"\nV15不同阈值效果 (Track权重={v15_weight}):")
        best_thresh_v15 = analyze_results(results_v15)
        
        # 置信度分布
        confs_v15 = [r['conf'] for r in results_v15]
        print(f"\nV15置信度分布:")
        print(f"  平均: {np.mean(confs_v15):.3f}, 中位数: {np.median(confs_v15):.3f}")
        print(f"  <0.5: {sum(1 for c in confs_v15 if c < 0.5)}个 ({100*sum(1 for c in confs_v15 if c < 0.5)/len(confs_v15):.1f}%)")
        print(f"  >=0.5: {sum(1 for c in confs_v15 if c >= 0.5)}个 ({100*sum(1 for c in confs_v15 if c >= 0.5)/len(confs_v15):.1f}%)")
        
        # Track模型置信度分布
        track_confs_v15 = [r['track_conf'] for r in results_v15]
        print(f"\nV15 Track单流置信度分布:")
        print(f"  平均: {np.mean(track_confs_v15):.3f}, 中位数: {np.median(track_confs_v15):.3f}")
    else:
        print("未找到V15 checkpoint")
        results_v15 = None
    
    # ==================== 对比总结 ====================
    print("\n" + "="*70)
    print("对比总结")
    print("="*70)
    
    if results_v14 and results_v15:
        # 找到V14和V15各自的最佳配置
        def get_best_config(results, thresholds=[0.35, 0.4, 0.45, 0.5]):
            best = {'thresh': 0.5, 'acc': 0, 'cov': 0, 'f1': 0}
            for thresh in thresholds:
                high_conf = [r for r in results if r['conf'] >= thresh]
                if not high_conf:
                    continue
                acc = 100 * sum(1 for r in high_conf if r['pred'] == r['true']) / len(high_conf)
                cov = 100 * len(high_conf) / len(results)
                f1 = 2 * (acc/100) * (cov/100) / ((acc/100) + (cov/100) + 1e-6)
                if f1 > best['f1']:
                    best = {'thresh': thresh, 'acc': acc, 'cov': cov, 'f1': f1}
            return best
        
        best_v14 = get_best_config(results_v14)
        best_v15 = get_best_config(results_v15)
        
        print(f"""
┌────────────────────────────────────────────────────────────┐
│                    V14 vs V15 对比                         │
├─────────────┬───────────────────┬──────────────────────────┤
│    指标     │       V14         │          V15             │
├─────────────┼───────────────────┼──────────────────────────┤
│  最佳阈值   │   {best_v14['thresh']:.2f}            │   {best_v15['thresh']:.2f}               │
│   准确率    │   {best_v14['acc']:.2f}%          │   {best_v15['acc']:.2f}%             │
│   覆盖率    │   {best_v14['cov']:.1f}%           │   {best_v15['cov']:.1f}%              │
│  F1分数     │   {best_v14['f1']:.3f}            │   {best_v15['f1']:.3f}               │
└─────────────┴───────────────────┴──────────────────────────┘
""")
        
        # 推荐
        print("\n推荐方案:")
        if best_v14['f1'] > best_v15['f1']:
            print(f"  → 使用V14，它在准确率和覆盖率之间有更好的平衡")
            print(f"    配置：阈值={best_v14['thresh']}, 准确率={best_v14['acc']:.2f}%, 覆盖率={best_v14['cov']:.1f}%")
        else:
            print(f"  → 使用V15，它综合表现更好")
            print(f"    配置：阈值={best_v15['thresh']}, 准确率={best_v15['acc']:.2f}%, 覆盖率={best_v15['cov']:.1f}%")
        
        # V15覆盖率低的原因分析
        print("\n" + "="*70)
        print("V15覆盖率低的原因分析")
        print("="*70)
        
        # 对比两个版本的置信度分布
        bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        print(f"\n置信度分布对比:")
        print(f"{'区间':^12}|{'V14':^12}|{'V15':^12}|{'差异':^10}")
        print("-" * 50)
        
        for i in range(len(bins)-1):
            low, high = bins[i], bins[i+1]
            n_v14 = sum(1 for r in results_v14 if low <= r['conf'] < high)
            n_v15 = sum(1 for r in results_v15 if low <= r['conf'] < high)
            pct_v14 = 100 * n_v14 / len(results_v14)
            pct_v15 = 100 * n_v15 / len(results_v15)
            diff = pct_v15 - pct_v14
            mark = "↑" if diff > 5 else ("↓" if diff < -5 else "")
            print(f"[{low:.1f}, {high:.1f})|{pct_v14:^12.1f}%|{pct_v15:^12.1f}%|{diff:^+10.1f}% {mark}")


if __name__ == '__main__':
    main()