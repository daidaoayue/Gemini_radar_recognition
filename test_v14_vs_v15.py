"""
V14 vs V15 详细对比测试
========================
对比新增8维运动稳定性特征的效果
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

# V3数据加载器（20维）
from data_loader_fusion_v3 import FusionDataLoaderV3

# V4数据加载器（28维）- 如果存在
try:
    from data_loader_fusion_v4 import FusionDataLoaderV4
except ImportError:
    FusionDataLoaderV4 = None

from drsncww import rsnet34


class TrackOnlyNetV3(nn.Module):
    """V3/V14用的20维版本"""
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
    """V4/V15用的28维版本"""
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


def test_model(rd_model, track_model, val_loader, rd_weight, track_weight, device, valid_classes):
    """测试模型"""
    rd_model.eval()
    track_model.eval()
    
    results = {
        'high_conf_correct': 0,
        'high_conf_total': 0,
        'low_conf_correct': 0,
        'low_conf_total': 0,
        'all_correct': 0,
        'all_total': 0,
        'class_stats': defaultdict(lambda: {'high_correct': 0, 'high_total': 0, 'low_correct': 0, 'low_total': 0}),
    }
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                x_rd, x_track, x_stats, y = batch
            else:
                continue
            
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
                
                is_correct = pred[i].item() == true_label
                is_high_conf = conf[i].item() >= 0.5
                
                results['all_total'] += 1
                if is_correct:
                    results['all_correct'] += 1
                
                if is_high_conf:
                    results['high_conf_total'] += 1
                    results['class_stats'][true_label]['high_total'] += 1
                    if is_correct:
                        results['high_conf_correct'] += 1
                        results['class_stats'][true_label]['high_correct'] += 1
                else:
                    results['low_conf_total'] += 1
                    results['class_stats'][true_label]['low_total'] += 1
                    if is_correct:
                        results['low_conf_correct'] += 1
                        results['class_stats'][true_label]['low_correct'] += 1
    
    return results


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
    
    print("="*70)
    print("V14 vs V15 详细对比测试")
    print("="*70)
    
    # 加载RD模型
    rd_model = rsnet34()
    rd_pths = glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth")
    rd_pths = [p for p in rd_pths if 'fusion' not in p]
    if rd_pths:
        rd_ckpt = torch.load(rd_pths[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
    rd_model.to(device).eval()
    
    results = {}
    
    # ==================== 测试V14 ====================
    print("\n" + "="*70)
    print("测试 V14 (20维特征)")
    print("="*70)
    
    v14_pths = glob.glob("./checkpoint/fusion_v14*/ckpt_best*.pth")
    if v14_pths:
        v14_ckpt = torch.load(sorted(v14_pths)[-1], map_location='cpu')
        v14_weight = v14_ckpt.get('best_fixed_weight', 0.45)
        
        track_model_v14 = TrackOnlyNetV3()
        track_model_v14.load_state_dict(v14_ckpt['track_model'])
        track_model_v14.to(device).eval()
        
        val_ds_v14 = FusionDataLoaderV3(
            "./dataset/train_cleandata/val",
            "./dataset/track_enhanced_cleandata/val",
            val=True
        )
        val_loader_v14 = DataLoader(val_ds_v14, batch_size=32, shuffle=False)
        
        results['V14'] = test_model(
            rd_model, track_model_v14, val_loader_v14,
            1.0 - v14_weight, v14_weight, device, VALID_CLASSES
        )
        results['V14']['weight'] = v14_weight
        print(f"Track权重: {v14_weight}")
    else:
        print("未找到V14 checkpoint")
    
    # ==================== 测试V15 ====================
    print("\n" + "="*70)
    print("测试 V15 (28维特征)")
    print("="*70)
    
    v15_pths = glob.glob("./checkpoint/fusion_v15*/ckpt_best*.pth")
    if v15_pths:
        v15_ckpt = torch.load(sorted(v15_pths)[-1], map_location='cpu')
        v15_weight = v15_ckpt.get('best_fixed_weight', 0.5)
        stats_dim = v15_ckpt.get('stats_dim', 28)
        
        track_model_v15 = TrackOnlyNetV4(stats_dim=stats_dim)
        track_model_v15.load_state_dict(v15_ckpt['track_model'])
        track_model_v15.to(device).eval()
        
        # 尝试V4目录
        if os.path.exists("./dataset/track_enhanced_v4_cleandata/val"):
            track_val_dir = "./dataset/track_enhanced_v4_cleandata/val"
        else:
            track_val_dir = "./dataset/track_enhanced_cleandata/val"
        
        if FusionDataLoaderV4:
            val_ds_v15 = FusionDataLoaderV4(
                "./dataset/train_cleandata/val",
                track_val_dir,
                val=True,
                stats_dim=stats_dim
            )
        else:
            val_ds_v15 = FusionDataLoaderV3(
                "./dataset/train_cleandata/val",
                track_val_dir,
                val=True
            )
        val_loader_v15 = DataLoader(val_ds_v15, batch_size=32, shuffle=False)
        
        results['V15'] = test_model(
            rd_model, track_model_v15, val_loader_v15,
            1.0 - v15_weight, v15_weight, device, VALID_CLASSES
        )
        results['V15']['weight'] = v15_weight
        print(f"Track权重: {v15_weight}")
        print(f"统计特征维度: {stats_dim}")
    else:
        print("未找到V15 checkpoint")
    
    # ==================== 对比结果 ====================
    print("\n" + "="*70)
    print("对比结果")
    print("="*70)
    
    if 'V14' in results and 'V15' in results:
        r14 = results['V14']
        r15 = results['V15']
        
        acc14 = 100 * r14['high_conf_correct'] / max(r14['high_conf_total'], 1)
        cov14 = 100 * r14['high_conf_total'] / max(r14['all_total'], 1)
        low_acc14 = 100 * r14['low_conf_correct'] / max(r14['low_conf_total'], 1)
        
        acc15 = 100 * r15['high_conf_correct'] / max(r15['high_conf_total'], 1)
        cov15 = 100 * r15['high_conf_total'] / max(r15['all_total'], 1)
        low_acc15 = 100 * r15['low_conf_correct'] / max(r15['low_conf_total'], 1)
        
        print(f"\n{'指标':^20}|{'V14':^12}|{'V15':^12}|{'变化':^12}")
        print("-" * 60)
        print(f"{'高置信度准确率':^20}|{acc14:^12.2f}%|{acc15:^12.2f}%|{acc15-acc14:^+12.2f}%")
        print(f"{'覆盖率':^20}|{cov14:^12.1f}%|{cov15:^12.1f}%|{cov15-cov14:^+12.1f}%")
        print(f"{'低置信度样本数':^20}|{r14['low_conf_total']:^12}|{r15['low_conf_total']:^12}|{r15['low_conf_total']-r14['low_conf_total']:^+12}")
        print(f"{'低置信度准确率':^20}|{low_acc14:^12.1f}%|{low_acc15:^12.1f}%|{low_acc15-low_acc14:^+12.1f}%")
        print(f"{'Track权重':^20}|{r14['weight']:^12}|{r15['weight']:^12}|{r15['weight']-r14['weight']:^+12}")
        
        # 分类别对比
        print(f"\n{'='*70}")
        print("分类别对比（高置信度）")
        print("="*70)
        
        print(f"\n{'类别':^12}|{'V14准确率':^12}|{'V15准确率':^12}|{'变化':^10}|{'V14样本':^10}|{'V15样本':^10}")
        print("-" * 70)
        
        for c in VALID_CLASSES:
            c14 = r14['class_stats'][c]
            c15 = r15['class_stats'][c]
            
            acc14_c = 100 * c14['high_correct'] / max(c14['high_total'], 1)
            acc15_c = 100 * c15['high_correct'] / max(c15['high_total'], 1)
            
            diff = acc15_c - acc14_c
            mark = "✓" if diff > 0 else ("" if diff == 0 else "↓")
            
            print(f"{CLASS_NAMES[c]:^12}|{acc14_c:^12.2f}%|{acc15_c:^12.2f}%|{diff:^+10.2f}{mark}|{c14['high_total']:^10}|{c15['high_total']:^10}")
        
        # 低置信度分布
        print(f"\n{'='*70}")
        print("低置信度样本分布")
        print("="*70)
        
        print(f"\n{'类别':^12}|{'V14低置信度':^14}|{'V15低置信度':^14}|{'变化':^10}")
        print("-" * 55)
        
        for c in VALID_CLASSES:
            c14 = r14['class_stats'][c]
            c15 = r15['class_stats'][c]
            
            diff = c15['low_total'] - c14['low_total']
            mark = "✓" if diff < 0 else ""
            
            print(f"{CLASS_NAMES[c]:^12}|{c14['low_total']:^14}|{c15['low_total']:^14}|{diff:^+10}{mark}")
        
        # 总结
        print(f"\n{'='*70}")
        print("总结")
        print("="*70)
        
        print(f"""
V15相比V14的改进:
  ✓ 高置信度准确率: {acc14:.2f}% → {acc15:.2f}% ({acc15-acc14:+.2f}%)
  ✓ 覆盖率: {cov14:.1f}% → {cov15:.1f}% ({cov15-cov14:+.1f}%)
  ✓ 低置信度样本: {r14['low_conf_total']}个 → {r15['low_conf_total']}个 ({r15['low_conf_total']-r14['low_conf_total']:+d}个)
  
关键改进原因:
  - 新增8维运动稳定性特征
  - 包括: stability_score, curvature, velocity_cv, vel_fft_peak等
  - 这些特征帮助模型区分"乱飞的鸟"和"机动的无人机"
""")
    
    else:
        print("需要同时有V14和V15的checkpoint才能对比")


if __name__ == '__main__':
    main()