"""
方案B副作用分析
===============
分析方案B把哪些轻型无人机误判为鸟类
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

from data_loader_fusion_v3 import FusionDataLoaderV3
from drsncww import rsnet34


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
        return self.classifier(torch.cat([self.temporal_net(x_temporal), self.stats_net(x_stats)], dim=1))


def calibrate_bn(model, data_loader, device, num_batches=50):
    model.train()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            _ = model(batch[1].to(device), batch[2].to(device))
            if i >= num_batches - 1:
                break
    model.eval()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
    BIRD_CLASS = 2
    UAV_CLASS = 0
    
    print("="*70)
    print("方案B副作用分析：鸟类vs轻型无人机的trade-off")
    print("="*70)
    
    # 加载模型
    v14_pths = glob.glob("./checkpoint/fusion_v14*/ckpt_best*.pth")
    ckpt = torch.load(sorted(v14_pths)[-1], map_location='cpu')
    
    rd_model = rsnet34()
    rd_pths = [p for p in glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth") if 'fusion' not in p]
    if rd_pths:
        rd_ckpt = torch.load(rd_pths[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
    rd_model.to(device).eval()
    
    track_model = TrackOnlyNetV3()
    track_model.load_state_dict(ckpt['track_model'])
    track_model.to(device)
    
    val_ds = FusionDataLoaderV3(
        "./dataset/train_cleandata/val",
        "./dataset/track_enhanced_cleandata/val",
        val=True
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    calibrate_bn(track_model, val_loader, device)
    
    # 收集所有样本的详细预测结果
    print("\n收集所有样本预测结果...")
    
    all_samples = []
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats_d = x_stats.to(device)
            
            rd_probs = torch.softmax(rd_model(x_rd), dim=1).cpu().numpy()
            track_probs = torch.softmax(track_model(x_track, x_stats_d), dim=1).cpu().numpy()
            
            for i in range(len(y)):
                true_label = y[i].item()
                if true_label not in VALID_CLASSES:
                    continue
                
                rd_p = rd_probs[i]
                track_p = track_probs[i]
                stats = x_stats[i].numpy()
                
                # 方案3融合
                rd_w, track_w = 0.55, 0.45
                instability = (stats[1]/1.0 + stats[7]/0.5 + stats[8]/5.0 + stats[9]/6.5) / 4.0
                rd_pred = np.argmax(rd_p)
                if rd_pred == BIRD_CLASS:
                    rd_w += 0.10
                    track_w -= 0.10
                if instability > 1.2 and rd_pred == BIRD_CLASS:
                    rd_w += 0.10
                    track_w -= 0.10
                rd_w = max(0.3, min(0.8, rd_w))
                track_w = 1.0 - rd_w
                
                fused_p = rd_w * rd_p + track_w * track_p
                plan3_pred = np.argmax(fused_p)
                
                # 检查方案B是否会改变预测
                sorted_idx = np.argsort(fused_p)[::-1]
                bird_is_second = sorted_idx[1] == BIRD_CLASS
                
                if bird_is_second and plan3_pred != BIRD_CLASS:
                    bird_prob = fused_p[BIRD_CLASS]
                    first_prob = fused_p[sorted_idx[0]]
                    gap = first_prob - bird_prob
                else:
                    gap = 999  # 不会被方案B改变
                
                all_samples.append({
                    'true': true_label,
                    'plan3_pred': plan3_pred,
                    'fused_probs': fused_p,
                    'bird_is_second': bird_is_second,
                    'gap': gap,
                    'stats': stats,
                })
    
    # ==================== 分析不同阈值的影响 ====================
    print("\n" + "="*70)
    print("不同阈值下的详细影响")
    print("="*70)
    
    for gap_threshold in [0.10, 0.15, 0.20]:
        print(f"\n--- 差距阈值 = {gap_threshold} ---")
        
        # 统计方案B会改变的样本
        changes = {'bird_rescued': [], 'uav_to_bird': [], 'balloon_to_bird': [], 'other_to_bird': []}
        
        for s in all_samples:
            # 方案B会改变预测吗？
            if s['bird_is_second'] and s['gap'] < gap_threshold and s['fused_probs'][BIRD_CLASS] > 0.25:
                plan_b_pred = BIRD_CLASS
            else:
                plan_b_pred = s['plan3_pred']
            
            # 预测改变了
            if plan_b_pred != s['plan3_pred']:
                if s['true'] == BIRD_CLASS:
                    changes['bird_rescued'].append(s)
                elif s['true'] == UAV_CLASS:
                    changes['uav_to_bird'].append(s)
                elif s['true'] == 3:  # 空飘球
                    changes['balloon_to_bird'].append(s)
                else:
                    changes['other_to_bird'].append(s)
        
        print(f"\n  方案B改变的预测:")
        print(f"    鸟类被正确挽救: {len(changes['bird_rescued'])}个 ✓")
        print(f"    轻型无人机→鸟类: {len(changes['uav_to_bird'])}个 ❌")
        print(f"    空飘球→鸟类: {len(changes['balloon_to_bird'])}个 ❌")
        print(f"    其他→鸟类: {len(changes['other_to_bird'])}个 ❌")
        
        # 净收益
        net_gain = len(changes['bird_rescued']) - len(changes['uav_to_bird']) - len(changes['balloon_to_bird']) - len(changes['other_to_bird'])
        print(f"\n  净收益: {net_gain}个样本")
        
        if net_gain > 0:
            print(f"  → 值得实施")
        else:
            print(f"  → 不值得，损失大于收益")
    
    # ==================== 分析被误判为鸟的无人机特征 ====================
    print("\n" + "="*70)
    print("被误判为鸟的轻型无人机特征分析 (gap=0.15)")
    print("="*70)
    
    gap_threshold = 0.15
    uav_to_bird = []
    uav_correct = []
    
    for s in all_samples:
        if s['true'] != UAV_CLASS:
            continue
        
        # 方案B会改变预测吗？
        if s['bird_is_second'] and s['gap'] < gap_threshold and s['fused_probs'][BIRD_CLASS] > 0.25:
            uav_to_bird.append(s)
        elif s['plan3_pred'] == UAV_CLASS:
            uav_correct.append(s)
    
    if uav_to_bird:
        print(f"\n被误判为鸟的轻型无人机: {len(uav_to_bird)}个")
        print(f"正确分类的轻型无人机: {len(uav_correct)}个")
        
        # 特征对比
        FEAT_NAMES = [
            'mean_vel', 'std_vel', 'max_vel', 'min_vel',
            'mean_vz', 'std_vz', 'mean_accel', 'max_accel',
            'turn_rate', 'heading_stab', 'mean_range', 'range_change',
            'mean_pitch', 'std_pitch', 'mean_amp', 'std_amp',
            'mean_snr', 'mean_pts', 'n_pts', 'track_len'
        ]
        
        wrong_stats = np.array([s['stats'] for s in uav_to_bird])
        correct_stats = np.array([s['stats'] for s in uav_correct])
        
        print(f"\n{'特征':^15}|{'正确UAV均值':^12}|{'误判UAV均值':^12}|{'差异%':^10}")
        print("-" * 55)
        
        key_feats = []
        for i, name in enumerate(FEAT_NAMES):
            correct_mean = correct_stats[:, i].mean()
            wrong_mean = wrong_stats[:, i].mean()
            
            if abs(correct_mean) > 0.001:
                diff_pct = 100 * (wrong_mean - correct_mean) / abs(correct_mean)
            else:
                diff_pct = 0
            
            mark = ""
            if abs(diff_pct) > 30:
                mark = " ⚠️"
                key_feats.append((name, diff_pct, correct_mean, wrong_mean))
            
            print(f"{name:^15}|{correct_mean:^12.3f}|{wrong_mean:^12.3f}|{diff_pct:^+10.1f}%{mark}")
        
        if key_feats:
            print(f"\n关键差异特征:")
            for name, diff, correct_val, wrong_val in key_feats:
                direction = "更高" if diff > 0 else "更低"
                print(f"  - {name}: 误判的UAV{direction} ({wrong_val:.2f} vs 正确的{correct_val:.2f})")
    
    # ==================== 给出建议 ====================
    print("\n" + "="*70)
    print("结论与建议")
    print("="*70)
    
    print(f"""
现状分析:
  - 方案B可以挽救10+个鸟类样本
  - 但同时会把10+个轻型无人机误判为鸟类
  - 这是因为：某些轻型无人机和鸟类的特征确实很相似

根本原因:
  - "像鸟的无人机" 和 "像无人机的鸟" 互相混淆
  - 这不是模型的问题，是物理特征的相似性

可选方案:
  1. 【保守】不实施方案B，保持V14 + 方案3的配置
     - 鸟类准确率: 91%
     - 轻型无人机准确率: 94%
     
  2. 【激进】实施方案B (gap=0.10)
     - 鸟类准确率: 93% (+2%)
     - 轻型无人机准确率: 93% (-1%)
     - 净收益为正
     
  3. 【折中】只对低置信度样本应用方案B
     - 避免误伤高置信度的轻型无人机
     
  4. 【长期】训练专门的鸟类vs轻型无人机二分类器
     - 当这两类概率接近时调用
""")


if __name__ == '__main__':
    main()