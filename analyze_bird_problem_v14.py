"""
V14 鸟类问题深度分析
====================
分析鸟类样本的误分类原因，找到改进方向

已知问题：
  - 鸟类高置信度准确率: ~94%
  - 鸟类全部准确率: ~86%
  - 低置信度样本占10%

分析目标：
  1. 鸟类被误分类到哪些类别？
  2. 哪些类别被误分类为鸟类？
  3. 低置信度鸟类样本的特点是什么？
  4. RD和Track模型在鸟类上的表现差异
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
import scipy.io as scio

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
        feat_temporal = self.temporal_net(x_temporal)
        feat_stats = self.stats_net(x_stats)
        return self.classifier(torch.cat([feat_temporal, feat_stats], dim=1))


def calibrate_bn(model, data_loader, device, num_batches=50):
    model.train()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            x_track = batch[1].to(device)
            x_stats = batch[2].to(device)
            _ = model(x_track, x_stats)
            if i >= num_batches - 1:
                break
    model.eval()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
    BIRD_CLASS = 2
    
    print("="*70)
    print("V14 鸟类问题深度分析")
    print("="*70)
    
    # 加载模型
    v14_pths = glob.glob("./checkpoint/fusion_v14*/ckpt_best*.pth")
    ckpt = torch.load(sorted(v14_pths)[-1], map_location='cpu')
    track_weight = ckpt.get('best_fixed_weight', 0.45)
    rd_weight = 1.0 - track_weight
    
    rd_model = rsnet34()
    rd_pths = [p for p in glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth") if 'fusion' not in p]
    if rd_pths:
        rd_ckpt = torch.load(rd_pths[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
    rd_model.to(device).eval()
    
    track_model = TrackOnlyNetV3()
    track_model.load_state_dict(ckpt['track_model'])
    track_model.to(device)
    
    # 加载数据
    val_ds = FusionDataLoaderV3(
        "./dataset/train_cleandata/val",
        "./dataset/track_enhanced_cleandata/val",
        val=True
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # 校准BN
    calibrate_bn(track_model, val_loader, device)
    
    # 收集所有预测结果
    print("\n收集预测结果...")
    
    all_results = []
    sample_idx = 0
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            
            rd_logits = rd_model(x_rd)
            track_logits = track_model(x_track, x_stats)
            
            rd_probs = torch.softmax(rd_logits, dim=1)
            track_probs = torch.softmax(track_logits, dim=1)
            fused_probs = rd_weight * rd_probs + track_weight * track_probs
            
            fused_conf, fused_pred = fused_probs.max(dim=1)
            rd_conf, rd_pred = rd_probs.max(dim=1)
            track_conf, track_pred = track_probs.max(dim=1)
            
            for i in range(len(y)):
                if y[i].item() in VALID_CLASSES:
                    all_results.append({
                        'idx': sample_idx,
                        'true': y[i].item(),
                        'fused_pred': fused_pred[i].item(),
                        'fused_conf': fused_conf[i].item(),
                        'rd_pred': rd_pred[i].item(),
                        'rd_conf': rd_conf[i].item(),
                        'track_pred': track_pred[i].item(),
                        'track_conf': track_conf[i].item(),
                        'rd_probs': rd_probs[i].cpu().numpy(),
                        'track_probs': track_probs[i].cpu().numpy(),
                        'fused_probs': fused_probs[i].cpu().numpy(),
                        'track_stats': x_stats[i].cpu().numpy(),
                    })
                sample_idx += 1
    
    # ==================== 1. 整体鸟类性能 ====================
    print("\n" + "="*70)
    print("1. 鸟类整体性能")
    print("="*70)
    
    bird_samples = [r for r in all_results if r['true'] == BIRD_CLASS]
    total_birds = len(bird_samples)
    
    bird_correct = sum(1 for r in bird_samples if r['fused_pred'] == BIRD_CLASS)
    bird_high_conf = [r for r in bird_samples if r['fused_conf'] >= 0.5]
    bird_low_conf = [r for r in bird_samples if r['fused_conf'] < 0.5]
    bird_high_correct = sum(1 for r in bird_high_conf if r['fused_pred'] == BIRD_CLASS)
    
    print(f"\n鸟类样本总数: {total_birds}")
    print(f"全部准确率: {100*bird_correct/total_birds:.2f}% ({bird_correct}/{total_birds})")
    print(f"高置信度样本: {len(bird_high_conf)} ({100*len(bird_high_conf)/total_birds:.1f}%)")
    print(f"高置信度准确率: {100*bird_high_correct/max(len(bird_high_conf),1):.2f}%")
    print(f"低置信度样本: {len(bird_low_conf)} ({100*len(bird_low_conf)/total_birds:.1f}%)")
    
    # ==================== 2. 鸟类误分类分析 ====================
    print("\n" + "="*70)
    print("2. 鸟类误分类分析")
    print("="*70)
    
    # 鸟类被误分类到哪里
    bird_wrong = [r for r in bird_samples if r['fused_pred'] != BIRD_CLASS]
    
    print(f"\n鸟类被误分类 ({len(bird_wrong)}个样本):")
    wrong_to = defaultdict(int)
    for r in bird_wrong:
        wrong_to[r['fused_pred']] += 1
    
    for c, count in sorted(wrong_to.items(), key=lambda x: -x[1]):
        print(f"  → 误分类为 {CLASS_NAMES.get(c, f'类别{c}')}: {count}个 ({100*count/len(bird_wrong):.1f}%)")
    
    # 哪些类别被误分类为鸟类
    print(f"\n其他类别被误分类为鸟类:")
    for true_c in [0, 1, 3]:  # 排除鸟类本身
        samples_c = [r for r in all_results if r['true'] == true_c]
        wrong_to_bird = [r for r in samples_c if r['fused_pred'] == BIRD_CLASS]
        if wrong_to_bird:
            print(f"  {CLASS_NAMES[true_c]} → 鸟类: {len(wrong_to_bird)}个")
    
    # ==================== 3. RD vs Track 在鸟类上的表现 ====================
    print("\n" + "="*70)
    print("3. RD vs Track 在鸟类上的表现")
    print("="*70)
    
    rd_bird_correct = sum(1 for r in bird_samples if r['rd_pred'] == BIRD_CLASS)
    track_bird_correct = sum(1 for r in bird_samples if r['track_pred'] == BIRD_CLASS)
    
    print(f"\n单模型鸟类准确率:")
    print(f"  RD模型:    {100*rd_bird_correct/total_birds:.2f}% ({rd_bird_correct}/{total_birds})")
    print(f"  Track模型: {100*track_bird_correct/total_birds:.2f}% ({track_bird_correct}/{total_birds})")
    print(f"  融合模型:  {100*bird_correct/total_birds:.2f}% ({bird_correct}/{total_birds})")
    
    # RD对Track不一致的情况
    print(f"\nRD与Track预测不一致的鸟类样本:")
    disagree = [r for r in bird_samples if r['rd_pred'] != r['track_pred']]
    print(f"  总数: {len(disagree)}个 ({100*len(disagree)/total_birds:.1f}%)")
    
    # 分析不一致时的情况
    rd_right_track_wrong = [r for r in disagree if r['rd_pred'] == BIRD_CLASS]
    track_right_rd_wrong = [r for r in disagree if r['track_pred'] == BIRD_CLASS]
    both_wrong = [r for r in disagree if r['rd_pred'] != BIRD_CLASS and r['track_pred'] != BIRD_CLASS]
    
    print(f"    RD对Track错: {len(rd_right_track_wrong)}个")
    print(f"    Track对RD错: {len(track_right_rd_wrong)}个")
    print(f"    两个都错:    {len(both_wrong)}个")
    
    # ==================== 4. 低置信度鸟类样本分析 ====================
    print("\n" + "="*70)
    print("4. 低置信度鸟类样本分析 (conf < 0.5)")
    print("="*70)
    
    print(f"\n低置信度鸟类样本数: {len(bird_low_conf)}")
    
    if bird_low_conf:
        # 预测分布
        low_pred_dist = defaultdict(int)
        for r in bird_low_conf:
            low_pred_dist[r['fused_pred']] += 1
        
        print(f"\n低置信度样本的预测分布:")
        for c, count in sorted(low_pred_dist.items(), key=lambda x: -x[1]):
            print(f"  预测为{CLASS_NAMES.get(c, f'类别{c}')}: {count}个 ({100*count/len(bird_low_conf):.1f}%)")
        
        # RD和Track在低置信度样本上的表现
        low_rd_correct = sum(1 for r in bird_low_conf if r['rd_pred'] == BIRD_CLASS)
        low_track_correct = sum(1 for r in bird_low_conf if r['track_pred'] == BIRD_CLASS)
        
        print(f"\n低置信度样本的单模型准确率:")
        print(f"  RD模型:    {100*low_rd_correct/len(bird_low_conf):.1f}% ({low_rd_correct}/{len(bird_low_conf)})")
        print(f"  Track模型: {100*low_track_correct/len(bird_low_conf):.1f}% ({low_track_correct}/{len(bird_low_conf)})")
        
        # 置信度分布
        print(f"\n低置信度样本的融合置信度分布:")
        conf_bins = [(0.3, 0.35), (0.35, 0.4), (0.4, 0.45), (0.45, 0.5)]
        for low, high in conf_bins:
            count = sum(1 for r in bird_low_conf if low <= r['fused_conf'] < high)
            print(f"  [{low:.2f}, {high:.2f}): {count}个")
        
        # 分析低置信度样本的特征
        print(f"\n低置信度样本的Track统计特征对比:")
        
        # 特征名称
        feat_names = [
            'mean_vel', 'std_vel', 'max_vel', 'min_vel',
            'mean_vz', 'std_vz', 'mean_accel', 'max_accel',
            'turn_rate', 'heading_stab', 'mean_range', 'range_change',
            'mean_pitch', 'std_pitch', 'mean_amp', 'std_amp',
            'mean_snr', 'mean_pts', 'n_pts', 'track_len'
        ]
        
        high_stats = np.array([r['track_stats'] for r in bird_high_conf])
        low_stats = np.array([r['track_stats'] for r in bird_low_conf])
        
        print(f"\n{'特征':^15}|{'高置信度均值':^12}|{'低置信度均值':^12}|{'差异%':^10}")
        print("-" * 55)
        
        significant_feats = []
        for i, name in enumerate(feat_names[:20]):
            high_mean = high_stats[:, i].mean() if len(high_stats) > 0 else 0
            low_mean = low_stats[:, i].mean() if len(low_stats) > 0 else 0
            
            if abs(high_mean) > 0.001:
                diff_pct = 100 * (low_mean - high_mean) / abs(high_mean)
            else:
                diff_pct = 0
            
            mark = ""
            if abs(diff_pct) > 30:
                mark = " ⚠️"
                significant_feats.append((name, diff_pct))
            
            print(f"{name:^15}|{high_mean:^12.3f}|{low_mean:^12.3f}|{diff_pct:^+10.1f}%{mark}")
        
        if significant_feats:
            print(f"\n显著差异的特征:")
            for name, diff in significant_feats:
                direction = "更高" if diff > 0 else "更低"
                print(f"  - {name}: 低置信度样本{direction} ({diff:+.1f}%)")
    
    # ==================== 5. 鸟类与其他类别的混淆分析 ====================
    print("\n" + "="*70)
    print("5. 鸟类与其他类别的混淆分析")
    print("="*70)
    
    # 鸟类最容易和哪个类别混淆
    print(f"\n鸟类样本的概率分布分析（平均概率）:")
    
    bird_probs_avg = np.mean([r['fused_probs'] for r in bird_samples], axis=0)
    for c in VALID_CLASSES:
        print(f"  P({CLASS_NAMES[c]}): {100*bird_probs_avg[c]:.2f}%")
    
    # 被误分类的鸟类样本的概率分析
    if bird_wrong:
        print(f"\n被误分类的鸟类样本的概率分布:")
        wrong_probs_avg = np.mean([r['fused_probs'] for r in bird_wrong], axis=0)
        for c in VALID_CLASSES:
            print(f"  P({CLASS_NAMES[c]}): {100*wrong_probs_avg[c]:.2f}%")
    
    # ==================== 6. 改进建议 ====================
    print("\n" + "="*70)
    print("6. 改进建议")
    print("="*70)
    
    print(f"""
基于以上分析，针对鸟类问题的改进方向：

1. 鸟类误分类去向分析：
   {'   '.join([f'{CLASS_NAMES.get(c, f"类别{c}")}: {count}个' for c, count in sorted(wrong_to.items(), key=lambda x: -x[1])])}

2. RD vs Track对比：
   - RD鸟类准确率: {100*rd_bird_correct/total_birds:.2f}%
   - Track鸟类准确率: {100*track_bird_correct/total_birds:.2f}%
   {'   → RD更擅长识别鸟类' if rd_bird_correct > track_bird_correct else '   → Track更擅长识别鸟类'}

3. 可能的改进方案：
   a) 针对鸟类增加RD权重（如果RD更准）
   b) 针对低置信度样本使用不同的融合策略
   c) 分析误分类样本的特征，添加针对性特征
   d) 对鸟类样本做数据增强或过采样
""")


if __name__ == '__main__':
    main()