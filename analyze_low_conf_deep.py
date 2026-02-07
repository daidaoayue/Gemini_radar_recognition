"""
低置信度样本深度分析
====================
目标：找出低置信度样本的根本原因，指导后续优化方向

分析内容：
1. RD和Track模型分别的表现
2. 低置信度样本的混淆模式
3. 航迹特征对比
4. 问题航迹分析
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
import re

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
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球', 4: '杂波', 5: '其它'}
    
    print("="*70)
    print("低置信度样本深度分析")
    print("="*70)
    
    # 加载数据
    print("\n加载数据...")
    val_ds = FusionDataLoaderV3(RD_VAL, TRACK_VAL, val=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # 加载模型
    print("加载模型...")
    
    # checkpoint
    v14_pths = glob.glob("./checkpoint/fusion_v14*/ckpt_best*.pth")
    if not v14_pths:
        v14_pths = glob.glob("./checkpoint/fusion_v13*/ckpt_best*.pth")
    CKPT_PATH = sorted(v14_pths)[-1]
    
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    track_weight = ckpt.get('best_fixed_weight', ckpt.get('track_weight', 0.45))
    rd_weight = 1.0 - track_weight
    
    # RD模型
    rd_model = rsnet34()
    rd_pths = glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth")
    rd_pths = [p for p in rd_pths if 'fusion' not in p]
    if rd_pths:
        rd_ckpt = torch.load(rd_pths[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
    rd_model.to(device).eval()
    
    # Track模型
    track_model = TrackOnlyNetV3()
    track_model.load_state_dict(ckpt['track_model'])
    track_model.to(device).eval()
    
    print(f"融合权重: RD={rd_weight:.2f}, Track={track_weight:.2f}")
    
    # 收集所有样本信息
    print("\n收集样本信息...")
    
    all_samples = []
    sample_idx = 0
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd_dev = x_rd.to(device)
            x_track_dev = x_track.to(device)
            x_stats_dev = x_stats.to(device)
            
            rd_logits = rd_model(x_rd_dev)
            rd_probs = torch.softmax(rd_logits, dim=1)
            rd_conf, rd_pred = rd_probs.max(dim=1)
            
            track_logits = track_model(x_track_dev, x_stats_dev)
            track_probs = torch.softmax(track_logits, dim=1)
            track_conf, track_pred = track_probs.max(dim=1)
            
            fused_probs = rd_weight * rd_probs + track_weight * track_probs
            fused_conf, fused_pred = fused_probs.max(dim=1)
            
            for i in range(len(y)):
                true_label = y[i].item()
                if true_label not in VALID_CLASSES:
                    sample_idx += 1
                    continue
                
                # 获取文件信息
                if sample_idx < len(val_ds.samples):
                    _, rd_path, _ = val_ds.samples[sample_idx]
                    filename = os.path.basename(rd_path)
                    match = re.match(r'Track(\d+)_Label(\d+)_Group(\d+)', filename)
                    track_id = int(match.group(1)) if match else 0
                else:
                    filename = f"sample_{sample_idx}"
                    track_id = 0
                
                all_samples.append({
                    'idx': sample_idx,
                    'filename': filename,
                    'track_id': track_id,
                    'true_label': true_label,
                    'fused_pred': fused_pred[i].item(),
                    'fused_conf': fused_conf[i].item(),
                    'rd_pred': rd_pred[i].item(),
                    'rd_conf': rd_conf[i].item(),
                    'track_pred': track_pred[i].item(),
                    'track_conf': track_conf[i].item(),
                    'stats': x_stats[i].cpu().numpy(),
                })
                sample_idx += 1
    
    # 分离高/低置信度
    high_conf = [s for s in all_samples if s['fused_conf'] >= 0.5]
    low_conf = [s for s in all_samples if s['fused_conf'] < 0.5]
    
    print(f"\n总样本: {len(all_samples)}")
    print(f"高置信度: {len(high_conf)} ({100*len(high_conf)/len(all_samples):.1f}%)")
    print(f"低置信度: {len(low_conf)} ({100*len(low_conf)/len(all_samples):.1f}%)")
    
    # ========== 分析1: RD vs Track ==========
    print("\n" + "="*70)
    print("分析1: RD vs Track 对低置信度样本的表现")
    print("="*70)
    
    rd_correct = sum(1 for s in low_conf if s['rd_pred'] == s['true_label'])
    track_correct = sum(1 for s in low_conf if s['track_pred'] == s['true_label'])
    fused_correct = sum(1 for s in low_conf if s['fused_pred'] == s['true_label'])
    
    both_correct = sum(1 for s in low_conf if s['rd_pred'] == s['true_label'] and s['track_pred'] == s['true_label'])
    both_wrong = sum(1 for s in low_conf if s['rd_pred'] != s['true_label'] and s['track_pred'] != s['true_label'])
    rd_only = sum(1 for s in low_conf if s['rd_pred'] == s['true_label'] and s['track_pred'] != s['true_label'])
    track_only = sum(1 for s in low_conf if s['rd_pred'] != s['true_label'] and s['track_pred'] == s['true_label'])
    
    print(f"""
单模型准确率:
  RD模型:    {rd_correct}/{len(low_conf)} = {100*rd_correct/len(low_conf):.1f}%
  Track模型: {track_correct}/{len(low_conf)} = {100*track_correct/len(low_conf):.1f}%
  融合模型:  {fused_correct}/{len(low_conf)} = {100*fused_correct/len(low_conf):.1f}%

两模型一致性分析:
  ✓ 都正确:       {both_correct:3d}个 ({100*both_correct/len(low_conf):5.1f}%)
  ✗ 都错误:       {both_wrong:3d}个 ({100*both_wrong/len(low_conf):5.1f}%) ← 最难样本
  ◐ 只有RD正确:   {rd_only:3d}个 ({100*rd_only/len(low_conf):5.1f}%)
  ◑ 只有Track正确: {track_only:3d}个 ({100*track_only/len(low_conf):5.1f}%)
""")
    
    # ========== 分析2: 混淆模式 ==========
    print("\n" + "="*70)
    print("分析2: 低置信度样本的混淆模式")
    print("="*70)
    
    confusion_pairs = defaultdict(int)
    for s in low_conf:
        if s['fused_pred'] != s['true_label']:
            confusion_pairs[(s['true_label'], s['fused_pred'])] += 1
    
    print(f"\n主要混淆对（真实→预测）:")
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: -x[1])
    for (true_c, pred_c), count in sorted_pairs[:10]:
        true_name = CLASS_NAMES.get(true_c, f'{true_c}')
        pred_name = CLASS_NAMES.get(pred_c, f'{pred_c}')
        print(f"  {true_name:8s} → {pred_name:8s}: {count}个")
    
    # ========== 分析3: 问题航迹 ==========
    print("\n" + "="*70)
    print("分析3: 低置信度样本的航迹聚集")
    print("="*70)
    
    track_low = defaultdict(list)
    for s in low_conf:
        track_low[s['track_id']].append(s)
    
    problem_tracks = [(tid, samps) for tid, samps in track_low.items() if len(samps) >= 2]
    problem_tracks.sort(key=lambda x: -len(x[1]))
    
    print(f"\n低置信度样本分布在 {len(track_low)} 条航迹")
    print(f"其中 {len(problem_tracks)} 条航迹有≥2个低置信度样本")
    
    if problem_tracks:
        print(f"\n问题航迹（低置信度样本≥2）:")
        print(f"{'航迹ID':^10}|{'低置信度':^8}|{'真实类别':^10}|{'RD对':^6}|{'Track对':^8}")
        print("-" * 50)
        
        for track_id, samples in problem_tracks[:15]:
            true_label = samples[0]['true_label']
            rd_right = sum(1 for s in samples if s['rd_pred'] == s['true_label'])
            track_right = sum(1 for s in samples if s['track_pred'] == s['true_label'])
            print(f"{track_id:^10}|{len(samples):^8}|{CLASS_NAMES[true_label]:^10}|{rd_right:^6}|{track_right:^8}")
    
    # ========== 分析4: 按类别分析 ==========
    print("\n" + "="*70)
    print("分析4: 各类别低置信度样本分析")
    print("="*70)
    
    for c in VALID_CLASSES:
        c_samples = [s for s in low_conf if s['true_label'] == c]
        if not c_samples:
            continue
        
        c_rd_correct = sum(1 for s in c_samples if s['rd_pred'] == c)
        c_track_correct = sum(1 for s in c_samples if s['track_pred'] == c)
        c_fused_correct = sum(1 for s in c_samples if s['fused_pred'] == c)
        
        print(f"\n{CLASS_NAMES[c]} ({len(c_samples)}个低置信度样本):")
        print(f"  RD准确率:    {100*c_rd_correct/len(c_samples):.1f}%")
        print(f"  Track准确率: {100*c_track_correct/len(c_samples):.1f}%")
        print(f"  融合准确率:  {100*c_fused_correct/len(c_samples):.1f}%")
        
        # 被误判为什么
        wrong = [s for s in c_samples if s['fused_pred'] != c]
        if wrong:
            pred_dist = defaultdict(int)
            for s in wrong:
                pred_dist[s['fused_pred']] += 1
            print(f"  被误判为: ", end='')
            for pred, cnt in sorted(pred_dist.items(), key=lambda x: -x[1])[:3]:
                print(f"{CLASS_NAMES.get(pred, pred)}({cnt})", end=' ')
            print()
    
    # ========== 结论 ==========
    print("\n" + "="*70)
    print("结论与建议")
    print("="*70)
    
    # 判断哪个模型更好
    if rd_correct > track_correct:
        better_model = "RD"
        worse_model = "Track"
        advantage = rd_correct - track_correct
    else:
        better_model = "Track"
        worse_model = "RD"
        advantage = track_correct - rd_correct
    
    # 可挽救的样本
    rescuable = rd_only + track_only
    
    print(f"""
核心发现:
  1. {better_model}模型对低置信度样本表现更好（多对{advantage}个）
  
  2. 可挽救样本: {rescuable}个
     - 只有RD正确: {rd_only}个
     - 只有Track正确: {track_only}个
     → 如果能正确选择信任哪个模型，可提高{100*rescuable/len(low_conf):.1f}%
  
  3. 无法挽救的样本: {both_wrong}个
     - 两个模型都判断错误
     - 需要从特征层面改进或检查数据标注

优化方案:

[方案A] 二次判断器（推荐）
  当融合置信度 < 0.5 时：
  - 比较RD置信度和Track置信度
  - 信任置信度更高的模型
  - 或训练专门区分混淆类别的二分类器
  预期效果: 可挽救约 {rescuable} 个样本 (+{100*rescuable/len(all_samples):.1f}%覆盖率)

[方案B] 动态权重
  对低置信度样本，增加{better_model}权重
  当前: RD={rd_weight:.2f}, Track={track_weight:.2f}
  建议: RD={0.7 if better_model=='RD' else 0.3:.2f}, Track={0.7 if better_model=='Track' else 0.3:.2f}

[方案C] 特征增强（根本解决）
  针对问题航迹分析：
  - 共{len(problem_tracks)}条问题航迹
  - 检查这些航迹的原始数据特点
  - 可能原因：点数少、信号弱、运动特征不明显
""")


if __name__ == '__main__':
    main()