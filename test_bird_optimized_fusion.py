"""
V14改进版 - 鸟类优化融合策略
==============================

核心发现:
  1. RD在鸟类上更准 (85% vs 76%)
  2. 鸟类主要和轻型无人机混淆 (50%误分类)
  3. 低置信度样本是"乱飞的鸟"，运动特征像无人机

改进策略:
  1. 对于低置信度样本，当RD预测为鸟时增加信任
  2. 当鸟类和轻型无人机概率接近时，使用特征辅助判断
  3. 利用运动稳定性特征来识别"乱飞的鸟"
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


def compute_motion_instability(track_stats):
    """
    计算运动不稳定性指数
    
    低置信度鸟类的特征:
      - turn_rate高 (index 8)
      - heading_stab高 (index 9)
      - std_vel高 (index 1)
      - max_accel高 (index 7)
    
    返回: 不稳定性分数 (越高越不稳定)
    """
    # 提取特征
    std_vel = track_stats[:, 1]       # 速度标准差
    max_accel = track_stats[:, 7]     # 最大加速度
    turn_rate = track_stats[:, 8]     # 转弯率
    heading_stab = track_stats[:, 9]  # 航向不稳定性
    
    # 标准化（使用经验值）
    # 高置信度鸟类的典型值: std_vel~0.9, max_accel~0.5, turn_rate~5, heading_stab~6.5
    # 低置信度鸟类的典型值: std_vel~1.7, max_accel~1.2, turn_rate~13, heading_stab~19
    
    norm_std_vel = std_vel / 1.0
    norm_max_accel = max_accel / 0.5
    norm_turn_rate = turn_rate / 5.0
    norm_heading_stab = heading_stab / 6.5
    
    # 综合不稳定性指数
    instability = (norm_std_vel + norm_max_accel + norm_turn_rate + norm_heading_stab) / 4.0
    
    return instability


class BirdOptimizedFusion:
    """
    鸟类优化的融合策略
    
    策略:
    1. 基础融合: rd_weight * rd_probs + track_weight * track_probs
    2. 鸟类增强: 当RD预测为鸟类且置信度低时，增加RD权重
    3. 运动辅助: 利用运动不稳定性来辅助判断"乱飞的鸟"
    """
    
    def __init__(self, rd_weight=0.55, track_weight=0.45, 
                 bird_rd_boost=0.15, instability_threshold=1.5):
        """
        Args:
            rd_weight: 基础RD权重
            track_weight: 基础Track权重
            bird_rd_boost: 当可能是鸟时，增加的RD权重
            instability_threshold: 不稳定性阈值，超过此值认为是"乱飞"
        """
        self.rd_weight = rd_weight
        self.track_weight = track_weight
        self.bird_rd_boost = bird_rd_boost
        self.instability_threshold = instability_threshold
        
        self.BIRD_CLASS = 2
        self.LIGHT_UAV_CLASS = 0
    
    def fuse(self, rd_probs, track_probs, track_stats):
        """
        智能融合
        
        Args:
            rd_probs: [B, 6] RD模型概率
            track_probs: [B, 6] Track模型概率
            track_stats: [B, 20] Track统计特征
        
        Returns:
            fused_probs: [B, 6] 融合后的概率
        """
        batch_size = rd_probs.shape[0]
        
        # 计算运动不稳定性
        instability = compute_motion_instability(track_stats)  # [B]
        
        # 初始化融合概率
        fused_probs = torch.zeros_like(rd_probs)
        
        for i in range(batch_size):
            rd_p = rd_probs[i]
            track_p = track_probs[i]
            inst = instability[i].item()
            
            # 获取RD和Track的预测
            rd_pred = rd_p.argmax().item()
            track_pred = track_p.argmax().item()
            
            # 基础权重
            w_rd = self.rd_weight
            w_track = self.track_weight
            
            # 策略1: 当RD预测为鸟类时，增加RD权重
            if rd_pred == self.BIRD_CLASS:
                # RD在鸟类上更准，增加信任
                w_rd += self.bird_rd_boost * 0.5
                w_track -= self.bird_rd_boost * 0.5
            
            # 策略2: 当运动不稳定且RD预测为鸟时，更信任RD
            # 因为"乱飞的鸟"容易被Track误判为无人机
            if inst > self.instability_threshold and rd_pred == self.BIRD_CLASS:
                w_rd += self.bird_rd_boost * 0.5
                w_track -= self.bird_rd_boost * 0.5
            
            # 策略3: 当鸟类和轻型无人机概率接近时
            bird_prob = (w_rd * rd_p[self.BIRD_CLASS] + w_track * track_p[self.BIRD_CLASS]).item()
            uav_prob = (w_rd * rd_p[self.LIGHT_UAV_CLASS] + w_track * track_p[self.LIGHT_UAV_CLASS]).item()
            
            if abs(bird_prob - uav_prob) < 0.1:  # 概率非常接近
                # 如果运动不稳定，更可能是鸟（鸟会乱飞，无人机相对稳定）
                if inst > self.instability_threshold:
                    w_rd += 0.05  # 轻微增加RD权重（RD更擅长鸟类）
                    w_track -= 0.05
            
            # 确保权重在合理范围
            w_rd = max(0.3, min(0.8, w_rd))
            w_track = 1.0 - w_rd
            
            # 融合
            fused_probs[i] = w_rd * rd_p + w_track * track_p
        
        return fused_probs


def test_fusion_strategy(rd_model, track_model, val_loader, fusion_strategy, device, valid_classes):
    """测试融合策略"""
    rd_model.eval()
    track_model.eval()
    
    all_results = []
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            
            rd_probs = torch.softmax(rd_model(x_rd), dim=1)
            track_probs = torch.softmax(track_model(x_track, x_stats), dim=1)
            
            # 使用融合策略
            fused_probs = fusion_strategy.fuse(rd_probs, track_probs, x_stats)
            
            conf, pred = fused_probs.max(dim=1)
            
            for i in range(len(y)):
                if y[i].item() in valid_classes:
                    all_results.append({
                        'true': y[i].item(),
                        'pred': pred[i].item(),
                        'conf': conf[i].item()
                    })
    
    return all_results


def evaluate_results(results, class_names, valid_classes):
    """评估结果"""
    total = len(results)
    correct = sum(1 for r in results if r['pred'] == r['true'])
    
    # 高置信度
    high_conf = [r for r in results if r['conf'] >= 0.5]
    high_correct = sum(1 for r in high_conf if r['pred'] == r['true'])
    
    # 分类别
    class_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'high_total': 0, 'high_correct': 0})
    for r in results:
        class_stats[r['true']]['total'] += 1
        if r['pred'] == r['true']:
            class_stats[r['true']]['correct'] += 1
        if r['conf'] >= 0.5:
            class_stats[r['true']]['high_total'] += 1
            if r['pred'] == r['true']:
                class_stats[r['true']]['high_correct'] += 1
    
    print(f"\n总体: 准确率={100*correct/total:.2f}%, 高置信度={100*len(high_conf)/total:.1f}%, 高置信度准确率={100*high_correct/len(high_conf):.2f}%")
    
    print(f"\n{'类别':^12}|{'全部准确率':^12}|{'高置信度准确率':^14}|{'覆盖率':^10}")
    print("-" * 55)
    
    for c in valid_classes:
        s = class_stats[c]
        acc_all = 100 * s['correct'] / max(s['total'], 1)
        acc_high = 100 * s['high_correct'] / max(s['high_total'], 1)
        cov = 100 * s['high_total'] / max(s['total'], 1)
        print(f"{class_names[c]:^12}|{acc_all:^12.2f}%|{acc_high:^14.2f}%|{cov:^10.1f}%")
    
    return class_stats


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
    
    print("="*70)
    print("V14改进版 - 鸟类优化融合策略测试")
    print("="*70)
    
    # 加载模型
    v14_pths = glob.glob("./checkpoint/fusion_v14*/ckpt_best*.pth")
    ckpt = torch.load(sorted(v14_pths)[-1], map_location='cpu')
    original_track_weight = ckpt.get('best_fixed_weight', 0.45)
    
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
    
    # ==================== 基线: 原始V14 ====================
    print("\n" + "="*70)
    print("基线: 原始V14融合 (RD=0.55, Track=0.45)")
    print("="*70)
    
    class OriginalFusion:
        def __init__(self, rd_weight, track_weight):
            self.rd_weight = rd_weight
            self.track_weight = track_weight
        
        def fuse(self, rd_probs, track_probs, track_stats):
            return self.rd_weight * rd_probs + self.track_weight * track_probs
    
    original_fusion = OriginalFusion(1.0 - original_track_weight, original_track_weight)
    results_original = test_fusion_strategy(rd_model, track_model, val_loader, original_fusion, device, VALID_CLASSES)
    stats_original = evaluate_results(results_original, CLASS_NAMES, VALID_CLASSES)
    
    # ==================== 方案1: 增加RD权重 ====================
    print("\n" + "="*70)
    print("方案1: 整体增加RD权重 (RD=0.60, Track=0.40)")
    print("="*70)
    
    fusion1 = OriginalFusion(0.60, 0.40)
    results1 = test_fusion_strategy(rd_model, track_model, val_loader, fusion1, device, VALID_CLASSES)
    stats1 = evaluate_results(results1, CLASS_NAMES, VALID_CLASSES)
    
    # ==================== 方案2: 鸟类优化融合 ====================
    print("\n" + "="*70)
    print("方案2: 鸟类优化融合 (基础RD=0.55, 鸟类时+0.15)")
    print("="*70)
    
    fusion2 = BirdOptimizedFusion(rd_weight=0.55, track_weight=0.45, 
                                   bird_rd_boost=0.15, instability_threshold=1.5)
    results2 = test_fusion_strategy(rd_model, track_model, val_loader, fusion2, device, VALID_CLASSES)
    stats2 = evaluate_results(results2, CLASS_NAMES, VALID_CLASSES)
    
    # ==================== 方案3: 更激进的鸟类优化 ====================
    print("\n" + "="*70)
    print("方案3: 更激进的鸟类优化 (基础RD=0.55, 鸟类时+0.20)")
    print("="*70)
    
    fusion3 = BirdOptimizedFusion(rd_weight=0.55, track_weight=0.45, 
                                   bird_rd_boost=0.20, instability_threshold=1.2)
    results3 = test_fusion_strategy(rd_model, track_model, val_loader, fusion3, device, VALID_CLASSES)
    stats3 = evaluate_results(results3, CLASS_NAMES, VALID_CLASSES)
    
    # ==================== 对比总结 ====================
    print("\n" + "="*70)
    print("对比总结")
    print("="*70)
    
    def get_bird_metrics(results, stats):
        bird_total = stats[2]['total']
        bird_correct = stats[2]['correct']
        bird_high_total = stats[2]['high_total']
        bird_high_correct = stats[2]['high_correct']
        
        overall_high = [r for r in results if r['conf'] >= 0.5]
        overall_high_correct = sum(1 for r in overall_high if r['pred'] == r['true'])
        
        return {
            'bird_all_acc': 100 * bird_correct / bird_total,
            'bird_high_acc': 100 * bird_high_correct / max(bird_high_total, 1),
            'bird_coverage': 100 * bird_high_total / bird_total,
            'overall_high_acc': 100 * overall_high_correct / len(overall_high),
            'overall_coverage': 100 * len(overall_high) / len(results)
        }
    
    m_orig = get_bird_metrics(results_original, stats_original)
    m1 = get_bird_metrics(results1, stats1)
    m2 = get_bird_metrics(results2, stats2)
    m3 = get_bird_metrics(results3, stats3)
    
    print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                            各方案对比                                      │
├─────────────┬────────────┬────────────┬────────────┬─────────────────────────┤
│    指标     │   原始V14  │   方案1    │   方案2    │   方案3                 │
│             │ RD=0.55    │ RD=0.60    │ 鸟类优化   │ 激进鸟类优化            │
├─────────────┼────────────┼────────────┼────────────┼─────────────────────────┤
│鸟类全部准确率│  {m_orig['bird_all_acc']:>6.2f}%  │  {m1['bird_all_acc']:>6.2f}%  │  {m2['bird_all_acc']:>6.2f}%  │  {m3['bird_all_acc']:>6.2f}%              │
│鸟类高置信度  │  {m_orig['bird_high_acc']:>6.2f}%  │  {m1['bird_high_acc']:>6.2f}%  │  {m2['bird_high_acc']:>6.2f}%  │  {m3['bird_high_acc']:>6.2f}%              │
│鸟类覆盖率   │  {m_orig['bird_coverage']:>6.1f}%  │  {m1['bird_coverage']:>6.1f}%  │  {m2['bird_coverage']:>6.1f}%  │  {m3['bird_coverage']:>6.1f}%              │
├─────────────┼────────────┼────────────┼────────────┼─────────────────────────┤
│总体高置信度  │  {m_orig['overall_high_acc']:>6.2f}%  │  {m1['overall_high_acc']:>6.2f}%  │  {m2['overall_high_acc']:>6.2f}%  │  {m3['overall_high_acc']:>6.2f}%              │
│总体覆盖率   │  {m_orig['overall_coverage']:>6.1f}%  │  {m1['overall_coverage']:>6.1f}%  │  {m2['overall_coverage']:>6.1f}%  │  {m3['overall_coverage']:>6.1f}%              │
└─────────────┴────────────┴────────────┴────────────┴─────────────────────────┘
""")
    
    # 推荐
    print("\n推荐:")
    best = max([
        ('原始V14', m_orig),
        ('方案1', m1),
        ('方案2', m2),
        ('方案3', m3)
    ], key=lambda x: x[1]['bird_all_acc'] + 0.5 * x[1]['overall_coverage'])
    
    print(f"  最佳方案: {best[0]}")
    print(f"  鸟类全部准确率: {best[1]['bird_all_acc']:.2f}%")
    print(f"  总体覆盖率: {best[1]['overall_coverage']:.1f}%")


if __name__ == '__main__':
    main()