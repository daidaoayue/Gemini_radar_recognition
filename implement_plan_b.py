"""
方案B实施：鸟类第二选择挽救策略
================================
当鸟类是第二选择且与第一名差距<0.15时，改判为鸟类

需要验证：这个策略会不会把其他类别误判为鸟类
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


class PlanBFusion:
    """
    方案B融合策略
    
    基础：方案3的鸟类优化
    新增：当鸟类是第二选择且差距小时，改判为鸟类
    """
    
    def __init__(self, gap_threshold=0.15, bird_prob_min=0.25):
        """
        Args:
            gap_threshold: 第一名和鸟类的概率差距阈值
            bird_prob_min: 鸟类最低概率要求
        """
        self.gap_threshold = gap_threshold
        self.bird_prob_min = bird_prob_min
        self.BIRD_CLASS = 2
    
    def fuse(self, rd_probs, track_probs, track_stats):
        batch_size = rd_probs.shape[0]
        final_preds = []
        final_confs = []
        rescue_count = 0
        
        for i in range(batch_size):
            rd_p = rd_probs[i].numpy() if isinstance(rd_probs, torch.Tensor) else rd_probs[i]
            track_p = track_probs[i].numpy() if isinstance(track_probs, torch.Tensor) else track_probs[i]
            stats = track_stats[i].numpy() if isinstance(track_stats, torch.Tensor) else track_stats[i]
            
            # 方案3的动态权重
            rd_w, track_w = 0.55, 0.45
            
            # 运动不稳定性
            instability = (stats[1]/1.0 + stats[7]/0.5 + stats[8]/5.0 + stats[9]/6.5) / 4.0
            
            rd_pred = np.argmax(rd_p)
            if rd_pred == self.BIRD_CLASS:
                rd_w += 0.10
                track_w -= 0.10
            if instability > 1.2 and rd_pred == self.BIRD_CLASS:
                rd_w += 0.10
                track_w -= 0.10
            
            rd_w = max(0.3, min(0.8, rd_w))
            track_w = 1.0 - rd_w
            
            # 融合
            fused_p = rd_w * rd_p + track_w * track_p
            pred = np.argmax(fused_p)
            conf = fused_p[pred]
            
            # 【方案B核心】鸟类第二选择挽救
            if pred != self.BIRD_CLASS:
                sorted_idx = np.argsort(fused_p)[::-1]
                if sorted_idx[1] == self.BIRD_CLASS:  # 鸟类是第二选择
                    bird_prob = fused_p[self.BIRD_CLASS]
                    first_prob = fused_p[sorted_idx[0]]
                    gap = first_prob - bird_prob
                    
                    # 如果差距小且鸟类概率达到最低要求
                    if gap < self.gap_threshold and bird_prob > self.bird_prob_min:
                        pred = self.BIRD_CLASS
                        conf = bird_prob
                        rescue_count += 1
            
            final_preds.append(pred)
            final_confs.append(conf)
        
        return final_preds, final_confs, rescue_count


def test_plan_b(rd_model, track_model, val_loader, device, valid_classes, gap_threshold, bird_prob_min):
    """测试方案B"""
    fusion = PlanBFusion(gap_threshold=gap_threshold, bird_prob_min=bird_prob_min)
    
    class_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'rescued': 0, 'new_errors': 0})
    total_rescued = 0
    
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
    BIRD_CLASS = 2
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats_d = x_stats.to(device)
            
            rd_probs = torch.softmax(rd_model(x_rd), dim=1).cpu()
            track_probs = torch.softmax(track_model(x_track, x_stats_d), dim=1).cpu()
            
            preds, confs, rescued = fusion.fuse(rd_probs, track_probs, x_stats)
            total_rescued += rescued
            
            for i in range(len(y)):
                true_label = y[i].item()
                if true_label not in valid_classes:
                    continue
                
                pred = preds[i]
                class_stats[true_label]['total'] += 1
                
                if pred == true_label:
                    class_stats[true_label]['correct'] += 1
    
    return class_stats, total_rescued


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
    
    print("="*70)
    print("方案B实施：鸟类第二选择挽救策略")
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
    
    # 测试不同参数
    print("\n测试不同参数组合:")
    print("="*70)
    
    params_to_test = [
        (0.10, 0.25),
        (0.12, 0.25),
        (0.15, 0.25),
        (0.15, 0.20),
        (0.18, 0.25),
        (0.20, 0.20),
    ]
    
    print(f"\n{'差距阈值':^10}|{'最低鸟概率':^10}|{'鸟类准确率':^12}|{'轻型无人机':^12}|{'空飘球':^10}|{'总体':^10}")
    print("-" * 70)
    
    best_config = None
    best_bird_acc = 0
    
    for gap_thresh, bird_min in params_to_test:
        stats, rescued = test_plan_b(rd_model, track_model, val_loader, device, 
                                      VALID_CLASSES, gap_thresh, bird_min)
        
        bird_acc = 100 * stats[2]['correct'] / stats[2]['total']
        uav_acc = 100 * stats[0]['correct'] / stats[0]['total']
        balloon_acc = 100 * stats[3]['correct'] / stats[3]['total']
        
        total_correct = sum(s['correct'] for s in stats.values())
        total_samples = sum(s['total'] for s in stats.values())
        total_acc = 100 * total_correct / total_samples
        
        print(f"{gap_thresh:^10.2f}|{bird_min:^10.2f}|{bird_acc:^12.2f}%|{uav_acc:^12.2f}%|{balloon_acc:^10.2f}%|{total_acc:^10.2f}%")
        
        # 找最佳配置（在不显著降低其他类别准确率的前提下）
        if bird_acc > best_bird_acc and uav_acc > 93 and balloon_acc > 96:
            best_bird_acc = bird_acc
            best_config = (gap_thresh, bird_min)
    
    # 详细展示最佳配置
    if best_config:
        print(f"\n" + "="*70)
        print(f"最佳配置: 差距阈值={best_config[0]}, 最低鸟概率={best_config[1]}")
        print("="*70)
        
        stats, rescued = test_plan_b(rd_model, track_model, val_loader, device,
                                      VALID_CLASSES, best_config[0], best_config[1])
        
        print(f"\n{'类别':^12}|{'总数':^8}|{'正确':^8}|{'准确率':^10}")
        print("-" * 45)
        
        for c in VALID_CLASSES:
            s = stats[c]
            acc = 100 * s['correct'] / s['total']
            print(f"{CLASS_NAMES[c]:^12}|{s['total']:^8}|{s['correct']:^8}|{acc:^10.2f}%")
        
        total_correct = sum(s['correct'] for s in stats.values())
        total_samples = sum(s['total'] for s in stats.values())
        print(f"{'总体':^12}|{total_samples:^8}|{total_correct:^8}|{100*total_correct/total_samples:^10.2f}%")
    
    # 与基线对比
    print(f"\n" + "="*70)
    print("与方案3基线对比")
    print("="*70)
    
    # 方案3基线
    class Plan3Fusion:
        def fuse(self, rd_probs, track_probs, track_stats):
            batch_size = rd_probs.shape[0]
            preds, confs = [], []
            for i in range(batch_size):
                rd_p = rd_probs[i].numpy() if isinstance(rd_probs, torch.Tensor) else rd_probs[i]
                track_p = track_probs[i].numpy() if isinstance(track_probs, torch.Tensor) else track_probs[i]
                stats = track_stats[i].numpy() if isinstance(track_stats, torch.Tensor) else track_stats[i]
                
                rd_w, track_w = 0.55, 0.45
                instability = (stats[1]/1.0 + stats[7]/0.5 + stats[8]/5.0 + stats[9]/6.5) / 4.0
                rd_pred = np.argmax(rd_p)
                if rd_pred == 2:
                    rd_w += 0.10
                    track_w -= 0.10
                if instability > 1.2 and rd_pred == 2:
                    rd_w += 0.10
                    track_w -= 0.10
                rd_w = max(0.3, min(0.8, rd_w))
                track_w = 1.0 - rd_w
                fused_p = rd_w * rd_p + track_w * track_p
                preds.append(np.argmax(fused_p))
                confs.append(fused_p[preds[-1]])
            return preds, confs, 0
    
    # 测试方案3基线
    baseline_fusion = Plan3Fusion()
    baseline_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            rd_probs = torch.softmax(rd_model(x_rd.to(device)), dim=1).cpu()
            track_probs = torch.softmax(track_model(x_track.to(device), x_stats.to(device)), dim=1).cpu()
            preds, confs, _ = baseline_fusion.fuse(rd_probs, track_probs, x_stats)
            
            for i in range(len(y)):
                if y[i].item() in VALID_CLASSES:
                    baseline_stats[y[i].item()]['total'] += 1
                    if preds[i] == y[i].item():
                        baseline_stats[y[i].item()]['correct'] += 1
    
    print(f"\n{'':^12}|{'方案3':^15}|{'方案B':^15}|{'改进':^10}")
    print("-" * 55)
    
    if best_config:
        for c in VALID_CLASSES:
            base_acc = 100 * baseline_stats[c]['correct'] / baseline_stats[c]['total']
            new_acc = 100 * stats[c]['correct'] / stats[c]['total']
            diff = new_acc - base_acc
            mark = "↑" if diff > 0 else ("↓" if diff < 0 else "")
            print(f"{CLASS_NAMES[c]:^12}|{base_acc:^15.2f}%|{new_acc:^15.2f}%|{diff:^+10.2f}% {mark}")


if __name__ == '__main__':
    main()