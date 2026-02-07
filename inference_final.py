"""
航迹级推理 - 最终版（置信度过滤+投票）
======================================
评估逻辑：
1. 一条航迹切成多个样本
2. 每个样本预测，过滤掉置信度<阈值的样本
3. 对剩余高置信度样本进行投票
4. 如果全部被过滤，使用概率累加作为fallback

这才是比赛的正确评估方式！
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
import re
import warnings

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


class FinalTrackPredictor:
    """最终版航迹预测器"""
    
    def __init__(self, rd_model_path, track_model_path, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.track_weight = 0.35
        self.rd_weight = 0.65
        self.conf_thresh = 0.6
        
        # 加载模型
        self.rd_model = rsnet34()
        ckpt = torch.load(rd_model_path, map_location='cpu')
        self.rd_model.load_state_dict(ckpt['net_weight'] if 'net_weight' in ckpt else ckpt)
        self.rd_model.to(self.device).eval()
        
        self.track_model = TrackOnlyNetV3().to(self.device)
        ckpt = torch.load(track_model_path, map_location='cpu')
        if 'track_model' in ckpt:
            self.track_model.load_state_dict(ckpt['track_model'])
        self.track_model.eval()
    
    def predict_track(self, samples):
        """
        预测一条航迹
        """
        predictions = []
        
        for x_rd, x_track, x_stats in samples:
            with torch.no_grad():
                x_rd = x_rd.unsqueeze(0).to(self.device) if x_rd.dim() == 3 else x_rd.to(self.device)
                x_track = x_track.unsqueeze(0).to(self.device) if x_track.dim() == 2 else x_track.to(self.device)
                x_stats = x_stats.unsqueeze(0).to(self.device) if x_stats.dim() == 1 else x_stats.to(self.device)
                
                rd_probs = torch.softmax(self.rd_model(x_rd), dim=1)
                track_probs = torch.softmax(self.track_model(x_track, x_stats), dim=1)
                fused_probs = self.rd_weight * rd_probs + self.track_weight * track_probs
                
                conf, pred = fused_probs.max(dim=1)
            
            predictions.append({
                'pred': pred.item(),
                'conf': conf.item(),
                'probs': fused_probs[0].cpu().numpy()
            })
        
        # 过滤高置信度样本（只看预测为类别0-3的）
        valid_preds = [p for p in predictions 
                       if p['conf'] >= self.conf_thresh and p['pred'] < 4]
        
        if len(valid_preds) > 0:
            # 高置信度样本加权投票
            vote_scores = np.zeros(4)
            for p in valid_preds:
                vote_scores[p['pred']] += p['conf']
            
            final_pred = vote_scores.argmax()
            final_conf = vote_scores[final_pred] / len(valid_preds)
            method = 'confident_voting'
        else:
            # Fallback: 概率累加
            prob_sum = np.zeros(4)
            for p in predictions:
                prob_sum += p['probs'][:4]
            
            final_pred = prob_sum.argmax()
            final_conf = prob_sum[final_pred] / prob_sum.sum()
            method = 'prob_fallback'
        
        return final_pred, final_conf, {
            'method': method,
            'n_samples': len(predictions),
            'n_valid': len(valid_preds),
            'predictions': predictions
        }


def parse_track_id(filename):
    match = re.match(r'Track(\d+)_Label(\d+)_', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def main():
    RD_VAL = "./dataset/train/2026-1-14/val"
    TRACK_VAL = "./dataset/track_enhanced/val"
    RD_PRETRAINED = "./checkpoint/ckpt_best_3_94.08.pth"
    FUSION_CKPT = "./checkpoint/fusion_v13_final/ckpt_best_97.39.pth"
    
    import glob
    if not os.path.exists(RD_PRETRAINED):
        pths = glob.glob("./checkpoint/*94*.pth") + glob.glob("./checkpoint/*93*.pth")
        if pths: RD_PRETRAINED = pths[0]
    if not os.path.exists(FUSION_CKPT):
        pths = glob.glob("./checkpoint/fusion_v13*/ckpt_best*.pth")
        if pths: FUSION_CKPT = pths[0]
    
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
    
    print(f"\n{'='*70}")
    print(f"航迹级推理 - 最终版")
    print(f"{'='*70}")
    
    print("\n加载数据...")
    val_ds = FusionDataLoaderV3(RD_VAL, TRACK_VAL, val=True)
    
    # 按航迹分组
    track_groups = defaultdict(list)
    for idx in range(len(val_ds)):
        label, rd_path, _ = val_ds.samples[idx]
        filename = os.path.basename(rd_path)
        track_id, _ = parse_track_id(filename)
        if track_id is not None:
            track_groups[(track_id, label)].append(idx)
    
    n_valid = sum(1 for (_, label) in track_groups.keys() if label in VALID_CLASSES)
    print(f"有效航迹数（类别0-3）: {n_valid}")
    
    print("\n创建预测器...")
    predictor = FinalTrackPredictor(RD_PRETRAINED, FUSION_CKPT)
    
    print(f"\n{'='*70}")
    print(f"测试不同置信度阈值")
    print(f"{'='*70}")
    
    best_acc = 0
    best_thresh = 0
    
    for thresh in [0.5, 0.55, 0.6, 0.65, 0.7]:
        predictor.conf_thresh = thresh
        
        results = []
        fallback_count = 0
        
        for (track_id, true_label), indices in track_groups.items():
            if true_label not in VALID_CLASSES:
                continue
            
            samples = [(val_ds[idx][0], val_ds[idx][1], val_ds[idx][2]) for idx in indices]
            pred, conf, details = predictor.predict_track(samples)
            
            if details['method'] == 'prob_fallback':
                fallback_count += 1
            
            results.append({
                'track_id': track_id,
                'true_label': true_label,
                'pred': pred,
                'correct': pred == true_label,
                'n_valid': details['n_valid'],
                'n_samples': details['n_samples'],
                'method': details['method']
            })
        
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        acc = 100 * correct / total
        
        # 逐类别
        bird_tracks = [r for r in results if r['true_label'] == 2]
        bird_correct = sum(1 for r in bird_tracks if r['correct'])
        bird_acc = 100 * bird_correct / len(bird_tracks)
        
        mark = " *" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            best_results = results
        
        print(f"阈值={thresh:.2f}: 准确率={acc:.2f}%, 鸟类={bird_acc:.2f}%, fallback={fallback_count}条{mark}")
    
    # 最佳结果
    print(f"\n{'='*70}")
    print(f"最佳阈值: {best_thresh}")
    print(f"航迹级准确率: {best_acc:.2f}%")
    print(f"{'='*70}")
    
    print(f"\n{'类别':^12}|{'航迹数':^8}|{'正确数':^8}|{'准确率':^10}")
    print("-" * 45)
    
    for c in VALID_CLASSES:
        c_tracks = [r for r in best_results if r['true_label'] == c]
        c_correct = sum(1 for r in c_tracks if r['correct'])
        c_acc = 100 * c_correct / len(c_tracks) if c_tracks else 0
        print(f"{CLASS_NAMES[c]:^12}|{len(c_tracks):^8}|{c_correct:^8}|{c_acc:^10.2f}%")
    
    # 错误分析
    wrong = [r for r in best_results if not r['correct']]
    print(f"\n错误航迹数: {len(wrong)}")
    
    if wrong:
        print(f"\n{'航迹ID':^8}|{'真实':^10}|{'预测':^10}|{'样本':^6}|{'有效':^6}|{'方法':^15}")
        print("-" * 65)
        for r in wrong:
            true_name = CLASS_NAMES[r['true_label']][:6]
            pred_name = CLASS_NAMES.get(r['pred'], f"类{r['pred']}")[:6]
            print(f"{r['track_id']:^8}|{true_name:^10}|{pred_name:^10}|{r['n_samples']:^6}|{r['n_valid']:^6}|{r['method']:^15}")
    
    # 方法分析
    confident_tracks = [r for r in best_results if r['method'] == 'confident_voting']
    fallback_tracks = [r for r in best_results if r['method'] == 'prob_fallback']
    
    confident_acc = 100 * sum(1 for r in confident_tracks if r['correct']) / len(confident_tracks) if confident_tracks else 0
    fallback_acc = 100 * sum(1 for r in fallback_tracks if r['correct']) / len(fallback_tracks) if fallback_tracks else 0
    
    print(f"\n{'='*70}")
    print(f"方法分析")
    print(f"{'='*70}")
    print(f"confident_voting: {len(confident_tracks)}条航迹, 准确率={confident_acc:.2f}%")
    print(f"prob_fallback:    {len(fallback_tracks)}条航迹, 准确率={fallback_acc:.2f}%")
    
    # 结论
    print(f"\n{'='*70}")
    print(f"最终结论")
    print(f"{'='*70}")
    print(f"\n航迹级准确率: {best_acc:.2f}%")
    if best_acc >= 98:
        print(f"✓ 已达到98%目标！")
    else:
        print(f"✗ 距离98%还差: {98 - best_acc:.2f}%")


if __name__ == '__main__':
    main()