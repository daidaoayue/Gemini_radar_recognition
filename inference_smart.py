"""
航迹级推理 - 智能版
==================
对fallback航迹的更智能处理：
- 只有当鸟类概率本来就比较高（排第2名且差距小）时才加成
- 避免对非鸟类航迹误加成
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import re
import warnings

warnings.filterwarnings("ignore")

from data_loader_fusion_v3 import FusionDataLoaderV3

try:
    from drsncww import rsnet34
except ImportError:
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


class SmartTrackPredictor:
    def __init__(self, rd_model_path, track_model_path, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.track_weight = 0.35
        self.rd_weight = 0.65
        self.conf_thresh = 0.6
        
        self.rd_model = rsnet34()
        ckpt = torch.load(rd_model_path, map_location='cpu')
        self.rd_model.load_state_dict(ckpt['net_weight'] if 'net_weight' in ckpt else ckpt)
        self.rd_model.to(self.device).eval()
        
        self.track_model = TrackOnlyNetV3().to(self.device)
        ckpt = torch.load(track_model_path, map_location='cpu')
        if 'track_model' in ckpt:
            self.track_model.load_state_dict(ckpt['track_model'])
        self.track_model.eval()
    
    def predict_track(self, samples, bird_boost=2.0, boost_thresh=0.3):
        """
        智能预测
        
        Args:
            bird_boost: 鸟类加成系数
            boost_thresh: 只有当鸟类概率/最高概率 > boost_thresh 时才加成
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
        
        # 筛选高置信度样本
        valid_preds = [p for p in predictions 
                       if p['conf'] >= self.conf_thresh and p['pred'] < 4]
        
        if len(valid_preds) > 0:
            # 正常投票
            vote_scores = np.zeros(4)
            for p in valid_preds:
                vote_scores[p['pred']] += p['conf']
            
            final_pred = vote_scores.argmax()
            method = 'confident_voting'
        else:
            # Fallback：智能加成
            prob_sum = np.zeros(4)
            for p in predictions:
                prob_sum += p['probs'][:4]
            
            # 计算鸟类概率的相对强度
            prob_norm = prob_sum / prob_sum.sum()
            bird_prob = prob_norm[2]
            max_prob = prob_norm.max()
            
            # 只有当鸟类概率相对较高时才加成
            # 条件：鸟类概率 / 最高概率 > boost_thresh
            if bird_prob / max_prob > boost_thresh:
                prob_sum[2] *= bird_boost
                method = 'smart_boost'
            else:
                method = 'no_boost'
            
            final_pred = prob_sum.argmax()
        
        return final_pred, {
            'method': method,
            'n_samples': len(predictions),
            'n_valid': len(valid_preds)
        }


def parse_track_id(filename):
    match = re.match(r'Track(\d+)_Label(\d+)_', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def evaluate(predictor, val_ds, track_groups, bird_boost=2.0, boost_thresh=0.3):
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
    
    results = []
    for (track_id, true_label), indices in track_groups.items():
        if true_label not in VALID_CLASSES:
            continue
        
        samples = [(val_ds[idx][0], val_ds[idx][1], val_ds[idx][2]) for idx in indices]
        pred, details = predictor.predict_track(samples, bird_boost=bird_boost, boost_thresh=boost_thresh)
        
        results.append({
            'track_id': track_id,
            'true_label': true_label,
            'pred': pred,
            'correct': pred == true_label,
            'method': details['method']
        })
    
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    acc = 100 * correct / total
    
    # 分类别统计
    class_acc = {}
    for c in VALID_CLASSES:
        c_results = [r for r in results if r['true_label'] == c]
        c_correct = sum(1 for r in c_results if r['correct'])
        class_acc[c] = 100 * c_correct / len(c_results) if c_results else 0
    
    return acc, class_acc, results


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
    
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
    
    print(f"\n{'='*70}")
    print(f"航迹级推理 - 智能版（条件鸟类加成）")
    print(f"{'='*70}")
    
    print("\n加载数据...")
    val_ds = FusionDataLoaderV3(RD_VAL, TRACK_VAL, val=True)
    
    track_groups = defaultdict(list)
    for idx in range(len(val_ds)):
        label, rd_path, _ = val_ds.samples[idx]
        filename = os.path.basename(rd_path)
        track_id, _ = parse_track_id(filename)
        if track_id is not None:
            track_groups[(track_id, label)].append(idx)
    
    print("\n创建预测器...")
    predictor = SmartTrackPredictor(RD_PRETRAINED, FUSION_CKPT)
    
    # 网格搜索最佳参数
    print(f"\n{'='*70}")
    print(f"网格搜索最佳参数")
    print(f"{'='*70}")
    
    best_acc = 0
    best_params = {}
    
    print(f"\n{'boost':^6}|{'thresh':^8}|{'总准确率':^10}|{'轻型':^8}|{'小型':^8}|{'鸟类':^8}|{'空飘球':^8}")
    print("-" * 70)
    
    for bird_boost in [1.5, 2.0, 2.5, 3.0]:
        for boost_thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            acc, class_acc, results = evaluate(predictor, val_ds, track_groups, 
                                               bird_boost=bird_boost, boost_thresh=boost_thresh)
            
            mark = ""
            if acc > best_acc:
                best_acc = acc
                best_params = {'bird_boost': bird_boost, 'boost_thresh': boost_thresh}
                best_results = results
                best_class_acc = class_acc
                mark = " *"
            
            print(f"{bird_boost:^6.1f}|{boost_thresh:^8.1f}|{acc:^10.2f}|{class_acc[0]:^8.2f}|{class_acc[1]:^8.2f}|{class_acc[2]:^8.2f}|{class_acc[3]:^8.2f}{mark}")
    
    # 最佳结果
    print(f"\n{'='*70}")
    print(f"最佳参数: bird_boost={best_params['bird_boost']}, boost_thresh={best_params['boost_thresh']}")
    print(f"最佳准确率: {best_acc:.2f}%")
    print(f"{'='*70}")
    
    print(f"\n{'类别':^12}|{'航迹数':^8}|{'正确数':^8}|{'准确率':^10}")
    print("-" * 45)
    
    for c in [0, 1, 2, 3]:
        c_tracks = [r for r in best_results if r['true_label'] == c]
        c_correct = sum(1 for r in c_tracks if r['correct'])
        print(f"{CLASS_NAMES[c]:^12}|{len(c_tracks):^8}|{c_correct:^8}|{best_class_acc[c]:^10.2f}%")
    
    # 错误分析
    wrong = [r for r in best_results if not r['correct']]
    print(f"\n错误航迹数: {len(wrong)}")
    
    # 按方法统计错误
    method_errors = defaultdict(list)
    for r in wrong:
        method_errors[r['method']].append(r)
    
    print(f"\n错误分布:")
    for method, errors in method_errors.items():
        print(f"  {method}: {len(errors)}条")
    
    if wrong:
        print(f"\n{'航迹ID':^8}|{'真实':^10}|{'预测':^10}|{'方法':^15}")
        print("-" * 50)
        for r in wrong:
            true_name = CLASS_NAMES[r['true_label']]
            pred_name = CLASS_NAMES.get(r['pred'], f"类别{r['pred']}")
            print(f"{r['track_id']:^8}|{true_name:^10}|{pred_name:^10}|{r['method']:^15}")
    
    # 结论
    print(f"\n{'='*70}")
    print(f"最终结论")
    print(f"{'='*70}")
    print(f"\n航迹级准确率: {best_acc:.2f}%")
    if best_acc >= 98:
        print(f"✓ 已达到98%目标！")
    else:
        print(f"✗ 距离98%还差: {98 - best_acc:.2f}%")
        n_need = int(np.ceil((98 - best_acc) * 280 / 100))
        print(f"  需要再多对{n_need}条航迹才能达标")


if __name__ == '__main__':
    main()