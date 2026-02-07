"""
V13 vs V14 对比测试脚本
=======================
功能：
1. 对比V13和V14的整体效果
2. 分析各类别准确率变化
3. 分析低置信度样本的变化
4. 输出详细对比报告
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import warnings
import glob

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


def evaluate_model(rd_model, track_model, val_loader, track_weight, device, conf_thresh=0.5):
    """评估融合模型"""
    rd_model.eval()
    track_model.eval()
    
    rd_weight = 1.0 - track_weight
    
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
    VALID_CLASSES = [0, 1, 2, 3]
    
    # 收集所有结果
    all_results = []
    
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
                true_label = y[i].item()
                if true_label not in VALID_CLASSES:
                    continue
                
                all_results.append({
                    'true': true_label,
                    'pred': pred[i].item(),
                    'conf': conf[i].item(),
                    'correct': pred[i].item() == true_label
                })
    
    # 计算统计
    total = len(all_results)
    
    # 高置信度统计
    high_conf = [r for r in all_results if r['conf'] >= conf_thresh]
    high_conf_correct = sum(1 for r in high_conf if r['correct'])
    high_conf_acc = 100 * high_conf_correct / len(high_conf) if high_conf else 0
    coverage = 100 * len(high_conf) / total
    
    # 低置信度统计
    low_conf = [r for r in all_results if r['conf'] < conf_thresh]
    low_conf_correct = sum(1 for r in low_conf if r['correct'])
    low_conf_acc = 100 * low_conf_correct / len(low_conf) if low_conf else 0
    
    # 全部样本统计
    all_correct = sum(1 for r in all_results if r['correct'])
    all_acc = 100 * all_correct / total
    
    # 分类别统计（高置信度）
    class_stats = {}
    for c in VALID_CLASSES:
        c_high = [r for r in high_conf if r['true'] == c]
        c_correct = sum(1 for r in c_high if r['correct'])
        c_acc = 100 * c_correct / len(c_high) if c_high else 0
        
        c_low = [r for r in low_conf if r['true'] == c]
        
        class_stats[c] = {
            'name': CLASS_NAMES[c],
            'high_conf_total': len(c_high),
            'high_conf_correct': c_correct,
            'high_conf_acc': c_acc,
            'low_conf_total': len(c_low),
            'low_conf_correct': sum(1 for r in c_low if r['correct'])
        }
    
    return {
        'total': total,
        'high_conf_acc': high_conf_acc,
        'coverage': coverage,
        'low_conf_total': len(low_conf),
        'low_conf_acc': low_conf_acc,
        'all_acc': all_acc,
        'class_stats': class_stats,
        'all_results': all_results
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 配置
    RD_VAL = "./dataset/train_cleandata/val"
    TRACK_VAL = "./dataset/track_enhanced_cleandata/val"
    
    # 查找checkpoints
    V13_CKPT = None
    V14_CKPT = None
    
    v13_pths = glob.glob("./checkpoint/fusion_v13*/ckpt_best*.pth")
    if v13_pths:
        V13_CKPT = sorted(v13_pths)[-1]
    
    v14_pths = glob.glob("./checkpoint/fusion_v14*/ckpt_best*.pth")
    if v14_pths:
        V14_CKPT = sorted(v14_pths)[-1]
    
    print("="*70)
    print("V13 vs V14 对比测试")
    print("="*70)
    print(f"V13 checkpoint: {V13_CKPT}")
    print(f"V14 checkpoint: {V14_CKPT}")
    
    if not V13_CKPT or not V14_CKPT:
        print("错误: 找不到checkpoint文件")
        return
    
    # 加载数据
    print(f"\n加载数据...")
    val_ds = FusionDataLoaderV3(RD_VAL, TRACK_VAL, val=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # 加载RD模型（共用）
    rd_model = rsnet34()
    rd_pths = glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth")
    rd_pths = [p for p in rd_pths if 'fusion' not in p]
    if rd_pths:
        rd_ckpt = torch.load(rd_pths[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
    rd_model.to(device).eval()
    
    results = {}
    
    # 测试V13
    print(f"\n测试V13...")
    v13_ckpt = torch.load(V13_CKPT, map_location='cpu')
    v13_track_weight = v13_ckpt.get('best_fixed_weight', v13_ckpt.get('track_weight', 0.40))
    
    v13_track_model = TrackOnlyNetV3()
    v13_track_model.load_state_dict(v13_ckpt['track_model'])
    v13_track_model.to(device).eval()
    
    results['V13'] = evaluate_model(rd_model, v13_track_model, val_loader, v13_track_weight, device)
    results['V13']['track_weight'] = v13_track_weight
    
    # 测试V14
    print(f"测试V14...")
    v14_ckpt = torch.load(V14_CKPT, map_location='cpu')
    v14_track_weight = v14_ckpt.get('best_fixed_weight', v14_ckpt.get('track_weight', 0.45))
    
    v14_track_model = TrackOnlyNetV3()
    v14_track_model.load_state_dict(v14_ckpt['track_model'])
    v14_track_model.to(device).eval()
    
    results['V14'] = evaluate_model(rd_model, v14_track_model, val_loader, v14_track_weight, device)
    results['V14']['track_weight'] = v14_track_weight
    
    # 输出对比报告
    print("\n" + "="*70)
    print("整体对比")
    print("="*70)
    
    print(f"\n{'指标':^20}|{'V13':^15}|{'V14':^15}|{'变化':^15}")
    print("-" * 70)
    
    metrics = [
        ('高置信度准确率', 'high_conf_acc', '%'),
        ('覆盖率', 'coverage', '%'),
        ('低置信度样本数', 'low_conf_total', ''),
        ('低置信度准确率', 'low_conf_acc', '%'),
        ('全部样本准确率', 'all_acc', '%'),
        ('Track权重', 'track_weight', ''),
    ]
    
    for name, key, unit in metrics:
        v13_val = results['V13'][key]
        v14_val = results['V14'][key]
        diff = v14_val - v13_val
        
        if unit == '%':
            v13_str = f"{v13_val:.2f}%"
            v14_str = f"{v14_val:.2f}%"
            diff_str = f"{diff:+.2f}%"
        else:
            v13_str = f"{v13_val}"
            v14_str = f"{v14_val}"
            diff_str = f"{diff:+.0f}" if isinstance(diff, (int, float)) else str(diff)
        
        # 标记提升
        if diff > 0 and key in ['high_conf_acc', 'coverage', 'low_conf_acc', 'all_acc']:
            diff_str += " ✓"
        elif diff < 0 and key == 'low_conf_total':
            diff_str += " ✓"
        
        print(f"{name:^20}|{v13_str:^15}|{v14_str:^15}|{diff_str:^15}")
    
    # 分类别对比
    print("\n" + "="*70)
    print("分类别准确率对比（高置信度）")
    print("="*70)
    
    print(f"\n{'类别':^12}|{'V13准确率':^12}|{'V14准确率':^12}|{'变化':^12}|{'V14样本数':^10}")
    print("-" * 65)
    
    for c in [0, 1, 2, 3]:
        v13_acc = results['V13']['class_stats'][c]['high_conf_acc']
        v14_acc = results['V14']['class_stats'][c]['high_conf_acc']
        diff = v14_acc - v13_acc
        count = results['V14']['class_stats'][c]['high_conf_total']
        name = results['V14']['class_stats'][c]['name']
        
        diff_str = f"{diff:+.2f}%"
        if diff > 0:
            diff_str += " ✓"
        
        print(f"{name:^12}|{v13_acc:^12.2f}%|{v14_acc:^12.2f}%|{diff_str:^12}|{count:^10}")
    
    # 低置信度样本分析
    print("\n" + "="*70)
    print("低置信度样本分类别分布")
    print("="*70)
    
    print(f"\n{'类别':^12}|{'V13低置信度':^12}|{'V14低置信度':^12}|{'变化':^10}")
    print("-" * 50)
    
    for c in [0, 1, 2, 3]:
        v13_low = results['V13']['class_stats'][c]['low_conf_total']
        v14_low = results['V14']['class_stats'][c]['low_conf_total']
        diff = v14_low - v13_low
        name = results['V14']['class_stats'][c]['name']
        
        diff_str = f"{diff:+d}"
        if diff < 0:
            diff_str += " ✓"
        
        print(f"{name:^12}|{v13_low:^12}|{v14_low:^12}|{diff_str:^10}")
    
    # 不同阈值对比
    print("\n" + "="*70)
    print("不同置信度阈值下的表现")
    print("="*70)
    
    print(f"\n{'阈值':^8}|{'V13准确率':^12}|{'V13覆盖率':^12}|{'V14准确率':^12}|{'V14覆盖率':^12}")
    print("-" * 60)
    
    for thresh in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]:
        # V13
        v13_high = [r for r in results['V13']['all_results'] if r['conf'] >= thresh]
        v13_acc = 100 * sum(1 for r in v13_high if r['correct']) / len(v13_high) if v13_high else 0
        v13_cov = 100 * len(v13_high) / results['V13']['total']
        
        # V14
        v14_high = [r for r in results['V14']['all_results'] if r['conf'] >= thresh]
        v14_acc = 100 * sum(1 for r in v14_high if r['correct']) / len(v14_high) if v14_high else 0
        v14_cov = 100 * len(v14_high) / results['V14']['total']
        
        mark = " *" if thresh == 0.5 else ""
        print(f"{thresh:^8.1f}|{v13_acc:^12.2f}%|{v13_cov:^12.1f}%|{v14_acc:^12.2f}%|{v14_cov:^12.1f}%{mark}")
    
    # 总结
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    
    v13_acc = results['V13']['high_conf_acc']
    v14_acc = results['V14']['high_conf_acc']
    v13_bird = results['V13']['class_stats'][2]['high_conf_acc']
    v14_bird = results['V14']['class_stats'][2]['high_conf_acc']
    
    print(f"""
V14 相比 V13 的改进:
  ✓ 高置信度准确率: {v13_acc:.2f}% → {v14_acc:.2f}% ({v14_acc-v13_acc:+.2f}%)
  ✓ 鸟类准确率: {v13_bird:.2f}% → {v14_bird:.2f}% ({v14_bird-v13_bird:+.2f}%)
  
关键改进:
  - 使用 Focal Loss 关注困难样本
  - 鸟类权重 2.0x 加强鸟类识别

低置信度样本情况:
  - V13: {results['V13']['low_conf_total']}个 (准确率{results['V13']['low_conf_acc']:.1f}%)
  - V14: {results['V14']['low_conf_total']}个 (准确率{results['V14']['low_conf_acc']:.1f}%)
""")
    
    if results['V14']['low_conf_total'] > 100:
        print(f"建议: 仍有{results['V14']['low_conf_total']}个低置信度样本，可考虑:")
        print(f"  1. 温度缩放(Temperature Scaling)提高覆盖率")
        print(f"  2. 降低阈值到0.4（覆盖率可达95%+）")
        print(f"  3. 对低置信度样本使用二次判断")


if __name__ == '__main__':
    main()