"""
智能二次判断器
==============
核心逻辑：
  - RD模型擅长识别无人机（类别0,1）
  - Track模型擅长识别鸟类和空飘球（类别2,3）
  
当融合置信度 < 阈值时，根据各模型的预测类别决定信任谁

使用方法:
  python smart_secondary_classifier.py
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


def smart_decision(rd_pred, rd_conf, track_pred, track_conf, fused_pred, fused_conf,
                   conf_thresh=0.5, rd_uav_bonus=0.1, track_bird_bonus=0.15):
    """
    智能决策函数
    
    Args:
        rd_pred, rd_conf: RD模型的预测和置信度
        track_pred, track_conf: Track模型的预测和置信度
        fused_pred, fused_conf: 融合模型的预测和置信度
        conf_thresh: 触发二次判断的阈值
        rd_uav_bonus: RD预测无人机时的加成
        track_bird_bonus: Track预测鸟类/空飘球时的加成
    
    Returns:
        final_pred: 最终预测
        final_conf: 最终置信度
        decision_type: 决策类型 ('fusion', 'rd', 'track')
    """
    # 高置信度直接使用融合结果
    if fused_conf >= conf_thresh:
        return fused_pred, fused_conf, 'fusion'
    
    # 低置信度启用智能判断
    # 计算调整后的置信度
    rd_adjusted = rd_conf
    track_adjusted = track_conf
    
    # RD对无人机有加成
    if rd_pred in [0, 1]:  # 轻型或小型无人机
        rd_adjusted += rd_uav_bonus
    
    # Track对鸟类和空飘球有加成
    if track_pred in [2, 3]:  # 鸟类或空飘球
        track_adjusted += track_bird_bonus
    
    # 选择调整后置信度更高的模型
    if rd_adjusted > track_adjusted:
        return rd_pred, rd_conf, 'rd'
    else:
        return track_pred, track_conf, 'track'


def test_strategy(all_samples, strategy_func, strategy_name, **kwargs):
    """测试某个策略的效果"""
    VALID_CLASSES = [0, 1, 2, 3]
    
    correct = 0
    total = 0
    
    low_conf_correct = 0
    low_conf_total = 0
    
    decisions = {'fusion': 0, 'rd': 0, 'track': 0}
    
    for s in all_samples:
        if s['true_label'] not in VALID_CLASSES:
            continue
        
        final_pred, final_conf, decision_type = strategy_func(
            s['rd_pred'], s['rd_conf'],
            s['track_pred'], s['track_conf'],
            s['fused_pred'], s['fused_conf'],
            **kwargs
        )
        
        total += 1
        if final_pred == s['true_label']:
            correct += 1
        
        decisions[decision_type] += 1
        
        # 统计原本低置信度的样本
        if s['fused_conf'] < 0.5:
            low_conf_total += 1
            if final_pred == s['true_label']:
                low_conf_correct += 1
    
    return {
        'name': strategy_name,
        'total_acc': 100 * correct / total,
        'low_conf_acc': 100 * low_conf_correct / low_conf_total if low_conf_total > 0 else 0,
        'low_conf_total': low_conf_total,
        'low_conf_correct': low_conf_correct,
        'decisions': decisions
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 配置
    RD_VAL = "./dataset/train_cleandata/val"
    TRACK_VAL = "./dataset/track_enhanced_cleandata/val"
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
    
    print("="*70)
    print("智能二次判断器测试")
    print("="*70)
    
    # 加载数据
    print("\n加载数据...")
    val_ds = FusionDataLoaderV3(RD_VAL, TRACK_VAL, val=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # 加载模型
    print("加载模型...")
    
    v14_pths = glob.glob("./checkpoint/fusion_v14*/ckpt_best*.pth")
    if not v14_pths:
        v14_pths = glob.glob("./checkpoint/fusion_v13*/ckpt_best*.pth")
    CKPT_PATH = sorted(v14_pths)[-1]
    
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    track_weight = ckpt.get('best_fixed_weight', ckpt.get('track_weight', 0.45))
    rd_weight = 1.0 - track_weight
    
    rd_model = rsnet34()
    rd_pths = glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth")
    rd_pths = [p for p in rd_pths if 'fusion' not in p]
    if rd_pths:
        rd_ckpt = torch.load(rd_pths[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
    rd_model.to(device).eval()
    
    track_model = TrackOnlyNetV3()
    track_model.load_state_dict(ckpt['track_model'])
    track_model.to(device).eval()
    
    # 收集所有样本
    print("收集样本...")
    
    all_samples = []
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd_dev = x_rd.to(device)
            x_track_dev = x_track.to(device)
            x_stats_dev = x_stats.to(device)
            
            rd_probs = torch.softmax(rd_model(x_rd_dev), dim=1)
            rd_conf, rd_pred = rd_probs.max(dim=1)
            
            track_probs = torch.softmax(track_model(x_track_dev, x_stats_dev), dim=1)
            track_conf, track_pred = track_probs.max(dim=1)
            
            fused_probs = rd_weight * rd_probs + track_weight * track_probs
            fused_conf, fused_pred = fused_probs.max(dim=1)
            
            for i in range(len(y)):
                all_samples.append({
                    'true_label': y[i].item(),
                    'rd_pred': rd_pred[i].item(),
                    'rd_conf': rd_conf[i].item(),
                    'track_pred': track_pred[i].item(),
                    'track_conf': track_conf[i].item(),
                    'fused_pred': fused_pred[i].item(),
                    'fused_conf': fused_conf[i].item(),
                })
    
    # 过滤有效类别
    all_samples = [s for s in all_samples if s['true_label'] in VALID_CLASSES]
    print(f"总样本数: {len(all_samples)}")
    
    # ========== 测试各种策略 ==========
    print("\n" + "="*70)
    print("测试各种二次判断策略")
    print("="*70)
    
    # 策略1：基线（纯融合，不做二次判断）
    def baseline_strategy(rd_pred, rd_conf, track_pred, track_conf, fused_pred, fused_conf, **kwargs):
        return fused_pred, fused_conf, 'fusion'
    
    # 策略2：简单置信度比较
    def simple_conf_strategy(rd_pred, rd_conf, track_pred, track_conf, fused_pred, fused_conf, conf_thresh=0.5, **kwargs):
        if fused_conf >= conf_thresh:
            return fused_pred, fused_conf, 'fusion'
        if rd_conf > track_conf:
            return rd_pred, rd_conf, 'rd'
        else:
            return track_pred, track_conf, 'track'
    
    # 策略3：智能判断（基于类别擅长度）
    # smart_decision已定义
    
    # 策略4：总是信任RD（对低置信度样本）
    def always_rd_strategy(rd_pred, rd_conf, track_pred, track_conf, fused_pred, fused_conf, conf_thresh=0.5, **kwargs):
        if fused_conf >= conf_thresh:
            return fused_pred, fused_conf, 'fusion'
        return rd_pred, rd_conf, 'rd'
    
    # 策略5：总是信任Track（对低置信度样本）
    def always_track_strategy(rd_pred, rd_conf, track_pred, track_conf, fused_pred, fused_conf, conf_thresh=0.5, **kwargs):
        if fused_conf >= conf_thresh:
            return fused_pred, fused_conf, 'fusion'
        return track_pred, track_conf, 'track'
    
    strategies = [
        (baseline_strategy, "基线(纯融合)", {}),
        (simple_conf_strategy, "简单置信度比较", {}),
        (always_rd_strategy, "低置信度→信任RD", {}),
        (always_track_strategy, "低置信度→信任Track", {}),
        (smart_decision, "智能判断(bonus=0.1/0.15)", {'rd_uav_bonus': 0.1, 'track_bird_bonus': 0.15}),
        (smart_decision, "智能判断(bonus=0.15/0.20)", {'rd_uav_bonus': 0.15, 'track_bird_bonus': 0.20}),
        (smart_decision, "智能判断(bonus=0.05/0.10)", {'rd_uav_bonus': 0.05, 'track_bird_bonus': 0.10}),
    ]
    
    results = []
    for func, name, kwargs in strategies:
        result = test_strategy(all_samples, func, name, **kwargs)
        results.append(result)
    
    # 打印结果
    print(f"\n{'策略':^25}|{'总准确率':^10}|{'低置信度准确率':^14}|{'低置信度正确数':^14}")
    print("-" * 70)
    
    best_result = max(results, key=lambda x: x['low_conf_acc'])
    
    for r in results:
        mark = " *" if r['name'] == best_result['name'] else ""
        print(f"{r['name']:^25}|{r['total_acc']:^10.2f}%|{r['low_conf_acc']:^14.1f}%|{r['low_conf_correct']:^14}/{r['low_conf_total']}{mark}")
    
    # ========== 最佳策略详细分析 ==========
    print("\n" + "="*70)
    print(f"最佳策略详细分析: {best_result['name']}")
    print("="*70)
    
    # 找到最佳策略的参数
    best_strategy = None
    best_kwargs = {}
    for func, name, kwargs in strategies:
        if name == best_result['name']:
            best_strategy = func
            best_kwargs = kwargs
            break
    
    # 分类别分析
    print(f"\n分类别效果:")
    print(f"{'类别':^12}|{'原融合正确':^12}|{'优化后正确':^12}|{'提升':^10}")
    print("-" * 50)
    
    for c in VALID_CLASSES:
        c_samples = [s for s in all_samples if s['true_label'] == c and s['fused_conf'] < 0.5]
        if not c_samples:
            continue
        
        orig_correct = sum(1 for s in c_samples if s['fused_pred'] == c)
        
        new_correct = 0
        for s in c_samples:
            pred, _, _ = best_strategy(
                s['rd_pred'], s['rd_conf'],
                s['track_pred'], s['track_conf'],
                s['fused_pred'], s['fused_conf'],
                **best_kwargs
            )
            if pred == c:
                new_correct += 1
        
        diff = new_correct - orig_correct
        print(f"{CLASS_NAMES[c]:^12}|{orig_correct:^12}|{new_correct:^12}|{diff:^+10}")
    
    # ========== 生成最终推理代码 ==========
    print("\n" + "="*70)
    print("推理代码")
    print("="*70)
    
    # 提取最佳参数
    rd_bonus = best_kwargs.get('rd_uav_bonus', 0.1)
    track_bonus = best_kwargs.get('track_bird_bonus', 0.15)
    
    print(f"""
def smart_inference(rd_probs, track_probs, rd_weight={rd_weight}, track_weight={track_weight},
                    conf_thresh=0.5, rd_uav_bonus={rd_bonus}, track_bird_bonus={track_bonus}):
    '''
    智能融合推理
    
    Args:
        rd_probs: RD模型输出的概率 [batch, 6]
        track_probs: Track模型输出的概率 [batch, 6]
    
    Returns:
        predictions: 最终预测 [batch]
        confidences: 最终置信度 [batch]
    '''
    # 融合
    fused_probs = rd_weight * rd_probs + track_weight * track_probs
    fused_conf, fused_pred = fused_probs.max(dim=1)
    
    rd_conf, rd_pred = rd_probs.max(dim=1)
    track_conf, track_pred = track_probs.max(dim=1)
    
    # 对低置信度样本进行二次判断
    final_pred = fused_pred.clone()
    final_conf = fused_conf.clone()
    
    low_conf_mask = fused_conf < conf_thresh
    
    for i in range(len(fused_pred)):
        if not low_conf_mask[i]:
            continue
        
        # 计算调整后置信度
        rd_adj = rd_conf[i].item()
        track_adj = track_conf[i].item()
        
        if rd_pred[i].item() in [0, 1]:  # RD预测无人机
            rd_adj += rd_uav_bonus
        if track_pred[i].item() in [2, 3]:  # Track预测鸟类/空飘球
            track_adj += track_bird_bonus
        
        if rd_adj > track_adj:
            final_pred[i] = rd_pred[i]
            final_conf[i] = rd_conf[i]
        else:
            final_pred[i] = track_pred[i]
            final_conf[i] = track_conf[i]
    
    return final_pred, final_conf
""")
    
    # ========== 总结 ==========
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    
    baseline = results[0]
    
    print(f"""
优化效果:
  基线(纯融合):
    - 总准确率: {baseline['total_acc']:.2f}%
    - 低置信度准确率: {baseline['low_conf_acc']:.1f}% ({baseline['low_conf_correct']}/{baseline['low_conf_total']})
  
  最佳策略({best_result['name']}):
    - 总准确率: {best_result['total_acc']:.2f}% ({best_result['total_acc']-baseline['total_acc']:+.2f}%)
    - 低置信度准确率: {best_result['low_conf_acc']:.1f}% ({best_result['low_conf_correct']}/{best_result['low_conf_total']})
    - 低置信度提升: {best_result['low_conf_acc']-baseline['low_conf_acc']:+.1f}%
    - 多挽救: {best_result['low_conf_correct']-baseline['low_conf_correct']}个样本

核心原理:
  - RD模型擅长识别无人机（微多普勒特征）
  - Track模型擅长识别鸟类/空飘球（运动模式特征）
  - 对低置信度样本，根据各模型预测的类别决定信任谁
""")


if __name__ == '__main__':
    main()