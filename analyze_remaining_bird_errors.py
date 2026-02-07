"""
深度分析：方案3下仍然误分类的鸟类样本
======================================
目标：找出剩余27个误分类鸟类的共同特征，设计针对性解决方案
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


def apply_strategy3(rd_probs, track_probs, track_stats):
    """
    方案3: 激进鸟类优化
    """
    BIRD_CLASS = 2
    batch_size = rd_probs.shape[0]
    
    results = []
    
    for i in range(batch_size):
        rd_p = rd_probs[i].numpy()
        track_p = track_probs[i].numpy()
        stats = track_stats[i].numpy()
        
        # 计算运动不稳定性
        std_vel = stats[1]
        max_accel = stats[7]
        turn_rate = stats[8]
        heading_stab = stats[9]
        
        instability = (std_vel/1.0 + max_accel/0.5 + turn_rate/5.0 + heading_stab/6.5) / 4.0
        
        # 基础权重
        rd_w = 0.55
        track_w = 0.45
        
        # 策略: 当RD预测为鸟类时，增加RD权重
        rd_pred = np.argmax(rd_p)
        if rd_pred == BIRD_CLASS:
            rd_w += 0.10
            track_w -= 0.10
        
        # 策略: 运动不稳定且RD预测为鸟时
        if instability > 1.2 and rd_pred == BIRD_CLASS:
            rd_w += 0.10
            track_w -= 0.10
        
        # 限制范围
        rd_w = max(0.3, min(0.8, rd_w))
        track_w = 1.0 - rd_w
        
        # 融合
        fused_p = rd_w * rd_p + track_w * track_p
        pred = np.argmax(fused_p)
        conf = fused_p[pred]
        
        results.append({
            'rd_probs': rd_p,
            'track_probs': track_p,
            'fused_probs': fused_p,
            'pred': pred,
            'conf': conf,
            'rd_weight': rd_w,
            'instability': instability,
            'stats': stats
        })
    
    return results


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
    BIRD_CLASS = 2
    
    FEAT_NAMES = [
        'mean_vel', 'std_vel', 'max_vel', 'min_vel',
        'mean_vz', 'std_vz', 'mean_accel', 'max_accel',
        'turn_rate', 'heading_stab', 'mean_range', 'range_change',
        'mean_pitch', 'std_pitch', 'mean_amp', 'std_amp',
        'mean_snr', 'mean_pts', 'n_pts', 'track_len'
    ]
    
    print("="*70)
    print("深度分析：方案3下仍然误分类的鸟类样本")
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
    
    # 加载数据
    val_ds = FusionDataLoaderV3(
        "./dataset/train_cleandata/val",
        "./dataset/track_enhanced_cleandata/val",
        val=True
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # 校准BN
    calibrate_bn(track_model, val_loader, device)
    
    # 收集所有鸟类样本的预测结果
    print("\n收集鸟类样本预测结果...")
    
    bird_samples = []
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            
            rd_probs = torch.softmax(rd_model(x_rd), dim=1).cpu()
            track_probs = torch.softmax(track_model(x_track, x_stats), dim=1).cpu()
            
            # 应用方案3
            results = apply_strategy3(rd_probs, track_probs, x_stats.cpu())
            
            for i in range(len(y)):
                if y[i].item() == BIRD_CLASS:
                    r = results[i]
                    r['true'] = BIRD_CLASS
                    r['correct'] = r['pred'] == BIRD_CLASS
                    bird_samples.append(r)
    
    print(f"鸟类样本总数: {len(bird_samples)}")
    
    # 分类
    correct_samples = [s for s in bird_samples if s['correct']]
    wrong_samples = [s for s in bird_samples if not s['correct']]
    
    print(f"正确分类: {len(correct_samples)} ({100*len(correct_samples)/len(bird_samples):.1f}%)")
    print(f"误分类: {len(wrong_samples)} ({100*len(wrong_samples)/len(bird_samples):.1f}%)")
    
    # ==================== 分析误分类样本 ====================
    print("\n" + "="*70)
    print(f"误分类样本分析 ({len(wrong_samples)}个)")
    print("="*70)
    
    # 1. 误分类去向
    print("\n1. 误分类去向:")
    wrong_to = defaultdict(list)
    for s in wrong_samples:
        wrong_to[s['pred']].append(s)
    
    for pred, samples in sorted(wrong_to.items(), key=lambda x: -len(x[1])):
        print(f"  → {CLASS_NAMES.get(pred, f'类别{pred}')}: {len(samples)}个 ({100*len(samples)/len(wrong_samples):.1f}%)")
    
    # 2. RD和Track的预测情况
    print("\n2. RD和Track在误分类样本上的预测:")
    
    rd_correct = sum(1 for s in wrong_samples if np.argmax(s['rd_probs']) == BIRD_CLASS)
    track_correct = sum(1 for s in wrong_samples if np.argmax(s['track_probs']) == BIRD_CLASS)
    
    print(f"  RD预测正确: {rd_correct}/{len(wrong_samples)} ({100*rd_correct/len(wrong_samples):.1f}%)")
    print(f"  Track预测正确: {track_correct}/{len(wrong_samples)} ({100*track_correct/len(wrong_samples):.1f}%)")
    
    # 3. 分析RD对但最终错的情况
    rd_right_final_wrong = [s for s in wrong_samples if np.argmax(s['rd_probs']) == BIRD_CLASS]
    print(f"\n3. RD预测正确但最终错误: {len(rd_right_final_wrong)}个")
    
    if rd_right_final_wrong:
        print("   这些样本可以通过更激进的RD策略挽救！")
        print(f"   详细分析:")
        for i, s in enumerate(rd_right_final_wrong[:5]):  # 只显示前5个
            rd_bird_prob = s['rd_probs'][BIRD_CLASS]
            track_bird_prob = s['track_probs'][BIRD_CLASS]
            fused_bird_prob = s['fused_probs'][BIRD_CLASS]
            final_pred = s['pred']
            print(f"   样本{i+1}: RD鸟={rd_bird_prob:.3f}, Track鸟={track_bird_prob:.3f}, "
                  f"融合鸟={fused_bird_prob:.3f}, 最终预测={CLASS_NAMES.get(final_pred, final_pred)}")
    
    # 4. 两个模型都错的情况
    both_wrong = [s for s in wrong_samples 
                  if np.argmax(s['rd_probs']) != BIRD_CLASS and np.argmax(s['track_probs']) != BIRD_CLASS]
    print(f"\n4. RD和Track都错: {len(both_wrong)}个")
    
    if both_wrong:
        print("   这些是真正困难的样本，需要其他方法解决")
        
        # 分析这些样本的特征
        print(f"\n   这些样本的特征分析:")
        both_wrong_stats = np.array([s['stats'] for s in both_wrong])
        correct_stats = np.array([s['stats'] for s in correct_samples])
        
        print(f"\n   {'特征':^15}|{'正确样本均值':^12}|{'两个都错均值':^12}|{'差异%':^10}")
        print("   " + "-" * 55)
        
        for i, name in enumerate(FEAT_NAMES):
            correct_mean = correct_stats[:, i].mean()
            wrong_mean = both_wrong_stats[:, i].mean()
            
            if abs(correct_mean) > 0.001:
                diff_pct = 100 * (wrong_mean - correct_mean) / abs(correct_mean)
            else:
                diff_pct = 0
            
            mark = " ⚠️" if abs(diff_pct) > 50 else ""
            print(f"   {name:^15}|{correct_mean:^12.3f}|{wrong_mean:^12.3f}|{diff_pct:^+10.1f}%{mark}")
    
    # 5. 置信度分析
    print("\n5. 置信度分析:")
    
    wrong_confs = [s['conf'] for s in wrong_samples]
    correct_confs = [s['conf'] for s in correct_samples]
    
    print(f"   误分类样本置信度: 平均={np.mean(wrong_confs):.3f}, 中位数={np.median(wrong_confs):.3f}")
    print(f"   正确样本置信度:   平均={np.mean(correct_confs):.3f}, 中位数={np.median(correct_confs):.3f}")
    
    # 按置信度分段
    print(f"\n   误分类样本置信度分布:")
    bins = [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]
    for low, high in bins:
        count = sum(1 for c in wrong_confs if low <= c < high)
        if count > 0:
            print(f"     [{low:.1f}, {high:.1f}): {count}个")
    
    # ==================== 可挽救性分析 ====================
    print("\n" + "="*70)
    print("可挽救性分析")
    print("="*70)
    
    # 类型1: RD对，可以通过更信任RD挽救
    type1 = rd_right_final_wrong
    
    # 类型2: 融合后鸟类概率是第二高的
    type2 = []
    for s in wrong_samples:
        sorted_idx = np.argsort(s['fused_probs'])[::-1]
        if sorted_idx[1] == BIRD_CLASS:  # 鸟类是第二选择
            type2.append(s)
    
    # 类型3: 鸟类概率>0.3（有一定识别能力）
    type3 = [s for s in wrong_samples if s['fused_probs'][BIRD_CLASS] > 0.3]
    
    print(f"\n可挽救类型:")
    print(f"  类型1 (RD预测正确): {len(type1)}个 → 增加RD权重可挽救")
    print(f"  类型2 (鸟类是第二选择): {len(type2)}个 → 微调阈值可挽救")
    print(f"  类型3 (鸟类概率>0.3): {len(type3)}个 → 有一定挽救可能")
    print(f"  困难样本 (两个都错): {len(both_wrong)}个 → 需要新方法")
    
    # ==================== 具体挽救方案 ====================
    print("\n" + "="*70)
    print("针对性挽救方案")
    print("="*70)
    
    # 方案A: 对类型1样本，当RD预测鸟且置信度较高时，直接采信RD
    print("\n方案A: 当RD预测鸟类且RD鸟类概率>0.5时，直接采信RD")
    rescued_a = 0
    for s in type1:
        if s['rd_probs'][BIRD_CLASS] > 0.5:
            rescued_a += 1
    print(f"  可挽救: {rescued_a}个")
    
    # 方案B: 对类型2样本，当鸟类是第二选择且差距小于0.1时，改判为鸟类
    print("\n方案B: 当鸟类是第二选择且与第一名差距<0.15时，改判为鸟类")
    rescued_b = 0
    for s in type2:
        sorted_probs = np.sort(s['fused_probs'])[::-1]
        gap = sorted_probs[0] - sorted_probs[1]
        if gap < 0.15:
            rescued_b += 1
    print(f"  可挽救: {rescued_b}个")
    
    # 方案C: 对低置信度样本，当RD鸟类概率>Track最高概率时，采信RD
    print("\n方案C: 低置信度时，当RD鸟类概率>0.4且>Track最高概率时，改判为鸟类")
    low_conf_wrong = [s for s in wrong_samples if s['conf'] < 0.5]
    rescued_c = 0
    for s in low_conf_wrong:
        rd_bird = s['rd_probs'][BIRD_CLASS]
        track_max = np.max(s['track_probs'])
        if rd_bird > 0.4 and rd_bird > track_max:
            rescued_c += 1
    print(f"  可挽救: {rescued_c}个 (在{len(low_conf_wrong)}个低置信度误分类中)")
    
    # 总结
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    
    # 估算改进后的准确率
    total_birds = len(bird_samples)
    current_correct = len(correct_samples)
    potential_rescue = rescued_a  # 保守估计只用方案A
    
    new_correct = current_correct + potential_rescue
    new_acc = 100 * new_correct / total_birds
    
    print(f"""
当前状态:
  - 鸟类样本: {total_birds}个
  - 正确分类: {current_correct}个 ({100*current_correct/total_birds:.2f}%)
  - 误分类: {len(wrong_samples)}个

误分类分析:
  - RD预测正确: {len(rd_right_final_wrong)}个 (可通过更信任RD挽救)
  - 两个模型都错: {len(both_wrong)}个 (困难样本)

建议的下一步:
  1. 实施方案A: 当RD预测鸟类且概率>0.5时，直接采信RD
     预计挽救: {rescued_a}个
     预计新准确率: {new_acc:.2f}%
  
  2. 如果还不够，实施方案B: 鸟类是第二选择且差距小时，改判为鸟类
  
  3. 对于{len(both_wrong)}个困难样本，可能需要:
     - 分析RD图像特征
     - 检查是否是标注错误
     - 或接受这是模型能力边界
""")


if __name__ == '__main__':
    main()