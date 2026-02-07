"""
鸟类识别分析脚本
================
分析鸟类（类别2）的识别问题：
1. 鸟类被错分成哪些类别？
2. RD和航迹分别在哪些鸟类样本上出错？
3. 鸟类样本的特征分布分析
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
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
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )
        
        self.stats_net = nn.Sequential(
            nn.Linear(20, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x_temporal, x_stats):
        feat_temporal = self.temporal_net(x_temporal)
        feat_stats = self.stats_net(x_stats)
        feat_combined = torch.cat([feat_temporal, feat_stats], dim=1)
        return self.classifier(feat_combined)


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 路径配置
    RD_VAL = "./dataset/train/2026-1-14/val"
    TRACK_VAL = "./dataset/track_enhanced/val"
    RD_PRETRAINED = "./checkpoint/ckpt_best_3_94.08.pth"
    FUSION_CKPT = "./checkpoint/fusion_v13_final/ckpt_best_97.39.pth"
    
    # 如果找不到，尝试其他路径
    if not os.path.exists(RD_PRETRAINED):
        import glob
        pths = glob.glob("./checkpoint/*93*.pth") + glob.glob("./checkpoint/*94*.pth")
        if pths:
            RD_PRETRAINED = pths[0]
    
    if not os.path.exists(FUSION_CKPT):
        import glob
        pths = glob.glob("./checkpoint/fusion_v13*/ckpt_best*.pth")
        if pths:
            FUSION_CKPT = pths[0]
    
    CONF_THRESH = 0.5
    TRACK_WEIGHT = 0.35
    RD_WEIGHT = 0.65
    
    CLASS_NAMES = {
        0: '轻型无人机',
        1: '小型无人机',
        2: '鸟类',
        3: '空飘球',
        4: '杂波',
        5: '其它'
    }
    
    print(f"\n{'='*70}")
    print(f"鸟类识别问题分析")
    print(f"{'='*70}")
    
    # 加载数据
    print("\n[1] 加载数据...")
    val_ds = FusionDataLoaderV3(RD_VAL, TRACK_VAL, val=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)  # batch=1便于逐样本分析
    
    # 加载RD模型
    print("\n[2] 加载RD模型...")
    rd_model = rsnet34()
    ckpt = torch.load(RD_PRETRAINED, map_location='cpu')
    state_dict = ckpt['net_weight'] if 'net_weight' in ckpt else ckpt
    rd_model.load_state_dict(state_dict, strict=True)
    rd_model.to(DEVICE)
    rd_model.eval()
    
    # 加载航迹模型
    print("\n[3] 加载航迹模型...")
    track_model = TrackOnlyNetV3(num_classes=6).to(DEVICE)
    if os.path.exists(FUSION_CKPT):
        ckpt = torch.load(FUSION_CKPT, map_location='cpu')
        if 'track_model' in ckpt:
            track_model.load_state_dict(ckpt['track_model'])
    track_model.eval()
    
    # ==========================================
    # 收集所有鸟类样本的预测结果
    # ==========================================
    print("\n[4] 分析鸟类样本...")
    
    bird_samples = []  # 存储所有鸟类样本的详细信息
    
    # 统计数据
    bird_total = 0
    bird_rd_correct = 0
    bird_track_correct = 0
    bird_fusion_correct = 0
    
    # 错误分类统计
    bird_misclassified_as = defaultdict(int)  # 融合后被错分为哪个类
    rd_misclassified_as = defaultdict(int)
    track_misclassified_as = defaultdict(int)
    
    # 特征统计
    bird_track_features = []
    bird_stats_features = []
    
    with torch.no_grad():
        sample_idx = 0
        for x_rd, x_track, x_stats, y in val_loader:
            true_label = y[0].item()
            
            # 只分析鸟类 (类别2)
            if true_label != 2:
                sample_idx += 1
                continue
            
            x_rd = x_rd.to(DEVICE)
            x_track = x_track.to(DEVICE)
            x_stats = x_stats.to(DEVICE)
            
            rd_logits = rd_model(x_rd)
            track_logits = track_model(x_track, x_stats)
            
            rd_probs = torch.softmax(rd_logits, dim=1)
            track_probs = torch.softmax(track_logits, dim=1)
            fused_probs = RD_WEIGHT * rd_probs + TRACK_WEIGHT * track_probs
            
            rd_conf, rd_pred = rd_probs.max(dim=1)
            track_conf, track_pred = track_probs.max(dim=1)
            fused_conf, fused_pred = fused_probs.max(dim=1)
            
            # 置信度过滤
            if fused_conf[0] < CONF_THRESH:
                sample_idx += 1
                continue
            
            bird_total += 1
            
            # 收集特征用于统计
            bird_track_features.append(x_track[0].cpu().numpy())
            bird_stats_features.append(x_stats[0].cpu().numpy())
            
            # 判断正误
            rd_correct = (rd_pred[0].item() == 2)
            track_correct = (track_pred[0].item() == 2)
            fused_correct = (fused_pred[0].item() == 2)
            
            if rd_correct:
                bird_rd_correct += 1
            else:
                rd_misclassified_as[rd_pred[0].item()] += 1
            
            if track_correct:
                bird_track_correct += 1
            else:
                track_misclassified_as[track_pred[0].item()] += 1
            
            if fused_correct:
                bird_fusion_correct += 1
            else:
                bird_misclassified_as[fused_pred[0].item()] += 1
            
            # 保存详细信息
            bird_samples.append({
                'idx': sample_idx,
                'rd_pred': rd_pred[0].item(),
                'rd_conf': rd_conf[0].item(),
                'track_pred': track_pred[0].item(),
                'track_conf': track_conf[0].item(),
                'fused_pred': fused_pred[0].item(),
                'fused_conf': fused_conf[0].item(),
                'rd_correct': rd_correct,
                'track_correct': track_correct,
                'fused_correct': fused_correct,
                'rd_probs': rd_probs[0].cpu().numpy(),
                'track_probs': track_probs[0].cpu().numpy(),
            })
            
            sample_idx += 1
    
    # ==========================================
    # 输出分析结果
    # ==========================================
    print(f"\n{'='*70}")
    print(f"鸟类识别统计")
    print(f"{'='*70}")
    
    print(f"\n鸟类样本总数: {bird_total}")
    print(f"  RD正确:   {bird_rd_correct}/{bird_total} ({100*bird_rd_correct/bird_total:.2f}%)")
    print(f"  航迹正确: {bird_track_correct}/{bird_total} ({100*bird_track_correct/bird_total:.2f}%)")
    print(f"  融合正确: {bird_fusion_correct}/{bird_total} ({100*bird_fusion_correct/bird_total:.2f}%)")
    
    # ==========================================
    # 错误分类去向分析
    # ==========================================
    print(f"\n{'='*70}")
    print(f"鸟类被错分为其他类别的统计")
    print(f"{'='*70}")
    
    print(f"\n融合后鸟类被错分为:")
    for pred_class, count in sorted(bird_misclassified_as.items(), key=lambda x: -x[1]):
        print(f"  -> {CLASS_NAMES[pred_class]}: {count}次 ({100*count/(bird_total-bird_fusion_correct):.1f}%)")
    
    print(f"\nRD单独预测时鸟类被错分为:")
    for pred_class, count in sorted(rd_misclassified_as.items(), key=lambda x: -x[1]):
        print(f"  -> {CLASS_NAMES[pred_class]}: {count}次 ({100*count/(bird_total-bird_rd_correct):.1f}%)")
    
    print(f"\n航迹单独预测时鸟类被错分为:")
    for pred_class, count in sorted(track_misclassified_as.items(), key=lambda x: -x[1]):
        print(f"  -> {CLASS_NAMES[pred_class]}: {count}次 ({100*count/(bird_total-bird_track_correct):.1f}%)")
    
    # ==========================================
    # 错误样本的置信度分析
    # ==========================================
    print(f"\n{'='*70}")
    print(f"错误样本的置信度分析")
    print(f"{'='*70}")
    
    wrong_samples = [s for s in bird_samples if not s['fused_correct']]
    correct_samples = [s for s in bird_samples if s['fused_correct']]
    
    if wrong_samples:
        wrong_rd_confs = [s['rd_conf'] for s in wrong_samples]
        wrong_track_confs = [s['track_conf'] for s in wrong_samples]
        wrong_fused_confs = [s['fused_conf'] for s in wrong_samples]
        
        print(f"\n被错分的{len(wrong_samples)}个鸟类样本:")
        print(f"  RD置信度:   均值={np.mean(wrong_rd_confs):.3f}, 范围=[{np.min(wrong_rd_confs):.3f}, {np.max(wrong_rd_confs):.3f}]")
        print(f"  航迹置信度: 均值={np.mean(wrong_track_confs):.3f}, 范围=[{np.min(wrong_track_confs):.3f}, {np.max(wrong_track_confs):.3f}]")
        print(f"  融合置信度: 均值={np.mean(wrong_fused_confs):.3f}, 范围=[{np.min(wrong_fused_confs):.3f}, {np.max(wrong_fused_confs):.3f}]")
    
    if correct_samples:
        correct_rd_confs = [s['rd_conf'] for s in correct_samples]
        correct_track_confs = [s['track_conf'] for s in correct_samples]
        correct_fused_confs = [s['fused_conf'] for s in correct_samples]
        
        print(f"\n正确分类的{len(correct_samples)}个鸟类样本:")
        print(f"  RD置信度:   均值={np.mean(correct_rd_confs):.3f}, 范围=[{np.min(correct_rd_confs):.3f}, {np.max(correct_rd_confs):.3f}]")
        print(f"  航迹置信度: 均值={np.mean(correct_track_confs):.3f}, 范围=[{np.min(correct_track_confs):.3f}, {np.max(correct_track_confs):.3f}]")
        print(f"  融合置信度: 均值={np.mean(correct_fused_confs):.3f}, 范围=[{np.min(correct_fused_confs):.3f}, {np.max(correct_fused_confs):.3f}]")
    
    # ==========================================
    # RD和航迹互补性分析（针对鸟类）
    # ==========================================
    print(f"\n{'='*70}")
    print(f"RD和航迹在鸟类上的互补性")
    print(f"{'='*70}")
    
    both_right = sum(1 for s in bird_samples if s['rd_correct'] and s['track_correct'])
    rd_only_right = sum(1 for s in bird_samples if s['rd_correct'] and not s['track_correct'])
    track_only_right = sum(1 for s in bird_samples if not s['rd_correct'] and s['track_correct'])
    both_wrong = sum(1 for s in bird_samples if not s['rd_correct'] and not s['track_correct'])
    
    print(f"\n  RD对, 航迹对: {both_right} ({100*both_right/bird_total:.1f}%)")
    print(f"  RD对, 航迹错: {rd_only_right} ({100*rd_only_right/bird_total:.1f}%)")
    print(f"  RD错, 航迹对: {track_only_right} ({100*track_only_right/bird_total:.1f}%) <- 航迹能补救")
    print(f"  RD错, 航迹错: {both_wrong} ({100*both_wrong/bird_total:.1f}%) <- 都搞不定")
    
    theoretical_max = 100 * (both_right + rd_only_right + track_only_right) / bird_total
    print(f"\n  鸟类理论最大准确率: {theoretical_max:.2f}%")
    print(f"  当前鸟类融合准确率: {100*bird_fusion_correct/bird_total:.2f}%")
    
    # ==========================================
    # 详细列出"都搞不定"的样本
    # ==========================================
    print(f"\n{'='*70}")
    print(f"RD和航迹都错的鸟类样本详情")
    print(f"{'='*70}")
    
    both_wrong_samples = [s for s in bird_samples if not s['rd_correct'] and not s['track_correct']]
    
    if both_wrong_samples:
        print(f"\n共{len(both_wrong_samples)}个样本:")
        print(f"{'样本':^6}|{'RD预测':^12}|{'RD置信度':^10}|{'航迹预测':^12}|{'航迹置信度':^10}")
        print("-" * 60)
        
        for s in both_wrong_samples[:20]:  # 只显示前20个
            rd_pred_name = CLASS_NAMES[s['rd_pred']][:6]
            track_pred_name = CLASS_NAMES[s['track_pred']][:6]
            print(f"{s['idx']:^6}|{rd_pred_name:^12}|{s['rd_conf']:^10.3f}|{track_pred_name:^12}|{s['track_conf']:^10.3f}")
    
    # ==========================================
    # 统计特征分析
    # ==========================================
    if bird_stats_features:
        print(f"\n{'='*70}")
        print(f"鸟类统计特征分析")
        print(f"{'='*70}")
        
        stats_array = np.array(bird_stats_features)
        
        feature_names = [
            '平均速度', '速度标准差', '最大速度', '最小速度',
            '平均垂直速度', '垂直速度波动', '平均加速度', '最大加速度',
            '平均转弯率', '航向稳定性', '平均距离', '距离变化',
            '平均俯仰', '俯仰波动', '平均幅度', '幅度波动',
            '平均信噪比', '平均点数', '点数', '轨迹长度'
        ]
        
        print(f"\n鸟类样本统计特征 (前10个):")
        print(f"{'特征名':^12}|{'均值':^10}|{'标准差':^10}|{'最小值':^10}|{'最大值':^10}")
        print("-" * 55)
        
        for i in range(min(10, stats_array.shape[1])):
            col = stats_array[:, i]
            name = feature_names[i] if i < len(feature_names) else f'特征{i}'
            print(f"{name:^12}|{np.mean(col):^10.3f}|{np.std(col):^10.3f}|{np.min(col):^10.3f}|{np.max(col):^10.3f}")
    
    # ==========================================
    # 改进建议
    # ==========================================
    print(f"\n{'='*70}")
    print(f"改进建议")
    print(f"{'='*70}")
    
    # 根据分析结果给出建议
    main_confusion = max(bird_misclassified_as.items(), key=lambda x: x[1]) if bird_misclassified_as else (0, 0)
    
    print(f"\n1. 主要混淆类别: 鸟类最常被错分为 {CLASS_NAMES[main_confusion[0]]} ({main_confusion[1]}次)")
    print(f"   -> 建议: 重点分析鸟类与{CLASS_NAMES[main_confusion[0]]}的区别特征")
    
    print(f"\n2. RD和航迹都错的样本: {both_wrong}个 ({100*both_wrong/bird_total:.1f}%)")
    print(f"   -> 这些是最难的样本，可能需要:")
    print(f"      - 检查标注是否正确")
    print(f"      - 增加更多类似的训练样本")
    print(f"      - 提取更有区分力的特征")
    
    if track_only_right > 0:
        print(f"\n3. 航迹能补救但融合没用上: {track_only_right}个样本")
        print(f"   -> 考虑针对鸟类提高航迹权重")
    
    print(f"\n4. 鸟类理论上限: {theoretical_max:.2f}%")
    print(f"   -> 当前: {100*bird_fusion_correct/bird_total:.2f}%")
    print(f"   -> 如果完美融合，鸟类可以达到 {theoretical_max:.2f}%")


if __name__ == '__main__':
    main()