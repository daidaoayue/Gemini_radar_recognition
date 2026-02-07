"""
V15 测试 - 完全复制训练脚本的验证逻辑
======================================
确保模型结构与训练时100%一致
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

from data_loader_fusion_v4 import FusionDataLoaderV4
from drsncww import rsnet34


# =========================================================
# 【关键】必须和 train_fusion_v15_enhanced.py 里的完全一致！
# =========================================================
class TrackOnlyNetV4(nn.Module):
    """航迹网络V4 - 支持28维统计特征"""
    def __init__(self, num_classes=6, stats_dim=28):
        super().__init__()
        self.stats_dim = stats_dim
        
        # 时序特征处理 [12, 16] -> 256维
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
        
        # 统计特征处理 [stats_dim] -> 128维
        self.stats_net = nn.Sequential(
            nn.Linear(stats_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # 分类器 (256 + 128 = 384维)
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
        return self.classifier(torch.cat([feat_temporal, feat_stats], dim=1))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
    
    print("="*70)
    print("V15 测试 - 复制训练验证逻辑")
    print("="*70)
    
    # 配置
    RD_VAL = "./dataset/train_cleandata/val"
    TRACK_VAL = "./dataset/track_enhanced_v4_cleandata/val"
    
    # 加载checkpoint
    v15_pths = glob.glob("./checkpoint/fusion_v15*/ckpt_best*.pth")
    if not v15_pths:
        print("错误: 找不到V15 checkpoint!")
        return
    
    v15_ckpt_path = sorted(v15_pths)[-1]
    print(f"\nCheckpoint: {v15_ckpt_path}")
    
    ckpt = torch.load(v15_ckpt_path, map_location='cpu')
    track_weight = ckpt.get('best_fixed_weight', 0.5)
    rd_weight = 1.0 - track_weight
    stats_dim = ckpt.get('stats_dim', 28)
    
    print(f"  Track权重: {track_weight}")
    print(f"  stats_dim: {stats_dim}")
    print(f"  训练最佳准确率: {ckpt.get('best_acc', 'N/A')}")
    
    # 检查checkpoint内容
    print(f"\nCheckpoint包含的键:")
    for key in ckpt.keys():
        if isinstance(ckpt[key], torch.Tensor):
            print(f"  {key}: Tensor {ckpt[key].shape}")
        elif isinstance(ckpt[key], dict):
            print(f"  {key}: dict with {len(ckpt[key])} keys")
        else:
            print(f"  {key}: {type(ckpt[key]).__name__} = {ckpt[key]}")
    
    # 加载RD模型
    print(f"\n加载RD模型...")
    rd_model = rsnet34()
    rd_pths = glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth")
    rd_pths = [p for p in rd_pths if 'fusion' not in p]
    if rd_pths:
        rd_ckpt = torch.load(rd_pths[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
        print(f"  已加载: {rd_pths[0]}")
    rd_model.to(device)
    rd_model.eval()
    
    # 加载Track模型
    print(f"\n加载Track模型...")
    track_model = TrackOnlyNetV4(num_classes=6, stats_dim=stats_dim)
    
    # 检查权重是否匹配
    model_keys = set(track_model.state_dict().keys())
    ckpt_keys = set(ckpt['track_model'].keys())
    
    if model_keys != ckpt_keys:
        print(f"  ⚠️ 模型结构不匹配!")
        print(f"  模型有但checkpoint没有: {model_keys - ckpt_keys}")
        print(f"  checkpoint有但模型没有: {ckpt_keys - model_keys}")
    else:
        print(f"  ✓ 模型结构匹配")
    
    # 加载权重
    track_model.load_state_dict(ckpt['track_model'])
    track_model.to(device)
    track_model.eval()
    
    # 加载数据
    print(f"\n加载数据...")
    val_ds = FusionDataLoaderV4(RD_VAL, TRACK_VAL, val=True, stats_dim=stats_dim)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # =========================================================
    # 【完全复制训练脚本的验证逻辑】
    # =========================================================
    print(f"\n开始测试 (使用训练脚本的验证逻辑)...")
    
    correct_high = 0
    total_high = 0
    correct_all = 0
    total_all = 0
    
    # 用于详细分析
    all_logits = []
    all_preds = []
    all_labels = []
    all_confs = []
    
    with torch.no_grad():
        for batch_idx, (x_rd, x_track, x_stats, y) in enumerate(val_loader):
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            
            rd_probs = torch.softmax(rd_model(x_rd), dim=1)
            track_probs = torch.softmax(track_model(x_track, x_stats), dim=1)
            
            fused_probs = rd_weight * rd_probs + track_weight * track_probs
            conf, pred = fused_probs.max(dim=1)
            
            # 保存用于分析
            if batch_idx == 0:
                print(f"\n  第一个batch的输出分析:")
                print(f"    RD probs[0]: {rd_probs[0].cpu().numpy()}")
                print(f"    Track probs[0]: {track_probs[0].cpu().numpy()}")
                print(f"    Fused probs[0]: {fused_probs[0].cpu().numpy()}")
                print(f"    预测: {pred[0].item()}, 真实: {y[0].item()}, 置信度: {conf[0].item():.4f}")
            
            for i in range(len(y)):
                true_label = y[i].item()
                if true_label not in VALID_CLASSES:
                    continue
                
                total_all += 1
                all_preds.append(pred[i].item())
                all_labels.append(true_label)
                all_confs.append(conf[i].item())
                
                if pred[i].item() == true_label:
                    correct_all += 1
                
                if conf[i].item() >= 0.5:
                    total_high += 1
                    if pred[i].item() == true_label:
                        correct_high += 1
    
    # 输出结果
    if total_high > 0:
        acc_high = 100 * correct_high / total_high
        coverage = 100 * total_high / total_all
    else:
        acc_high = 0
        coverage = 0
    
    print(f"\n" + "="*70)
    print(f"测试结果")
    print(f"="*70)
    print(f"\n总样本数: {total_all}")
    print(f"高置信度样本: {total_high} ({coverage:.1f}%)")
    print(f"高置信度准确率: {acc_high:.2f}%")
    print(f"总体准确率: {100*correct_all/total_all:.2f}%")
    
    # 预测分布
    print(f"\n预测分布:")
    pred_counts = defaultdict(int)
    for p in all_preds:
        pred_counts[p] += 1
    for c in sorted(pred_counts.keys()):
        name = CLASS_NAMES.get(c, f"类别{c}")
        print(f"  {name}: {pred_counts[c]}个 ({100*pred_counts[c]/total_all:.1f}%)")
    
    # 真实标签分布
    print(f"\n真实标签分布:")
    label_counts = defaultdict(int)
    for l in all_labels:
        label_counts[l] += 1
    for c in sorted(label_counts.keys()):
        name = CLASS_NAMES.get(c, f"类别{c}")
        print(f"  {name}: {label_counts[c]}个")
    
    # 检查是否有问题
    if len(set(all_preds)) == 1:
        print(f"\n⚠️ 警告: 所有预测都是同一个类别 ({CLASS_NAMES.get(all_preds[0], all_preds[0])})!")
        print(f"这说明模型输出有问题，可能原因:")
        print(f"  1. 模型结构与训练时不一致")
        print(f"  2. 权重加载不正确")
        print(f"  3. 数据预处理不一致")
        
        # 进一步诊断
        print(f"\n进一步诊断 - 单独测试Track模型:")
        with torch.no_grad():
            x_rd, x_track, x_stats, y = next(iter(val_loader))
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            
            track_logits = track_model(x_track, x_stats)
            track_probs = torch.softmax(track_logits, dim=1)
            
            print(f"  Track logits[0]: {track_logits[0].cpu().numpy()}")
            print(f"  Track probs[0]: {track_probs[0].cpu().numpy()}")
            print(f"  Track预测: {track_probs[0].argmax().item()}")
            
            # 检查所有样本的track预测
            track_preds = track_probs.argmax(dim=1).cpu().numpy()
            print(f"  这个batch的Track预测分布: {np.bincount(track_preds, minlength=6)}")


if __name__ == '__main__':
    main()