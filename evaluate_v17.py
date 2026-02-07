"""
模型验证脚本 - V17 MultiScale+SE
================================
在验证集上全面评估模型性能，输出：
1. 各类别准确率、召回率、F1
2. 混淆矩阵
3. 高/低置信度样本分析
4. 鸟类误分类详情
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import warnings
import glob
import scipy.io as scio
from collections import defaultdict

warnings.filterwarnings("ignore")

try:
    from data_loader_fusion_v4 import FusionDataLoaderV4 as FusionDataLoader
except ImportError:
    from data_loader_fusion_v3 import FusionDataLoaderV3 as FusionDataLoader

try:
    from drsncww import rsnet34
except ImportError:
    print("错误: 找不到 drsncww.py")
    exit()


# ==================== 网络定义 ====================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = x.mean(dim=2)
        y = self.fc(y).unsqueeze(2)
        return x * y


class MultiScaleSEConv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels - 2*(out_channels // 3), kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
    
    def forward(self, x):
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out = torch.cat([out1, out3, out5], dim=1)
        out = F.relu(self.bn(out))
        out = self.se(out)
        return out


class TrackNetV17_MultiScaleSE(nn.Module):
    def __init__(self, num_classes=6, stats_dim=46):
        super().__init__()
        
        self.ms_se_conv1 = MultiScaleSEConv1d(12, 64)
        self.ms_se_conv2 = MultiScaleSEConv1d(64, 128)
        self.ms_se_conv3 = MultiScaleSEConv1d(128, 256)
        
        self.temporal_pool = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )
        
        self.stats_fc1 = nn.Linear(stats_dim, 128)
        self.stats_bn1 = nn.BatchNorm1d(128)
        self.stats_se = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.Sigmoid()
        )
        
        self.stats_fc2 = nn.Sequential(
            nn.Linear(128, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 + 192, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(192, num_classes)
        )
    
    def forward(self, x_temporal, x_stats):
        t = self.ms_se_conv1(x_temporal)
        t = F.dropout(t, 0.2, self.training)
        t = self.ms_se_conv2(t)
        t = F.dropout(t, 0.2, self.training)
        t = self.ms_se_conv3(t)
        t = self.temporal_pool(t)
        
        s = F.relu(self.stats_bn1(self.stats_fc1(x_stats)))
        s_attn = self.stats_se(s)
        s = s * s_attn
        s = self.stats_fc2(s)
        
        return self.classifier(torch.cat([t, s], dim=1))


def evaluate():
    # ==================== 配置 ====================
    RD_VAL = "./dataset/train_cleandata/val"
    
    if os.path.exists("./dataset/track_enhanced_v5_cleandata"):
        TRACK_VAL = "./dataset/track_enhanced_v5_cleandata/val"
    elif os.path.exists("./dataset/track_enhanced_v4_cleandata"):
        TRACK_VAL = "./dataset/track_enhanced_v4_cleandata/val"
    else:
        TRACK_VAL = "./dataset/track_enhanced_cleandata/val"
    
    # 查找最佳模型
    model_dir = "./checkpoint/fusion_v17_multiscale_se"
    model_files = glob.glob(os.path.join(model_dir, "ckpt_best_*.pth"))
    
    if not model_files:
        print("错误: 找不到训练好的模型!")
        print(f"请确认目录: {model_dir}")
        return
    
    # 选择最新的模型
    model_path = sorted(model_files)[-1]
    
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球', 4: '杂波', 5: '其他'}
    CONF_THRESH = 0.5
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("V17 MultiScale+SE 模型验证")
    print("="*70)
    print(f"设备: {device}")
    print(f"模型: {model_path}")
    
    # ==================== 加载模型 ====================
    print("\n加载模型...")
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    stats_dim = checkpoint.get('stats_dim', 46)
    track_weight = checkpoint.get('best_weight', 0.5)
    
    print(f"  特征维度: {stats_dim}")
    print(f"  Track权重: {track_weight}")
    print(f"  训练时准确率: {checkpoint.get('best_acc', 'N/A')}")
    print(f"  训练时覆盖率: {checkpoint.get('best_coverage', 'N/A')}")
    
    # RD模型
    rd_model = rsnet34()
    rd_ckpts = glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth")
    rd_ckpts = [p for p in rd_ckpts if 'fusion' not in p]
    if rd_ckpts:
        rd_ckpt = torch.load(rd_ckpts[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
    rd_model.to(device)
    rd_model.eval()
    
    # Track模型
    track_model = TrackNetV17_MultiScaleSE(num_classes=6, stats_dim=stats_dim)
    track_model.load_state_dict(checkpoint['track_model'])
    track_model.to(device)
    track_model.eval()
    
    # ==================== 加载数据 ====================
    print("\n加载验证数据...")
    val_ds = FusionDataLoader(RD_VAL, TRACK_VAL, val=True, stats_dim=stats_dim)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # ==================== 评估 ====================
    print("\n开始评估...")
    
    all_labels = []
    all_preds = []
    all_confs = []
    all_rd_preds = []
    all_track_preds = []
    
    rd_w = 1.0 - track_weight
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            
            rd_probs = torch.softmax(rd_model(x_rd), dim=1)
            track_probs = torch.softmax(track_model(x_track, x_stats), dim=1)
            fused_probs = rd_w * rd_probs + track_weight * track_probs
            
            conf, pred = fused_probs.max(dim=1)
            _, rd_pred = rd_probs.max(dim=1)
            _, track_pred = track_probs.max(dim=1)
            
            all_labels.extend(y.numpy())
            all_preds.extend(pred.cpu().numpy())
            all_confs.extend(conf.cpu().numpy())
            all_rd_preds.extend(rd_pred.cpu().numpy())
            all_track_preds.extend(track_pred.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_confs = np.array(all_confs)
    all_rd_preds = np.array(all_rd_preds)
    all_track_preds = np.array(all_track_preds)
    
    # ==================== 整体结果 ====================
    print("\n" + "="*70)
    print("整体评估结果")
    print("="*70)
    
    # 有效类别mask
    mask_valid = np.isin(all_labels, VALID_CLASSES)
    mask_high = all_confs >= CONF_THRESH
    mask_low = all_confs < CONF_THRESH
    
    # 全部样本
    total_valid = mask_valid.sum()
    correct_all = (all_preds[mask_valid] == all_labels[mask_valid]).sum()
    acc_all = 100 * correct_all / total_valid
    
    # 高置信度样本
    mask_high_valid = mask_valid & mask_high
    total_high = mask_high_valid.sum()
    correct_high = (all_preds[mask_high_valid] == all_labels[mask_high_valid]).sum()
    acc_high = 100 * correct_high / total_high
    coverage = 100 * total_high / total_valid
    
    # 低置信度样本
    mask_low_valid = mask_valid & mask_low
    total_low = mask_low_valid.sum()
    correct_low = (all_preds[mask_low_valid] == all_labels[mask_low_valid]).sum() if total_low > 0 else 0
    acc_low = 100 * correct_low / total_low if total_low > 0 else 0
    
    print(f"\n{'指标':<20} {'值':>15}")
    print("-"*40)
    print(f"{'有效样本总数':<20} {total_valid:>15}")
    print(f"{'全部样本准确率':<20} {acc_all:>14.2f}%")
    print(f"{'高置信度样本数':<20} {total_high:>15}")
    print(f"{'高置信度准确率':<20} {acc_high:>14.2f}%")
    print(f"{'覆盖率':<20} {coverage:>14.2f}%")
    print(f"{'低置信度样本数':<20} {total_low:>15}")
    print(f"{'低置信度准确率':<20} {acc_low:>14.2f}%")
    
    # ==================== 各类别详细分析 ====================
    print("\n" + "="*70)
    print("各类别详细分析（高置信度样本）")
    print("="*70)
    
    print(f"\n{'类别':<12} {'样本数':>8} {'正确':>8} {'准确率':>10} {'召回率':>10} {'F1':>10}")
    print("-"*60)
    
    for c in VALID_CLASSES:
        # 该类别的真实样本（高置信度）
        mask_class = mask_high_valid & (all_labels == c)
        total_class = mask_class.sum()
        correct_class = (all_preds[mask_class] == c).sum()
        
        # 预测为该类别的样本
        mask_pred_class = mask_high_valid & (all_preds == c)
        pred_class = mask_pred_class.sum()
        true_positive = (all_labels[mask_pred_class] == c).sum()
        
        # 计算指标
        precision = 100 * true_positive / pred_class if pred_class > 0 else 0
        recall = 100 * correct_class / total_class if total_class > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{CLASS_NAMES[c]:<12} {total_class:>8} {correct_class:>8} {precision:>9.2f}% {recall:>9.2f}% {f1:>9.2f}%")
    
    # ==================== 混淆矩阵 ====================
    print("\n" + "="*70)
    print("混淆矩阵（高置信度样本）")
    print("="*70)
    
    # 构建混淆矩阵
    conf_matrix = np.zeros((4, 4), dtype=int)
    for i, c_true in enumerate(VALID_CLASSES):
        for j, c_pred in enumerate(VALID_CLASSES):
            mask = mask_high_valid & (all_labels == c_true) & (all_preds == c_pred)
            conf_matrix[i, j] = mask.sum()
    
    # 打印
    header = "真实\\预测"
    print(f"\n{header:<12}", end="")
    for c in VALID_CLASSES:
        print(f"{CLASS_NAMES[c][:4]:>10}", end="")
    print(f"{'总计':>10}")
    print("-"*65)
    
    for i, c_true in enumerate(VALID_CLASSES):
        print(f"{CLASS_NAMES[c_true]:<12}", end="")
        for j in range(4):
            val = conf_matrix[i, j]
            if i == j:
                print(f"{val:>10}", end="")  # 对角线（正确）
            elif val > 0:
                print(f"{val:>10}", end="")  # 误分类
            else:
                print(f"{'-':>10}", end="")
        print(f"{conf_matrix[i].sum():>10}")
    
    # ==================== 鸟类误分类分析 ====================
    print("\n" + "="*70)
    print("鸟类误分类详细分析")
    print("="*70)
    
    # 鸟类被误分类的情况
    mask_bird = mask_high_valid & (all_labels == 2)
    bird_total = mask_bird.sum()
    bird_correct = (all_preds[mask_bird] == 2).sum()
    bird_wrong = bird_total - bird_correct
    
    print(f"\n鸟类样本总数（高置信度）: {bird_total}")
    print(f"正确识别: {bird_correct} ({100*bird_correct/bird_total:.2f}%)")
    print(f"误分类: {bird_wrong} ({100*bird_wrong/bird_total:.2f}%)")
    
    if bird_wrong > 0:
        print(f"\n误分类去向:")
        for c in VALID_CLASSES:
            if c == 2:
                continue
            mask_wrong = mask_bird & (all_preds == c)
            count = mask_wrong.sum()
            if count > 0:
                print(f"  → {CLASS_NAMES[c]}: {count} ({100*count/bird_wrong:.1f}%)")
    
    # 其他类别被误认为鸟类
    print(f"\n被误认为鸟类的样本:")
    for c in VALID_CLASSES:
        if c == 2:
            continue
        mask_as_bird = mask_high_valid & (all_labels == c) & (all_preds == 2)
        count = mask_as_bird.sum()
        if count > 0:
            total_c = (mask_high_valid & (all_labels == c)).sum()
            print(f"  {CLASS_NAMES[c]} → 鸟类: {count} ({100*count/total_c:.1f}%)")
    
    # ==================== RD vs Track 对比 ====================
    print("\n" + "="*70)
    print("RD模型 vs Track模型 对比")
    print("="*70)
    
    # RD模型单独准确率
    rd_correct = (all_rd_preds[mask_high_valid] == all_labels[mask_high_valid]).sum()
    rd_acc = 100 * rd_correct / total_high
    
    # Track模型单独准确率
    track_correct = (all_track_preds[mask_high_valid] == all_labels[mask_high_valid]).sum()
    track_acc = 100 * track_correct / total_high
    
    print(f"\n{'模型':<20} {'准确率':>15}")
    print("-"*40)
    print(f"{'RD模型单独':<20} {rd_acc:>14.2f}%")
    print(f"{'Track模型单独':<20} {track_acc:>14.2f}%")
    print(f"{'融合模型':<20} {acc_high:>14.2f}%")
    print(f"{'融合提升':<20} {acc_high - max(rd_acc, track_acc):>+14.2f}%")
    
    # ==================== 低置信度样本分析 ====================
    if total_low > 0:
        print("\n" + "="*70)
        print("低置信度样本分析")
        print("="*70)
        
        print(f"\n低置信度样本数: {total_low} ({100*total_low/total_valid:.1f}%)")
        print(f"低置信度准确率: {acc_low:.2f}%")
        
        print(f"\n低置信度样本类别分布:")
        for c in VALID_CLASSES:
            mask_low_class = mask_low_valid & (all_labels == c)
            count = mask_low_class.sum()
            if count > 0:
                correct = (all_preds[mask_low_class] == c).sum()
                print(f"  {CLASS_NAMES[c]}: {count} 个, 准确率 {100*correct/count:.1f}%")
    
    # ==================== 总结 ====================
    print("\n" + "="*70)
    print("验证总结")
    print("="*70)
    
    print(f"""
┌─────────────────────────────────────┐
│  高置信度准确率:  {acc_high:>6.2f}%           │
│  覆盖率:          {coverage:>6.2f}%           │
│  鸟类准确率:      {100*bird_correct/bird_total:>6.2f}%           │
│  低置信度样本:    {total_low:>6} ({100*total_low/total_valid:.1f}%)       │
└─────────────────────────────────────┘
    """)
    
    # 与训练时对比
    train_acc = checkpoint.get('best_acc', 0)
    train_cov = checkpoint.get('best_coverage', 0)
    
    if train_acc > 0:
        print(f"与训练时对比:")
        print(f"  准确率: {train_acc:.2f}% → {acc_high:.2f}% ({acc_high-train_acc:+.2f}%)")
        print(f"  覆盖率: {train_cov:.2f}% → {coverage:.2f}% ({coverage-train_cov:+.2f}%)")


if __name__ == '__main__':
    evaluate()