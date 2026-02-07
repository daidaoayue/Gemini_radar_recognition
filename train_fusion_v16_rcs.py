"""
双流融合训练 V16
================
基于V15成功框架 + RCS/多普勒谱宽特征

特征维度:
- 原V15: 28维 (达到99.53%准确率)
- V16: 46维 (28维航迹 + 18维RCS/Doppler)

新增RCS/Doppler特征 (29-46):
  29. rcs_mean          - RCS功率均值
  30. rcs_peak_mean     - 峰值功率均值
  31. rcs_std           - RCS功率标准差 ★鸟类区分关键
  32. rcs_peak_std      - 峰值功率标准差
  33. rcs_range         - RCS动态范围 ★
  34. rcs_peak_range    - 峰值动态范围
  35. rcs_diff_mean     - 帧间RCS变化均值
  36. rcs_diff_max      - 帧间RCS变化最大值
  37. rcs_cv            - RCS变异系数 ★
  38. doppler_width_mean    - 多普勒谱宽均值 ★
  39. doppler_width_std     - 多普勒谱宽标准差
  40. doppler_width_max     - 多普勒谱宽最大值
  41. bandwidth_3db_mean    - 3dB带宽均值
  42. bandwidth_3db_std     - 3dB带宽标准差
  43. energy_conc_mean      - 能量集中度均值 ★
  44. energy_conc_std       - 能量集中度标准差
  45. peak_mean_ratio       - 峰均比均值
  46. peak_mean_ratio_std   - 峰均比标准差
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import warnings
import glob
import scipy.io as scio

warnings.filterwarnings("ignore")

# 数据加载器（复用V4，支持动态stats_dim）
try:
    from data_loader_fusion_v4 import FusionDataLoaderV4 as FusionDataLoader
except ImportError:
    from data_loader_fusion_v3 import FusionDataLoaderV3 as FusionDataLoader

try:
    from drsncww import rsnet34
except ImportError:
    print("错误: 找不到 drsncww.py")
    exit()


class FocalLoss(nn.Module):
    """Focal Loss - 关注困难样本"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class TrackOnlyNetV5(nn.Module):
    """
    航迹网络V5 - 支持46维统计特征
    
    相比V4的改进:
    1. 增加stats_net容量以处理更多特征
    2. 添加特征分组处理（航迹特征 vs RCS特征）
    """
    def __init__(self, num_classes=6, stats_dim=46):
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
        )  # -> [B, 256]
        
        # 统计特征处理 [stats_dim] -> 192维
        # 增加网络容量以处理46维特征
        self.stats_net = nn.Sequential(
            nn.Linear(stats_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(192, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
        )  # -> [B, 192]
        
        # 分类器 (256 + 192 = 448维)
        self.classifier = nn.Sequential(
            nn.Linear(256 + 192, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(192, num_classes)
        )
    
    def forward(self, x_temporal, x_stats):
        feat_temporal = self.temporal_net(x_temporal)  # [B, 256]
        feat_stats = self.stats_net(x_stats)           # [B, 192]
        
        combined = torch.cat([feat_temporal, feat_stats], dim=1)  # [B, 448]
        return self.classifier(combined)


class TrackOnlyNetV4(nn.Module):
    """航迹网络V4 - 兼容28维（与你原有的V15一致）"""
    def __init__(self, num_classes=6, stats_dim=28):
        super().__init__()
        self.stats_dim = stats_dim
        
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
        combined = torch.cat([feat_temporal, feat_stats], dim=1)
        return self.classifier(combined)


def detect_stats_dim(track_feat_dir):
    """自动检测统计特征维度"""
    for split in ['train', 'val']:
        split_dir = os.path.join(track_feat_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        for label in range(6):
            label_dir = os.path.join(split_dir, str(label))
            if not os.path.exists(label_dir):
                continue
            
            mat_files = glob.glob(os.path.join(label_dir, '*.mat'))
            if mat_files:
                try:
                    mat = scio.loadmat(mat_files[0])
                    if 'track_stats' in mat:
                        stats = mat['track_stats'].flatten()
                        print(f"检测到统计特征维度: {len(stats)}")
                        return len(stats)
                except:
                    pass
    
    print("无法检测维度，使用默认值28")
    return 28


def analyze_rcs_features(val_loader, device, stats_dim):
    """分析RCS特征在各类别的分布（仅当stats_dim>=46时）"""
    if stats_dim < 46:
        print("当前特征维度不包含RCS特征，跳过分析")
        return
    
    print("\n" + "="*60)
    print("RCS/多普勒特征分布分析")
    print("="*60)
    
    class_features = {c: [] for c in [0, 1, 2, 3]}
    
    for x_rd, x_track, x_stats, y in val_loader:
        for i in range(len(y)):
            c = y[i].item()
            if c in class_features:
                # 提取RCS特征 (29-46, 即索引28-45)
                rcs_feat = x_stats[i, 28:46].numpy()
                class_features[c].append(rcs_feat)
    
    feature_names = [
        'rcs_mean', 'rcs_peak_mean', 'rcs_std★', 'rcs_peak_std',
        'rcs_range★', 'rcs_peak_range', 'rcs_diff_mean', 'rcs_diff_max', 'rcs_cv★',
        'dop_width★', 'dop_width_std', 'dop_width_max',
        'bw_3db_mean', 'bw_3db_std',
        'energy_conc★', 'energy_conc_std',
        'peak_ratio', 'peak_ratio_std'
    ]
    
    class_names = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
    
    print(f"\n{'特征':18s}|", end='')
    for c in [0, 1, 2, 3]:
        print(f"{class_names[c]:^10s}|", end='')
    print(" 鸟类差异")
    print("-" * 75)
    
    for feat_idx, feat_name in enumerate(feature_names):
        print(f"{feat_name:18s}|", end='')
        values = []
        for c in [0, 1, 2, 3]:
            if class_features[c]:
                arr = np.array(class_features[c])[:, feat_idx]
                mean_val = np.mean(arr)
                values.append(mean_val)
                print(f"{mean_val:^10.2f}|", end='')
            else:
                values.append(0)
                print(f"{'N/A':^10s}|", end='')
        
        # 计算鸟类与无人机的差异
        if len(values) >= 3 and values[2] != 0:
            bird_val = values[2]
            drone_avg = (values[0] + values[1]) / 2
            diff_pct = 100 * (bird_val - drone_avg) / (abs(drone_avg) + 1e-6)
            if abs(diff_pct) > 20:
                print(f" {diff_pct:+.0f}% ★")
            else:
                print(f" {diff_pct:+.0f}%")
        else:
            print()


def train():
    # ==================== 配置 ====================
    RD_TRAIN = "./dataset/train_cleandata/train"
    RD_VAL = "./dataset/train_cleandata/val"
    
    # 尝试V5特征目录（46维），否则用V4（28维）
    if os.path.exists("./dataset/track_enhanced_v5_cleandata"):
        TRACK_TRAIN = "./dataset/track_enhanced_v5_cleandata/train"
        TRACK_VAL = "./dataset/track_enhanced_v5_cleandata/val"
        print("使用V5特征目录 (46维，含RCS/Doppler)")
    elif os.path.exists("./dataset/track_enhanced_v4_cleandata"):
        TRACK_TRAIN = "./dataset/track_enhanced_v4_cleandata/train"
        TRACK_VAL = "./dataset/track_enhanced_v4_cleandata/val"
        print("使用V4特征目录 (28维)")
    else:
        TRACK_TRAIN = "./dataset/track_enhanced_cleandata/train"
        TRACK_VAL = "./dataset/track_enhanced_cleandata/val"
        print("使用默认特征目录")
    
    VALID_CLASSES = [0, 1, 2, 3]
    BATCH_SIZE = 32
    EPOCHS = 60
    LR = 1e-3
    
    # 类别权重: 鸟类(2)权重2.0
    CLASS_WEIGHTS = [1.0, 1.0, 2.0, 1.0, 0.5, 0.5]
    
    SAVE_DIR = "./checkpoint/fusion_v16_rcs"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # ==================== 检测特征维度 ====================
    stats_dim = detect_stats_dim(TRACK_TRAIN.replace('/train', ''))
    
    # ==================== 数据加载 ====================
    print("\n加载数据...")
    train_ds = FusionDataLoader(RD_TRAIN, TRACK_TRAIN, val=False, stats_dim=stats_dim)
    val_ds = FusionDataLoader(RD_VAL, TRACK_VAL, val=True, stats_dim=stats_dim)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # ==================== 分析RCS特征分布 ====================
    if stats_dim >= 46:
        analyze_rcs_features(val_loader, device, stats_dim)
    
    # ==================== 加载RD模型 ====================
    print("\n加载RD模型...")
    rd_model = rsnet34()
    
    rd_ckpts = glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth")
    rd_ckpts = [p for p in rd_ckpts if 'fusion' not in p]
    
    if rd_ckpts:
        rd_ckpt = torch.load(rd_ckpts[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
        print(f"   已加载: {rd_ckpts[0]}")
    
    rd_model.to(device)
    rd_model.eval()
    for p in rd_model.parameters():
        p.requires_grad = False
    
    # ==================== Track模型 ====================
    print(f"\n创建Track模型 (stats_dim={stats_dim})...")
    
    # 根据维度选择网络
    if stats_dim >= 46:
        track_model = TrackOnlyNetV5(num_classes=6, stats_dim=stats_dim)
        print("  使用V5网络结构（增强版）")
    else:
        track_model = TrackOnlyNetV4(num_classes=6, stats_dim=stats_dim)
        print("  使用V4网络结构")
    
    track_model.to(device)
    
    # ==================== 损失函数和优化器 ====================
    class_weights = torch.tensor(CLASS_WEIGHTS, device=device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    optimizer = optim.AdamW(track_model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # ==================== 训练循环 ====================
    best_acc = 0
    best_bird_acc = 0
    best_weight = 0.4
    best_coverage = 0
    best_epoch = 0
    
    print("\n开始训练...")
    print("="*70)
    
    for epoch in range(EPOCHS):
        # --- 训练 ---
        track_model.train()
        train_loss = 0
        
        for x_rd, x_track, x_stats, y in train_loader:
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            track_logits = track_model(x_track, x_stats)
            loss = criterion(track_logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # --- 验证 ---
        track_model.eval()
        weight_results = {}
        
        with torch.no_grad():
            for track_w in [0.3, 0.35, 0.4, 0.45, 0.5]:
                rd_w = 1.0 - track_w
                
                correct_high = 0
                total_high = 0
                correct_all = 0
                total_all = 0
                bird_correct = 0
                bird_total = 0
                
                for x_rd, x_track, x_stats, y in val_loader:
                    x_rd = x_rd.to(device)
                    x_track = x_track.to(device)
                    x_stats = x_stats.to(device)
                    
                    rd_probs = torch.softmax(rd_model(x_rd), dim=1)
                    track_probs = torch.softmax(track_model(x_track, x_stats), dim=1)
                    fused_probs = rd_w * rd_probs + track_w * track_probs
                    
                    conf, pred = fused_probs.max(dim=1)
                    
                    for i in range(len(y)):
                        if y[i].item() not in VALID_CLASSES:
                            continue
                        
                        total_all += 1
                        if pred[i].item() == y[i].item():
                            correct_all += 1
                        
                        if conf[i].item() >= 0.5:
                            total_high += 1
                            if pred[i].item() == y[i].item():
                                correct_high += 1
                            
                            # 鸟类统计
                            if y[i].item() == 2:
                                bird_total += 1
                                if pred[i].item() == 2:
                                    bird_correct += 1
                
                if total_high > 0:
                    acc_high = 100 * correct_high / total_high
                    coverage = 100 * total_high / total_all
                    bird_acc = 100 * bird_correct / max(bird_total, 1)
                    weight_results[track_w] = (acc_high, coverage, bird_acc, bird_total)
        
        # 找最佳权重
        best_w_this = max(weight_results.keys(), key=lambda w: weight_results[w][0])
        acc_high, coverage, bird_acc, bird_n = weight_results[best_w_this]
        
        # 打印
        if (epoch + 1) % 5 == 0 or acc_high > best_acc:
            print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | "
                  f"Acc: {acc_high:.2f}% | Bird: {bird_acc:.1f}% | Cov: {coverage:.1f}% | W: {best_w_this}")
        
        # 保存最佳
        if acc_high > best_acc:
            best_acc = acc_high
            best_bird_acc = bird_acc
            best_weight = best_w_this
            best_coverage = coverage
            best_epoch = epoch + 1
            
            torch.save({
                'track_model': track_model.state_dict(),
                'best_acc': best_acc,
                'best_bird_acc': best_bird_acc,
                'best_coverage': best_coverage,
                'best_fixed_weight': best_weight,
                'stats_dim': stats_dim,
                'epoch': best_epoch,
                'version': 'v16_rcs'
            }, os.path.join(SAVE_DIR, f'ckpt_best_ep{best_epoch}_{best_acc:.2f}_bird{best_bird_acc:.0f}_cov{best_coverage:.0f}.pth'))
    
    print("="*70)
    print(f"训练完成!")
    print(f"  最佳Epoch: {best_epoch}")
    print(f"  最佳准确率: {best_acc:.2f}%")
    print(f"  鸟类准确率: {best_bird_acc:.1f}%")
    print(f"  覆盖率: {best_coverage:.1f}%")
    print(f"  最佳Track权重: {best_weight}")
    print(f"  统计特征维度: {stats_dim}")
    print(f"  保存目录: {SAVE_DIR}")
    print(f"  模型文件: ckpt_best_ep{best_epoch}_{best_acc:.2f}_bird{best_bird_acc:.0f}_cov{best_coverage:.0f}.pth")


if __name__ == '__main__':
    train()