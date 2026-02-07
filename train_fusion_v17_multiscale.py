"""
双流融合训练 V17 - MultiScale版
================================
基于网络变体测试结果，multiscale是最佳选择：
- 覆盖率93.0% (最高)
- 准确率99.21%
- 参数量最少246K
- 训练稳定（Epoch=14）

多尺度卷积原理：
- 1x1卷积：捕捉瞬时特征
- 3x3卷积：捕捉短期模式
- 5x5卷积：捕捉长期趋势
三种尺度并行处理，拼接融合
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import warnings
import glob
import scipy.io as scio
from tqdm import tqdm

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


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


class MultiScaleConv1d(nn.Module):
    """多尺度卷积模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 三种尺度的卷积核
        self.conv1 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels - 2*(out_channels // 3), kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        out1 = self.conv1(x)  # 瞬时特征
        out3 = self.conv3(x)  # 短期模式
        out5 = self.conv5(x)  # 长期趋势
        out = torch.cat([out1, out3, out5], dim=1)
        return F.relu(self.bn(out))


class TrackNetV17_MultiScale(nn.Module):
    """V17多尺度卷积网络"""
    def __init__(self, num_classes=6, stats_dim=46):
        super().__init__()
        
        # 多尺度时序处理
        self.ms_conv1 = MultiScaleConv1d(12, 64)
        self.ms_conv2 = MultiScaleConv1d(64, 128)
        self.ms_conv3 = MultiScaleConv1d(128, 256)
        
        self.temporal_pool = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )  # -> [B, 256]
        
        # 统计特征处理
        self.stats_net = nn.Sequential(
            nn.Linear(stats_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
        )  # -> [B, 192]
        
        # 分类器 (256 + 192 = 448)
        self.classifier = nn.Sequential(
            nn.Linear(256 + 192, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(192, num_classes)
        )
    
    def forward(self, x_temporal, x_stats):
        # 多尺度时序处理
        t = self.ms_conv1(x_temporal)
        t = F.dropout(t, 0.2, self.training)
        t = self.ms_conv2(t)
        t = F.dropout(t, 0.2, self.training)
        t = self.ms_conv3(t)
        t = self.temporal_pool(t)  # [B, 256]
        
        # 统计特征
        s = self.stats_net(x_stats)  # [B, 192]
        
        return self.classifier(torch.cat([t, s], dim=1))


def detect_stats_dim(track_feat_dir):
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
                        return len(mat['track_stats'].flatten())
                except:
                    pass
    return 46


def train():
    # ==================== 配置 ====================
    RD_TRAIN = "./dataset/train_cleandata/train"
    RD_VAL = "./dataset/train_cleandata/val"
    
    if os.path.exists("./dataset/track_enhanced_v5_cleandata"):
        TRACK_TRAIN = "./dataset/track_enhanced_v5_cleandata/train"
        TRACK_VAL = "./dataset/track_enhanced_v5_cleandata/val"
        print("使用V5特征目录 (46维)")
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
    EPOCHS = 60  # 多训练一些，multiscale在14 epoch达到最佳
    LR = 1e-3
    
    CLASS_WEIGHTS = [1.0, 1.0, 2.0, 1.0, 0.5, 0.5]
    
    SAVE_DIR = "./checkpoint/fusion_v17_multiscale"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # ==================== 检测特征维度 ====================
    stats_dim = detect_stats_dim(TRACK_TRAIN.replace('/train', ''))
    print(f"特征维度: {stats_dim}")
    
    # ==================== 数据加载 ====================
    print("\n加载数据...")
    train_ds = FusionDataLoader(RD_TRAIN, TRACK_TRAIN, val=False, stats_dim=stats_dim)
    val_ds = FusionDataLoader(RD_VAL, TRACK_VAL, val=True, stats_dim=stats_dim)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # ==================== 加载RD模型 ====================
    print("\n加载RD模型...")
    rd_model = rsnet34()
    
    rd_ckpts = glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth")
    rd_ckpts = [p for p in rd_ckpts if 'fusion' not in p]
    
    if rd_ckpts:
        rd_ckpt = torch.load(rd_ckpts[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
        print(f"  已加载: {rd_ckpts[0]}")
    
    rd_model.to(device)
    rd_model.eval()
    for p in rd_model.parameters():
        p.requires_grad = False
    
    # ==================== 创建Track模型 ====================
    print(f"\n创建MultiScale模型...")
    track_model = TrackNetV17_MultiScale(num_classes=6, stats_dim=stats_dim)
    track_model.to(device)
    
    total_params = sum(p.numel() for p in track_model.parameters())
    print(f"  参数量: {total_params:,}")
    
    # ==================== 损失函数和优化器 ====================
    class_weights = torch.tensor(CLASS_WEIGHTS, device=device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    optimizer = optim.AdamW(track_model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # ==================== 训练循环 ====================
    best_acc = 0
    best_bird_acc = 0
    best_coverage = 0
    best_epoch = 0
    best_weight = 0.4
    
    print("\n开始训练...")
    print("="*70)
    
    for epoch in range(EPOCHS):
        # --- 训练 ---
        track_model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Ep{epoch+1:02d}", ncols=80, leave=False)
        for x_rd, x_track, x_stats, y in pbar:
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            loss = criterion(track_model(x_track, x_stats), y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        # --- 验证 ---
        track_model.eval()
        weight_results = {}
        
        with torch.no_grad():
            all_preds = {w: [] for w in [0.35, 0.4, 0.45, 0.5]}
            all_confs = {w: [] for w in [0.35, 0.4, 0.45, 0.5]}
            all_labels = []
            
            for x_rd, x_track, x_stats, y in val_loader:
                x_rd = x_rd.to(device)
                x_track = x_track.to(device)
                x_stats = x_stats.to(device)
                
                rd_probs = torch.softmax(rd_model(x_rd), dim=1)
                track_probs = torch.softmax(track_model(x_track, x_stats), dim=1)
                
                for track_w in [0.35, 0.4, 0.45, 0.5]:
                    rd_w = 1.0 - track_w
                    fused_probs = rd_w * rd_probs + track_w * track_probs
                    conf, pred = fused_probs.max(dim=1)
                    all_preds[track_w].extend(pred.cpu().numpy())
                    all_confs[track_w].extend(conf.cpu().numpy())
                
                all_labels.extend(y.numpy())
            
            all_labels = np.array(all_labels)
            
            for track_w in [0.35, 0.4, 0.45, 0.5]:
                preds = np.array(all_preds[track_w])
                confs = np.array(all_confs[track_w])
                
                mask_valid = np.isin(all_labels, VALID_CLASSES)
                mask_high = confs >= 0.5
                mask = mask_valid & mask_high
                
                total_all = mask_valid.sum()
                total_high = mask.sum()
                correct_high = (preds[mask] == all_labels[mask]).sum()
                
                mask_bird = mask & (all_labels == 2)
                bird_total = mask_bird.sum()
                bird_correct = (preds[mask_bird] == 2).sum() if bird_total > 0 else 0
                
                if total_high > 0:
                    acc = 100 * correct_high / total_high
                    cov = 100 * total_high / total_all
                    bird = 100 * bird_correct / max(bird_total, 1)
                    weight_results[track_w] = (acc, cov, bird)
        
        # 找最佳权重
        best_w_this = max(weight_results.keys(), key=lambda w: weight_results[w][0])
        acc, cov, bird = weight_results[best_w_this]
        
        # 打印
        mark = ""
        if acc > best_acc or (acc == best_acc and cov > best_coverage):
            best_acc = acc
            best_bird_acc = bird
            best_coverage = cov
            best_epoch = epoch + 1
            best_weight = best_w_this
            mark = " *"
            
            # 保存最佳模型
            torch.save({
                'track_model': track_model.state_dict(),
                'best_acc': best_acc,
                'best_bird_acc': best_bird_acc,
                'best_coverage': best_coverage,
                'best_epoch': best_epoch,
                'best_weight': best_weight,
                'stats_dim': stats_dim,
                'version': 'v17_multiscale'
            }, os.path.join(SAVE_DIR, f'ckpt_best_ep{best_epoch}_{best_acc:.2f}_bird{best_bird_acc:.0f}_cov{best_coverage:.0f}.pth'))
        
        if (epoch + 1) % 5 == 0 or mark:
            print(f"Ep{epoch+1:2d} | Loss:{train_loss/len(train_loader):.4f} | "
                  f"Acc:{acc:.2f}% | Bird:{bird:.1f}% | Cov:{cov:.1f}% | W:{best_w_this}{mark}")
    
    print("="*70)
    print(f"训练完成!")
    print(f"  最佳Epoch: {best_epoch}")
    print(f"  准确率: {best_acc:.2f}%")
    print(f"  鸟类: {best_bird_acc:.1f}%")
    print(f"  覆盖率: {best_coverage:.1f}%")
    print(f"  Track权重: {best_weight}")
    print(f"  模型: ckpt_best_ep{best_epoch}_{best_acc:.2f}_bird{best_bird_acc:.0f}_cov{best_coverage:.0f}.pth")


if __name__ == '__main__':
    train()