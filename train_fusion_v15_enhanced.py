"""
双流融合训练 V15
================
基于问题航迹分析的特征增强版本

改进:
1. 支持28维航迹统计特征（新增8维运动稳定性特征）
2. 保持Focal Loss和鸟类权重增强
3. 自动检测特征维度（兼容20维和28维）

新增特征:
  21. stability_score  - 运动稳定性综合指数
  22. curvature_mean   - 运动曲率均值
  23. curvature_max    - 运动曲率最大值
  24. curvature_std    - 运动曲率标准差
  25. velocity_cv      - 速度变异系数
  26. vz_ratio         - 垂直运动比例
  27. vel_fft_peak     - 速度FFT主频幅度
  28. direction_consistency - 运动方向一致性
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

# 尝试导入V4，否则用V3
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
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 聚焦参数
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
        # 新增特征可能有更大的方差，增加网络容量
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
        # x_temporal: [B, 12, 16]
        # x_stats: [B, stats_dim]
        
        feat_temporal = self.temporal_net(x_temporal)  # [B, 256]
        feat_stats = self.stats_net(x_stats)           # [B, 128]
        
        combined = torch.cat([feat_temporal, feat_stats], dim=1)  # [B, 384]
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
    
    print("无法检测维度，使用默认值20")
    return 20


def train():
    # ==================== 配置 ====================
    RD_TRAIN = "./dataset/train_cleandata/train"
    RD_VAL = "./dataset/train_cleandata/val"
    
    # 强制使用V3目录（先测试）
    # TRACK_TRAIN = "./dataset/track_enhanced_cleandata/train"
    # TRACK_VAL = "./dataset/track_enhanced_cleandata/val"
    # 尝试V4特征目录，否则用V3
    if os.path.exists("./dataset/track_enhanced_v4_cleandata"):
        TRACK_TRAIN = "./dataset/track_enhanced_v4_cleandata/train"
        TRACK_VAL = "./dataset/track_enhanced_v4_cleandata/val"
        print("使用V4特征目录")
    else:
        TRACK_TRAIN = "./dataset/track_enhanced_cleandata/train"
        TRACK_VAL = "./dataset/track_enhanced_cleandata/val"
        print("使用V3特征目录")
    
    VALID_CLASSES = [0, 1, 2, 3]
    BATCH_SIZE = 32
    EPOCHS = 60
    LR = 1e-3
    
    # 类别权重: 鸟类(2)权重2.0
    CLASS_WEIGHTS = [1.0, 1.0, 2.0, 1.0, 0.5, 0.5]
    
    SAVE_DIR = "./checkpoint/fusion_v15_enhanced"
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
    track_model = TrackOnlyNetV4(num_classes=6, stats_dim=stats_dim)
    track_model.to(device)
    
    # ==================== 损失函数和优化器 ====================
    class_weights = torch.tensor(CLASS_WEIGHTS, device=device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    optimizer = optim.AdamW(track_model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # ==================== 训练循环 ====================
    best_acc = 0
    best_weight = 0.4
    
    print("\n开始训练...")
    print("="*60)
    
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
        
        # 测试不同融合权重
        weight_results = {}
        
        with torch.no_grad():
            for track_w in [0.3, 0.35, 0.4, 0.45, 0.5]:
                rd_w = 1.0 - track_w
                
                correct_high = 0
                total_high = 0
                correct_all = 0
                total_all = 0
                
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
                
                if total_high > 0:
                    acc_high = 100 * correct_high / total_high
                    coverage = 100 * total_high / total_all
                    weight_results[track_w] = (acc_high, coverage, total_high)
        
        # 找最佳权重
        best_w_this = max(weight_results.keys(), key=lambda w: weight_results[w][0])
        acc_high, coverage, n_high = weight_results[best_w_this]
        
        # 打印
        if (epoch + 1) % 5 == 0 or acc_high > best_acc:
            print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | "
                  f"Acc: {acc_high:.2f}% | Cov: {coverage:.1f}% | W: {best_w_this}")
        
        # 保存最佳
        if acc_high > best_acc:
            best_acc = acc_high
            best_weight = best_w_this
            
            torch.save({
                'track_model': track_model.state_dict(),
                'best_acc': best_acc,
                'best_fixed_weight': best_weight,
                'stats_dim': stats_dim,
                'epoch': epoch + 1,
            }, os.path.join(SAVE_DIR, f'ckpt_best_{best_acc:.2f}.pth'))
    
    print("="*60)
    print(f"训练完成!")
    print(f"最佳准确率: {best_acc:.2f}%")
    print(f"最佳Track权重: {best_weight}")
    print(f"统计特征维度: {stats_dim}")


if __name__ == '__main__':
    train()