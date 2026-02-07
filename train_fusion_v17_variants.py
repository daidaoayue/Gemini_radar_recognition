"""
双流融合训练 V17 - 网络结构优化版
==================================
基于V16稳定结果，尝试多种网络结构改进

网络变体:
1. V5_SE: 添加通道注意力(Squeeze-and-Excitation)
2. V5_Split: 分组处理航迹特征和RCS特征
3. V5_Transformer: 用Transformer处理时序特征
4. V5_MultiScale: 多尺度卷积捕捉不同时间尺度
5. V5_Deep: 更深的网络 + 残差连接
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
import math
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


# ============================================================
# 方案1: SE注意力增强 (Squeeze-and-Excitation)
# ============================================================
class SEBlock(nn.Module):
    """通道注意力模块"""
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, C] or [B, C, L]
        if x.dim() == 3:
            b, c, _ = x.size()
            y = x.mean(dim=2)  # [B, C]
        else:
            b, c = x.size()
            y = x
        
        y = self.fc(y)  # [B, C]
        
        if x.dim() == 3:
            y = y.unsqueeze(2)  # [B, C, 1]
        
        return x * y


class TrackNetV5_SE(nn.Module):
    """SE注意力增强版"""
    def __init__(self, num_classes=6, stats_dim=46):
        super().__init__()
        
        # 时序特征 + SE注意力
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.temporal_se1 = SEBlock(64)
        
        self.temporal_conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.temporal_se2 = SEBlock(128)
        
        self.temporal_conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )
        
        # 统计特征 + SE注意力
        self.stats_fc1 = nn.Linear(stats_dim, 128)
        self.stats_bn1 = nn.BatchNorm1d(128)
        self.stats_se = SEBlock(128)
        
        self.stats_fc2 = nn.Sequential(
            nn.Linear(128, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256 + 192, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(192, num_classes)
        )
    
    def forward(self, x_temporal, x_stats):
        # 时序分支
        t = self.temporal_conv(x_temporal)
        t = self.temporal_se1(t)
        t = self.temporal_conv2(t)
        t = self.temporal_se2(t)
        t = self.temporal_conv3(t)  # [B, 256]
        
        # 统计分支
        s = F.relu(self.stats_bn1(self.stats_fc1(x_stats)))
        s = self.stats_se(s)
        s = self.stats_fc2(s)  # [B, 192]
        
        return self.classifier(torch.cat([t, s], dim=1))


# ============================================================
# 方案2: 分组特征处理 (航迹28维 + RCS18维 分开处理)
# ============================================================
class TrackNetV5_Split(nn.Module):
    """分组处理航迹和RCS特征"""
    def __init__(self, num_classes=6, stats_dim=46):
        super().__init__()
        self.track_dim = 28
        self.rcs_dim = stats_dim - 28 if stats_dim > 28 else 0
        
        # 时序特征处理
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
        
        # 航迹统计特征处理 (28维)
        self.track_stats_net = nn.Sequential(
            nn.Linear(self.track_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )  # -> [B, 128]
        
        # RCS特征处理 (18维) - 如果有的话
        if self.rcs_dim > 0:
            self.rcs_net = nn.Sequential(
                nn.Linear(self.rcs_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )  # -> [B, 64]
            classifier_input = 256 + 128 + 64
        else:
            self.rcs_net = None
            classifier_input = 256 + 128
        
        # 特征融合门控
        self.gate = nn.Sequential(
            nn.Linear(classifier_input, classifier_input // 2),
            nn.ReLU(),
            nn.Linear(classifier_input // 2, classifier_input),
            nn.Sigmoid()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(192, num_classes)
        )
    
    def forward(self, x_temporal, x_stats):
        # 时序特征
        feat_temporal = self.temporal_net(x_temporal)  # [B, 256]
        
        # 航迹统计特征
        feat_track = self.track_stats_net(x_stats[:, :self.track_dim])  # [B, 128]
        
        # RCS特征
        if self.rcs_net is not None and x_stats.shape[1] > self.track_dim:
            feat_rcs = self.rcs_net(x_stats[:, self.track_dim:])  # [B, 64]
            combined = torch.cat([feat_temporal, feat_track, feat_rcs], dim=1)
        else:
            combined = torch.cat([feat_temporal, feat_track], dim=1)
        
        # 门控机制
        gate = self.gate(combined)
        combined = combined * gate
        
        return self.classifier(combined)


# ============================================================
# 方案3: Transformer时序编码器
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TrackNetV5_Transformer(nn.Module):
    """Transformer处理时序特征"""
    def __init__(self, num_classes=6, stats_dim=46):
        super().__init__()
        
        # 时序特征: [B, 12, 16] -> [B, 16, 64]
        self.temporal_embed = nn.Linear(12, 64)
        self.pos_encoder = PositionalEncoding(64, max_len=20)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=256, 
            dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.temporal_pool = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # 统计特征
        self.stats_net = nn.Sequential(
            nn.Linear(stats_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256 + 192, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(192, num_classes)
        )
    
    def forward(self, x_temporal, x_stats):
        # x_temporal: [B, 12, 16] -> [B, 16, 12] -> [B, 16, 64]
        t = x_temporal.permute(0, 2, 1)  # [B, 16, 12]
        t = self.temporal_embed(t)        # [B, 16, 64]
        t = self.pos_encoder(t)
        t = self.transformer(t)           # [B, 16, 64]
        t = self.temporal_pool(t)         # [B, 256]
        
        s = self.stats_net(x_stats)       # [B, 192]
        
        return self.classifier(torch.cat([t, s], dim=1))


# ============================================================
# 方案4: 多尺度卷积
# ============================================================
class MultiScaleConv1d(nn.Module):
    """多尺度卷积模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 不同大小的卷积核
        self.conv1 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels - 2*(out_channels // 3), kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out = torch.cat([out1, out3, out5], dim=1)
        return F.relu(self.bn(out))


class TrackNetV5_MultiScale(nn.Module):
    """多尺度卷积版"""
    def __init__(self, num_classes=6, stats_dim=46):
        super().__init__()
        
        # 多尺度时序处理
        self.ms_conv1 = MultiScaleConv1d(12, 64)
        self.ms_conv2 = MultiScaleConv1d(64, 128)
        self.ms_conv3 = MultiScaleConv1d(128, 256)
        
        self.temporal_pool = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )
        
        # 统计特征
        self.stats_net = nn.Sequential(
            nn.Linear(stats_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256 + 192, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(192, num_classes)
        )
    
    def forward(self, x_temporal, x_stats):
        t = self.ms_conv1(x_temporal)
        t = F.dropout(t, 0.2, self.training)
        t = self.ms_conv2(t)
        t = F.dropout(t, 0.2, self.training)
        t = self.ms_conv3(t)
        t = self.temporal_pool(t)  # [B, 256]
        
        s = self.stats_net(x_stats)  # [B, 192]
        
        return self.classifier(torch.cat([t, s], dim=1))


# ============================================================
# 方案5: 更深的网络 + 残差连接
# ============================================================
class ResBlock1d(nn.Module):
    """1D残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class TrackNetV5_Deep(nn.Module):
    """更深的残差网络"""
    def __init__(self, num_classes=6, stats_dim=46):
        super().__init__()
        
        # 时序特征 - 更深的残差网络
        self.temporal_stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        self.temporal_res1 = ResBlock1d(64)
        self.temporal_res2 = ResBlock1d(64)
        
        self.temporal_down1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.temporal_res3 = ResBlock1d(128)
        self.temporal_res4 = ResBlock1d(128)
        
        self.temporal_down2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        self.temporal_res5 = ResBlock1d(256)
        
        self.temporal_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 统计特征 - 残差MLP
        self.stats_fc1 = nn.Linear(stats_dim, 192)
        self.stats_bn1 = nn.BatchNorm1d(192)
        self.stats_fc2 = nn.Linear(192, 192)
        self.stats_bn2 = nn.BatchNorm1d(192)
        self.stats_fc3 = nn.Linear(192, 192)
        self.stats_bn3 = nn.BatchNorm1d(192)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256 + 192, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x_temporal, x_stats):
        # 时序分支
        t = self.temporal_stem(x_temporal)
        t = self.temporal_res1(t)
        t = self.temporal_res2(t)
        t = self.temporal_down1(t)
        t = self.temporal_res3(t)
        t = self.temporal_res4(t)
        t = self.temporal_down2(t)
        t = self.temporal_res5(t)
        t = self.temporal_pool(t)  # [B, 256]
        
        # 统计分支 - 残差连接
        s = F.relu(self.stats_bn1(self.stats_fc1(x_stats)))
        s_residual = s
        s = F.relu(self.stats_bn2(self.stats_fc2(s)))
        s = self.stats_bn3(self.stats_fc3(s))
        s = F.relu(s + s_residual)  # [B, 192]
        
        return self.classifier(torch.cat([t, s], dim=1))


# ============================================================
# 原始V5网络（作为基准）
# ============================================================
class TrackNetV5_Base(nn.Module):
    """V16的基准网络"""
    def __init__(self, num_classes=6, stats_dim=46):
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
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 + 192, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(192, num_classes)
        )
    
    def forward(self, x_temporal, x_stats):
        t = self.temporal_net(x_temporal)
        s = self.stats_net(x_stats)
        return self.classifier(torch.cat([t, s], dim=1))


# ============================================================
# 网络选择器
# ============================================================
NETWORK_VARIANTS = {
    'base': TrackNetV5_Base,
    'se': TrackNetV5_SE,
    'split': TrackNetV5_Split,
    'transformer': TrackNetV5_Transformer,
    'multiscale': TrackNetV5_MultiScale,
    'deep': TrackNetV5_Deep,
}


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


def train_variant(variant_name, device, train_loader, val_loader, rd_model, 
                  stats_dim, epochs=60, class_weights=None, valid_classes=[0,1,2,3]):
    """训练指定变体"""
    from tqdm import tqdm
    
    print(f"\n{'='*60}")
    print(f"训练网络变体: {variant_name.upper()}")
    print(f"{'='*60}")
    
    # 创建模型
    ModelClass = NETWORK_VARIANTS[variant_name]
    track_model = ModelClass(num_classes=6, stats_dim=stats_dim)
    track_model.to(device)
    
    # 打印参数量
    total_params = sum(p.numel() for p in track_model.parameters())
    print(f"参数量: {total_params:,}")
    
    # 优化器
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = optim.AdamW(track_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0
    best_bird_acc = 0
    best_coverage = 0
    best_epoch = 0
    best_weight = 0.4
    best_state = None
    
    # 外层epoch进度条
    epoch_pbar = tqdm(range(epochs), desc=f"{variant_name}", ncols=100)
    
    for epoch in epoch_pbar:
        # 训练
        track_model.train()
        train_loss = 0
        
        # 内层batch进度条
        batch_pbar = tqdm(train_loader, desc=f"Ep{epoch+1:02d} Train", 
                          ncols=80, leave=False)
        for x_rd, x_track, x_stats, y in batch_pbar:
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            loss = criterion(track_model(x_track, x_stats), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        # 验证
        track_model.eval()
        weight_results = {}
        
        val_pbar = tqdm(val_loader, desc=f"Ep{epoch+1:02d} Val", 
                        ncols=80, leave=False)
        
        with torch.no_grad():
            # 收集所有预测
            all_preds = {w: [] for w in [0.35, 0.4, 0.45, 0.5]}
            all_confs = {w: [] for w in [0.35, 0.4, 0.45, 0.5]}
            all_labels = []
            
            for x_rd, x_track, x_stats, y in val_pbar:
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
            
            # 计算各权重结果
            all_labels = np.array(all_labels)
            for track_w in [0.35, 0.4, 0.45, 0.5]:
                preds = np.array(all_preds[track_w])
                confs = np.array(all_confs[track_w])
                
                mask_valid = np.isin(all_labels, valid_classes)
                mask_high = confs >= 0.5
                mask = mask_valid & mask_high
                
                total_all = mask_valid.sum()
                total_high = mask.sum()
                correct_high = (preds[mask] == all_labels[mask]).sum()
                
                # 鸟类
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
        
        # 更新进度条显示
        epoch_pbar.set_postfix({
            'Acc': f'{acc:.2f}%', 
            'Bird': f'{bird:.1f}%', 
            'Cov': f'{cov:.1f}%',
            'Best': f'{best_acc:.2f}%'
        })
        
        if acc > best_acc:
            best_acc = acc
            best_bird_acc = bird
            best_coverage = cov
            best_epoch = epoch + 1
            best_weight = best_w_this
            best_state = track_model.state_dict().copy()
    
    # 训练完成总结
    print(f"\n  ✓ {variant_name.upper()} 完成: Acc={best_acc:.2f}% | Bird={best_bird_acc:.1f}% | Cov={best_coverage:.1f}% | Ep={best_epoch}")
    
    return {
        'name': variant_name,
        'acc': best_acc,
        'bird': best_bird_acc,
        'cov': best_coverage,
        'epoch': best_epoch,
        'weight': best_weight,
        'state': best_state,
        'params': total_params
    }


def main():
    # 配置
    RD_TRAIN = "./dataset/train_cleandata/train"
    RD_VAL = "./dataset/train_cleandata/val"
    
    if os.path.exists("./dataset/track_enhanced_v5_cleandata"):
        TRACK_TRAIN = "./dataset/track_enhanced_v5_cleandata/train"
        TRACK_VAL = "./dataset/track_enhanced_v5_cleandata/val"
    elif os.path.exists("./dataset/track_enhanced_v4_cleandata"):
        TRACK_TRAIN = "./dataset/track_enhanced_v4_cleandata/train"
        TRACK_VAL = "./dataset/track_enhanced_v4_cleandata/val"
    else:
        TRACK_TRAIN = "./dataset/track_enhanced_cleandata/train"
        TRACK_VAL = "./dataset/track_enhanced_cleandata/val"
    
    SAVE_DIR = "./checkpoint/fusion_v17_variants"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 检测维度
    stats_dim = detect_stats_dim(TRACK_TRAIN.replace('/train', ''))
    print(f"特征维度: {stats_dim}")
    
    # 加载数据
    print("\n加载数据...")
    train_ds = FusionDataLoader(RD_TRAIN, TRACK_TRAIN, val=False, stats_dim=stats_dim)
    val_ds = FusionDataLoader(RD_VAL, TRACK_VAL, val=True, stats_dim=stats_dim)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # 加载RD模型
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
    
    # 类别权重
    class_weights = torch.tensor([1.0, 1.0, 2.0, 1.0, 0.5, 0.5], device=device)
    
    # 要测试的变体
    variants_to_test = ['base', 'se', 'split', 'multiscale', 'deep', 'transformer']
    
    print(f"\n将测试 {len(variants_to_test)} 种网络变体: {', '.join(variants_to_test)}")
    print("="*70)
    
    # 训练所有变体
    results = []
    for idx, variant in enumerate(variants_to_test):
        print(f"\n[{idx+1}/{len(variants_to_test)}] ", end="")
        result = train_variant(
            variant, device, train_loader, val_loader, rd_model,
            stats_dim, epochs=40, class_weights=class_weights
        )
        results.append(result)
        
        # 保存每个变体的最佳模型
        save_path = os.path.join(
            SAVE_DIR, 
            f'ckpt_{variant}_ep{result["epoch"]}_{result["acc"]:.2f}_bird{result["bird"]:.0f}_cov{result["cov"]:.0f}.pth'
        )
        torch.save({
            'track_model': result['state'],
            'variant': variant,
            'best_acc': result['acc'],
            'best_bird_acc': result['bird'],
            'best_coverage': result['cov'],
            'best_epoch': result['epoch'],
            'best_weight': result['weight'],
            'stats_dim': stats_dim,
        }, save_path)
    
    # 汇总结果
    print("\n" + "="*70)
    print("网络变体对比结果")
    print("="*70)
    print(f"{'变体':^12}|{'参数量':^12}|{'准确率':^10}|{'鸟类':^8}|{'覆盖率':^8}|{'最佳Epoch':^10}")
    print("-"*70)
    
    # 按准确率排序
    results.sort(key=lambda x: x['acc'], reverse=True)
    
    for r in results:
        mark = " ★" if r['acc'] == results[0]['acc'] else ""
        print(f"{r['name']:^12}|{r['params']:^12,}|{r['acc']:^10.2f}%|{r['bird']:^8.1f}%|{r['cov']:^8.1f}%|{r['epoch']:^10}{mark}")
    
    # 最佳结果
    best = results[0]
    print(f"\n最佳变体: {best['name'].upper()}")
    print(f"  准确率: {best['acc']:.2f}%")
    print(f"  鸟类: {best['bird']:.1f}%")
    print(f"  覆盖率: {best['cov']:.1f}%")


if __name__ == '__main__':
    main()