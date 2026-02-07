"""
低置信度样本专用分类器 V17-Cascade
===================================
策略：
1. 用主模型筛选出低置信度样本
2. 单独训练一个针对困难样本的分类器
3. 推理时：高置信度用主模型，低置信度用级联分类器

目标：提升低置信度样本的52%准确率
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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


# ==================== 主模型定义 ====================
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
        out = torch.cat([self.conv1(x), self.conv3(x), self.conv5(x)], dim=1)
        return self.se(F.relu(self.bn(out)))


class TrackNetV17_MultiScaleSE(nn.Module):
    def __init__(self, num_classes=6, stats_dim=46):
        super().__init__()
        self.ms_se_conv1 = MultiScaleSEConv1d(12, 64)
        self.ms_se_conv2 = MultiScaleSEConv1d(64, 128)
        self.ms_se_conv3 = MultiScaleSEConv1d(128, 256)
        self.temporal_pool = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten())
        
        self.stats_fc1 = nn.Linear(stats_dim, 128)
        self.stats_bn1 = nn.BatchNorm1d(128)
        self.stats_se = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 128), nn.Sigmoid())
        self.stats_fc2 = nn.Sequential(nn.Linear(128, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(0.2))
        
        self.classifier = nn.Sequential(
            nn.Linear(256 + 192, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(0.3),
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
        s = s * self.stats_se(s)
        s = self.stats_fc2(s)
        
        return self.classifier(torch.cat([t, s], dim=1))
    
    def get_features(self, x_temporal, x_stats):
        """获取融合特征（用于级联分类器）"""
        t = self.ms_se_conv1(x_temporal)
        t = self.ms_se_conv2(t)
        t = self.ms_se_conv3(t)
        t = self.temporal_pool(t)
        
        s = F.relu(self.stats_bn1(self.stats_fc1(x_stats)))
        s = s * self.stats_se(s)
        s = self.stats_fc2(s)
        
        return torch.cat([t, s], dim=1)  # [B, 448]


# ==================== 级联分类器 ====================
class CascadeClassifier(nn.Module):
    """专门处理低置信度样本的分类器"""
    def __init__(self, input_dim=448+6+6, num_classes=4):
        """
        输入：
        - 主模型特征 448维
        - RD模型概率 6维
        - Track模型概率 6维
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def collect_low_conf_samples():
    """收集低置信度样本用于训练级联分类器"""
    
    # 配置
    RD_TRAIN = "./dataset/train_cleandata/train"
    RD_VAL = "./dataset/train_cleandata/val"
    
    if os.path.exists("./dataset/track_enhanced_v5_cleandata"):
        TRACK_TRAIN = "./dataset/track_enhanced_v5_cleandata/train"
        TRACK_VAL = "./dataset/track_enhanced_v5_cleandata/val"
    else:
        TRACK_TRAIN = "./dataset/track_enhanced_v4_cleandata/train"
        TRACK_VAL = "./dataset/track_enhanced_v4_cleandata/val"
    
    CONF_THRESH = 0.5
    TRACK_WEIGHT = 0.5
    VALID_CLASSES = [0, 1, 2, 3]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 加载主模型
    print("\n加载主模型...")
    
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
    model_dir = "./checkpoint/fusion_v17_multiscale_se"
    model_files = glob.glob(os.path.join(model_dir, "ckpt_best_*.pth"))
    if not model_files:
        print("错误: 找不到V17模型!")
        return None, None, None, None
    
    model_path = sorted(model_files, key=os.path.getmtime)[-1]
    print(f"  Track模型: {os.path.basename(model_path)}")
    
    track_model = TrackNetV17_MultiScaleSE(num_classes=6, stats_dim=46)
    checkpoint = torch.load(model_path, map_location='cpu')
    track_model.load_state_dict(checkpoint['track_model'])
    track_model.to(device)
    track_model.eval()
    
    # 收集样本
    print("\n收集低置信度样本...")
    
    low_conf_features = []
    low_conf_labels = []
    all_features = []
    all_labels = []
    all_confs = []
    
    rd_w = 1.0 - TRACK_WEIGHT
    
    for split, rd_dir, track_dir in [('train', RD_TRAIN, TRACK_TRAIN), ('val', RD_VAL, TRACK_VAL)]:
        print(f"  处理 {split} 集...")
        
        ds = FusionDataLoader(rd_dir, track_dir, val=True, stats_dim=46)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        
        with torch.no_grad():
            for x_rd, x_track, x_stats, y in tqdm(loader, desc=split, ncols=80):
                x_rd = x_rd.to(device)
                x_track = x_track.to(device)
                x_stats = x_stats.to(device)
                
                # 获取概率
                rd_probs = torch.softmax(rd_model(x_rd), dim=1)
                track_probs = torch.softmax(track_model(x_track, x_stats), dim=1)
                fused_probs = rd_w * rd_probs + TRACK_WEIGHT * track_probs
                
                # 获取特征
                track_features = track_model.get_features(x_track, x_stats)
                
                # 组合特征：track特征 + rd概率 + track概率
                combined_features = torch.cat([
                    track_features,
                    rd_probs,
                    track_probs
                ], dim=1)  # [B, 448+6+6=460]
                
                conf, pred = fused_probs.max(dim=1)
                
                for i in range(len(y)):
                    label = y[i].item()
                    if label not in VALID_CLASSES:
                        continue
                    
                    feat = combined_features[i].cpu().numpy()
                    c = conf[i].item()
                    
                    all_features.append(feat)
                    all_labels.append(label)
                    all_confs.append(c)
                    
                    # 低置信度样本
                    if c < CONF_THRESH:
                        low_conf_features.append(feat)
                        low_conf_labels.append(label)
    
    print(f"\n收集完成:")
    print(f"  总样本数: {len(all_features)}")
    print(f"  低置信度样本: {len(low_conf_features)} ({100*len(low_conf_features)/len(all_features):.1f}%)")
    
    # 转换为tensor
    X_low = torch.tensor(np.array(low_conf_features), dtype=torch.float32)
    y_low = torch.tensor(low_conf_labels, dtype=torch.long)
    X_all = torch.tensor(np.array(all_features), dtype=torch.float32)
    y_all = torch.tensor(all_labels, dtype=torch.long)
    confs_all = torch.tensor(all_confs, dtype=torch.float32)
    
    return X_low, y_low, X_all, y_all, confs_all


def train_cascade_classifier(X_low, y_low, X_all, y_all, confs_all):
    """训练级联分类器"""
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*60)
    print("训练级联分类器")
    print("="*60)
    
    # 数据增强：对低置信度样本过采样
    print(f"低置信度样本数: {len(X_low)}")
    
    # 类别平衡
    class_counts = {}
    for c in range(4):
        class_counts[c] = (y_low == c).sum().item()
        print(f"  类别 {c}: {class_counts[c]}")
    
    max_count = max(class_counts.values())
    
    # 过采样到平衡
    X_balanced = []
    y_balanced = []
    
    for c in range(4):
        mask = y_low == c
        X_c = X_low[mask]
        y_c = y_low[mask]
        
        n_c = len(X_c)
        if n_c == 0:
            continue
        
        # 重复采样到max_count
        repeats = max_count // n_c + 1
        X_c_repeated = X_c.repeat(repeats, 1)[:max_count]
        y_c_repeated = y_c.repeat(repeats)[:max_count]
        
        X_balanced.append(X_c_repeated)
        y_balanced.append(y_c_repeated)
    
    X_train = torch.cat(X_balanced, dim=0)
    y_train = torch.cat(y_balanced, dim=0)
    
    print(f"平衡后训练样本: {len(X_train)}")
    
    # 验证集：使用低置信度的验证集样本
    # 简单起见，用20%做验证
    n_val = len(X_low) // 5
    indices = torch.randperm(len(X_low))
    X_val = X_low[indices[:n_val]]
    y_val = y_low[indices[:n_val]]
    
    # 创建DataLoader
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    # 模型
    cascade_model = CascadeClassifier(input_dim=460, num_classes=4)
    cascade_model.to(device)
    
    # 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(cascade_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    best_acc = 0
    best_state = None
    
    print("\n开始训练...")
    
    for epoch in range(50):
        cascade_model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            logits = cascade_model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # 验证
        cascade_model.eval()
        with torch.no_grad():
            X_val_d = X_val.to(device)
            y_val_d = y_val.to(device)
            
            logits = cascade_model(X_val_d)
            pred = logits.argmax(dim=1)
            acc = 100 * (pred == y_val_d).float().mean().item()
        
        if acc > best_acc:
            best_acc = acc
            best_state = cascade_model.state_dict().copy()
            mark = " *"
        else:
            mark = ""
        
        if (epoch + 1) % 10 == 0 or mark:
            print(f"Ep{epoch+1:2d} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%{mark}")
    
    print(f"\n级联分类器最佳准确率: {best_acc:.2f}%")
    
    # 保存
    save_dir = "./checkpoint/fusion_v17_cascade"
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save({
        'cascade_model': best_state,
        'best_acc': best_acc,
    }, os.path.join(save_dir, f'cascade_classifier_{best_acc:.1f}.pth'))
    
    print(f"模型已保存到: {save_dir}")
    
    return cascade_model, best_acc


def evaluate_with_cascade():
    """使用级联策略评估"""
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 加载所有模型
    print("\n加载模型...")
    
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
    model_dir = "./checkpoint/fusion_v17_multiscale_se"
    model_files = glob.glob(os.path.join(model_dir, "ckpt_best_*.pth"))
    model_path = sorted(model_files, key=os.path.getmtime)[-1]
    
    track_model = TrackNetV17_MultiScaleSE(num_classes=6, stats_dim=46)
    checkpoint = torch.load(model_path, map_location='cpu')
    track_model.load_state_dict(checkpoint['track_model'])
    track_model.to(device)
    track_model.eval()
    
    # 级联分类器
    cascade_dir = "./checkpoint/fusion_v17_cascade"
    cascade_files = glob.glob(os.path.join(cascade_dir, "cascade_*.pth"))
    
    if not cascade_files:
        print("错误: 找不到级联分类器!")
        return
    
    cascade_path = sorted(cascade_files)[-1]
    cascade_model = CascadeClassifier(input_dim=460, num_classes=4)
    cascade_ckpt = torch.load(cascade_path, map_location='cpu')
    cascade_model.load_state_dict(cascade_ckpt['cascade_model'])
    cascade_model.to(device)
    cascade_model.eval()
    
    print(f"  级联分类器: {os.path.basename(cascade_path)}")
    
    # 数据
    RD_VAL = "./dataset/train_cleandata/val"
    TRACK_VAL = "./dataset/track_enhanced_v5_cleandata/val"
    
    val_ds = FusionDataLoader(RD_VAL, TRACK_VAL, val=True, stats_dim=46)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # 评估
    print("\n评估级联策略...")
    
    CONF_THRESH = 0.5
    TRACK_WEIGHT = 0.5
    VALID_CLASSES = [0, 1, 2, 3]
    rd_w = 1.0 - TRACK_WEIGHT
    
    # 统计
    high_conf_correct = 0
    high_conf_total = 0
    low_conf_correct_main = 0
    low_conf_correct_cascade = 0
    low_conf_total = 0
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            
            # 主模型预测
            rd_probs = torch.softmax(rd_model(x_rd), dim=1)
            track_probs = torch.softmax(track_model(x_track, x_stats), dim=1)
            fused_probs = rd_w * rd_probs + TRACK_WEIGHT * track_probs
            
            conf, main_pred = fused_probs.max(dim=1)
            
            # 级联分类器输入
            track_features = track_model.get_features(x_track, x_stats)
            cascade_input = torch.cat([track_features, rd_probs, track_probs], dim=1)
            cascade_logits = cascade_model(cascade_input)
            cascade_pred = cascade_logits.argmax(dim=1)
            
            for i in range(len(y)):
                label = y[i].item()
                if label not in VALID_CLASSES:
                    continue
                
                c = conf[i].item()
                
                if c >= CONF_THRESH:
                    # 高置信度：用主模型
                    high_conf_total += 1
                    if main_pred[i].item() == label:
                        high_conf_correct += 1
                else:
                    # 低置信度：对比主模型和级联分类器
                    low_conf_total += 1
                    if main_pred[i].item() == label:
                        low_conf_correct_main += 1
                    if cascade_pred[i].item() == label:
                        low_conf_correct_cascade += 1
    
    # 结果
    print("\n" + "="*60)
    print("级联策略评估结果")
    print("="*60)
    
    high_acc = 100 * high_conf_correct / high_conf_total
    low_acc_main = 100 * low_conf_correct_main / low_conf_total
    low_acc_cascade = 100 * low_conf_correct_cascade / low_conf_total
    
    print(f"\n高置信度样本 ({high_conf_total}):")
    print(f"  主模型准确率: {high_acc:.2f}%")
    
    print(f"\n低置信度样本 ({low_conf_total}):")
    print(f"  主模型准确率:     {low_acc_main:.2f}%")
    print(f"  级联分类器准确率: {low_acc_cascade:.2f}%")
    print(f"  提升: {low_acc_cascade - low_acc_main:+.2f}%")
    
    # 整体
    total = high_conf_total + low_conf_total
    correct_main = high_conf_correct + low_conf_correct_main
    correct_cascade = high_conf_correct + low_conf_correct_cascade
    
    print(f"\n整体准确率:")
    print(f"  仅主模型:   {100*correct_main/total:.2f}%")
    print(f"  级联策略:   {100*correct_cascade/total:.2f}%")


def main():
    print("="*60)
    print("V17 级联分类器训练")
    print("="*60)
    
    # 1. 收集低置信度样本
    result = collect_low_conf_samples()
    if result[0] is None:
        return
    
    X_low, y_low, X_all, y_all, confs_all = result
    
    # 2. 训练级联分类器
    cascade_model, best_acc = train_cascade_classifier(X_low, y_low, X_all, y_all, confs_all)
    
    # 3. 评估级联策略
    evaluate_with_cascade()


if __name__ == '__main__':
    main()