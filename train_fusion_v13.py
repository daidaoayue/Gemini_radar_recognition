"""
双流融合训练脚本 V13 (预训练冻结版)
===================================
策略：
1. 先单独训练航迹模型到最佳(~83%)
2. 融合时冻结航迹模型，只学习融合权重
3. 这样避免航迹分支在融合训练中"学坏"

使用方法：
1. 先运行 train_track_only_v3.py 训练航迹模型
2. 保存最佳权重
3. 在本脚本中加载该权重
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import math

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

from data_loader_fusion_v3 import FusionDataLoaderV3

try:
    from drsncww import rsnet34
except ImportError:
    print("错误: 找不到 drsncww.py")
    exit()


# ================= 预训练航迹模型（与train_track_only_v3相同结构）=================
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


# ================= 双冻结融合模型 =================
class DualFrozenFusion(nn.Module):
    """
    双冻结融合：RD和航迹模型都冻结，只学习融合策略
    """
    
    def __init__(self, rd_model, track_model, num_classes=6):
        super().__init__()
        
        self.rd_model = rd_model
        self.track_model = track_model
        self.num_classes = num_classes
        
        # 冻结两个模型
        for p in self.rd_model.parameters():
            p.requires_grad = False
        for p in self.track_model.parameters():
            p.requires_grad = False
        
        # 可学习的融合参数
        # 方案1: 固定权重（直接测试不同权重）
        # 方案2: 可学习的动态权重
        
        # 使用一个小网络来预测融合权重
        self.fusion_net = nn.Sequential(
            nn.Linear(num_classes * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 初始化让航迹权重较小
        with torch.no_grad():
            self.fusion_net[-2].bias.fill_(-1.0)  # sigmoid(-1) ≈ 0.27
    
    def forward(self, x_rd, x_track, x_stats):
        # 1. 两个模型都是eval模式
        self.rd_model.eval()
        self.track_model.eval()
        
        with torch.no_grad():
            rd_logits = self.rd_model(x_rd)
            track_logits = self.track_model(x_track, x_stats)
        
        # 2. 计算融合权重
        combined_logits = torch.cat([rd_logits, track_logits], dim=1)
        track_weight = self.fusion_net(combined_logits)  # [B, 1]
        track_weight = track_weight * 0.4  # 最大权重0.4
        rd_weight = 1.0 - track_weight
        
        # 3. 概率融合
        rd_probs = torch.softmax(rd_logits, dim=1)
        track_probs = torch.softmax(track_logits, dim=1)
        fused_probs = rd_weight * rd_probs + track_weight * track_probs
        
        # 转回logits
        fused_logits = torch.log(fused_probs + 1e-8)
        
        return fused_logits, rd_logits, track_logits, track_weight.mean()


def train_track_model(device, train_loader, val_loader, epochs=30, conf_thresh=0.5):
    """训练航迹模型"""
    print("\n" + "="*60)
    print("第一阶段：训练航迹模型")
    print("="*60)
    
    model = TrackOnlyNetV3(num_classes=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        
        pbar = tqdm(train_loader, desc=f"Track Ep{epoch:02d}", ncols=60, leave=False)
        for x_rd, x_track, x_stats, y in pbar:
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            out = model(x_track, x_stats)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
        
        scheduler.step()
        train_acc = 100 * correct / total
        
        # 验证
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x_rd, x_track, x_stats, y in val_loader:
                x_track = x_track.to(device)
                x_stats = x_stats.to(device)
                y = y.to(device)
                
                out = model(x_track, x_stats)
                out_softmax = torch.softmax(out, dim=1)
                conf, pred = out_softmax.max(dim=1)
                mask = conf >= conf_thresh
                if mask.sum() > 0:
                    val_correct += pred[mask].eq(y[mask]).sum().item()
                    val_total += mask.sum().item()
        
        val_acc = 100 * val_correct / max(val_total, 1)
        
        mark = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()
            mark = "*"
        
        print(f"  Epoch {epoch:2d} | TrainAcc: {train_acc:.2f}% | ValAcc: {val_acc:.2f}% | Best: {best_acc:.2f}% {mark}")
    
    # 加载最佳权重
    model.load_state_dict(best_state)
    print(f"\n航迹模型训练完成，最佳准确率: {best_acc:.2f}%")
    
    return model, best_acc


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 路径配置
    RD_TRAIN = "./dataset/train/2026-1-14/train"
    RD_VAL = "./dataset/train/2026-1-14/val"
    TRACK_TRAIN = "./dataset/track_enhanced/train"
    TRACK_VAL = "./dataset/track_enhanced/val"
    PRETRAINED_PATH = "./checkpoint/ckpt_best_1_93.17.pth"
    SAVE_DIR = "./checkpoint/fusion_v13"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    if not os.path.exists(PRETRAINED_PATH):
        import glob
        pths = glob.glob("./checkpoint/*93*.pth") + glob.glob("./checkpoint/*94*.pth")
        if pths:
            PRETRAINED_PATH = pths[0]
    
    BATCH_SIZE = 16
    CONF_THRESH = 0.5
    
    print(f"\n{'='*60}")
    print(f"双流融合训练 V13 (预训练冻结版)")
    print(f"{'='*60}")
    print(f"设备: {DEVICE}")
    
    # Step 1: 加载数据
    print(f"\n[Step 1] 加载数据...")
    train_ds = FusionDataLoaderV3(RD_TRAIN, TRACK_TRAIN, val=False)
    val_ds = FusionDataLoaderV3(RD_VAL, TRACK_VAL, val=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Step 2: 加载RD模型
    print(f"\n[Step 2] 加载RD预训练模型...")
    rd_model = rsnet34()
    checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')
    state_dict = checkpoint['net_weight'] if 'net_weight' in checkpoint else checkpoint
    rd_model.load_state_dict(state_dict, strict=True)
    rd_model.to(DEVICE)
    rd_model.eval()
    print(f"已加载: {os.path.basename(PRETRAINED_PATH)}")
    
    # 验证RD基线
    rd_correct, rd_total = 0, 0
    with torch.no_grad():
        for x_rd, _, _, y in val_loader:
            x_rd, y = x_rd.to(DEVICE), y.to(DEVICE)
            out = rd_model(x_rd)
            out_softmax = torch.softmax(out, dim=1)
            conf, pred = out_softmax.max(dim=1)
            mask = conf >= CONF_THRESH
            if mask.sum() > 0:
                rd_correct += pred[mask].eq(y[mask]).sum().item()
                rd_total += mask.sum().item()
    rd_baseline = 100 * rd_correct / max(rd_total, 1)
    print(f"RD基线准确率: {rd_baseline:.2f}%")
    
    # Step 3: 训练航迹模型
    track_model, track_baseline = train_track_model(DEVICE, train_loader, val_loader, epochs=30, conf_thresh=CONF_THRESH)
    
    # Step 4: 测试不同融合权重（不训练）
    print("\n" + "="*60)
    print("第二阶段：测试固定权重融合")
    print("="*60)
    
    best_fusion_acc = 0
    best_weight = 0
    
    rd_model.eval()
    track_model.eval()
    
    for track_weight in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        rd_weight = 1.0 - track_weight
        
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for x_rd, x_track, x_stats, y in val_loader:
                x_rd = x_rd.to(DEVICE)
                x_track = x_track.to(DEVICE)
                x_stats = x_stats.to(DEVICE)
                y = y.to(DEVICE)
                
                rd_logits = rd_model(x_rd)
                track_logits = track_model(x_track, x_stats)
                
                rd_probs = torch.softmax(rd_logits, dim=1)
                track_probs = torch.softmax(track_logits, dim=1)
                
                fused_probs = rd_weight * rd_probs + track_weight * track_probs
                
                conf, pred = fused_probs.max(dim=1)
                mask = conf >= CONF_THRESH
                if mask.sum() > 0:
                    val_correct += pred[mask].eq(y[mask]).sum().item()
                    val_total += mask.sum().item()
        
        fusion_acc = 100 * val_correct / max(val_total, 1)
        
        mark = ""
        if fusion_acc > best_fusion_acc:
            best_fusion_acc = fusion_acc
            best_weight = track_weight
            mark = " <-- Best"
        
        print(f"  track_weight={track_weight:.2f} | 融合准确率: {fusion_acc:.2f}%{mark}")
    
    print(f"\n最佳融合权重: track={best_weight:.2f}, rd={1-best_weight:.2f}")
    print(f"最佳融合准确率: {best_fusion_acc:.2f}%")
    
    # Step 5: 用可学习融合网络微调
    print("\n" + "="*60)
    print("第三阶段：可学习融合网络微调")
    print("="*60)
    
    fusion_model = DualFrozenFusion(rd_model, track_model, num_classes=6).to(DEVICE)
    
    trainable = [p for p in fusion_model.parameters() if p.requires_grad]
    print(f"可训练参数: {sum(p.numel() for p in trainable)}")
    
    optimizer = optim.Adam(trainable, lr=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_learned_acc = best_fusion_acc
    
    print(f"\n{'Ep':^4}|{'Loss':^8}|{'ValAcc':^8}|{'TrkW':^8}|{'Best':^8}")
    print("-" * 45)
    
    for epoch in range(20):
        fusion_model.train()
        run_loss = 0
        track_weights = []
        
        pbar = tqdm(train_loader, desc=f"Ep{epoch:02d}", ncols=50, leave=False)
        for x_rd, x_track, x_stats, y in pbar:
            x_rd = x_rd.to(DEVICE)
            x_track = x_track.to(DEVICE)
            x_stats = x_stats.to(DEVICE)
            y = y.to(DEVICE)
            
            optimizer.zero_grad()
            fused, _, _, tw = fusion_model(x_rd, x_track, x_stats)
            loss = criterion(fused, y)
            loss.backward()
            optimizer.step()
            
            run_loss += loss.item()
            track_weights.append(tw.item())
        
        avg_loss = run_loss / len(train_loader)
        avg_tw = np.mean(track_weights)
        
        # 验证
        fusion_model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x_rd, x_track, x_stats, y in val_loader:
                x_rd = x_rd.to(DEVICE)
                x_track = x_track.to(DEVICE)
                x_stats = x_stats.to(DEVICE)
                y = y.to(DEVICE)
                
                fused, _, _, _ = fusion_model(x_rd, x_track, x_stats)
                fused_probs = torch.softmax(fused, dim=1)
                conf, pred = fused_probs.max(dim=1)
                mask = conf >= CONF_THRESH
                if mask.sum() > 0:
                    val_correct += pred[mask].eq(y[mask]).sum().item()
                    val_total += mask.sum().item()
        
        val_acc = 100 * val_correct / max(val_total, 1)
        
        mark = ""
        if val_acc > best_learned_acc:
            best_learned_acc = val_acc
            torch.save({
                'fusion_model': fusion_model.state_dict(),
                'track_model': track_model.state_dict(),
                'val_acc': val_acc,
            }, f"{SAVE_DIR}/ckpt_best_{val_acc:.2f}.pth")
            mark = "*"
        
        print(f"{epoch:^4}|{avg_loss:^8.4f}|{val_acc:^8.2f}|{avg_tw:^8.3f}|{best_learned_acc:^8.2f} {mark}")
    
    # 最终总结
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"   RD基线:        {rd_baseline:.2f}%")
    print(f"   航迹基线:      {track_baseline:.2f}%")
    print(f"   固定权重融合:  {best_fusion_acc:.2f}% (track_weight={best_weight:.2f})")
    print(f"   可学习融合:    {best_learned_acc:.2f}%")
    print(f"   相比RD提升:    {best_learned_acc - rd_baseline:+.2f}%")


if __name__ == '__main__':
    main()