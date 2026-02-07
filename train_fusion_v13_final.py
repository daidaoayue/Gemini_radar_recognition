"""
双流融合训练脚本 V13 (最终版)
==============================
改进：
1. 固定权重测试范围扩大到0.50
2. 默认最佳权重 track_weight=0.40
3. 支持动态权重调整（基于RD置信度）
4. 只统计类别0-3
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


# ================= 航迹模型 =================
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


# ================= 动态权重融合模型 =================
class DynamicFusionModel(nn.Module):
    """
    动态权重融合模型
    根据RD置信度动态调整航迹权重：
    - RD置信度高 -> 航迹权重低
    - RD置信度低 -> 航迹权重高
    """
    
    def __init__(self, rd_model, track_model, num_classes=6, base_track_weight=0.40):
        super().__init__()
        
        self.rd_model = rd_model
        self.track_model = track_model
        self.num_classes = num_classes
        self.base_track_weight = base_track_weight
        
        # 冻结两个模型
        for p in self.rd_model.parameters():
            p.requires_grad = False
        for p in self.track_model.parameters():
            p.requires_grad = False
        
        # 动态权重网络：根据RD输出和航迹输出决定权重
        self.weight_net = nn.Sequential(
            nn.Linear(num_classes * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 初始化让输出接近base_track_weight
        # sigmoid(x) = base_track_weight -> x = log(w/(1-w))
        init_bias = math.log(base_track_weight / (1 - base_track_weight))
        with torch.no_grad():
            self.weight_net[-2].bias.fill_(init_bias)
    
    def forward(self, x_rd, x_track, x_stats, use_dynamic=True):
        """
        Args:
            use_dynamic: 是否使用动态权重，False则使用固定权重
        """
        self.rd_model.eval()
        self.track_model.eval()
        
        with torch.no_grad():
            rd_logits = self.rd_model(x_rd)
            track_logits = self.track_model(x_track, x_stats)
        
        rd_probs = torch.softmax(rd_logits, dim=1)
        track_probs = torch.softmax(track_logits, dim=1)
        
        if use_dynamic:
            # 动态权重
            combined = torch.cat([rd_logits, track_logits], dim=1)
            track_weight = self.weight_net(combined)  # [B, 1]
            
            # 限制范围 [0.1, 0.6]
            track_weight = 0.1 + 0.5 * track_weight
        else:
            # 固定权重
            track_weight = torch.full((x_rd.size(0), 1), self.base_track_weight, device=x_rd.device)
        
        rd_weight = 1.0 - track_weight
        
        fused_probs = rd_weight * rd_probs + track_weight * track_probs
        fused_logits = torch.log(fused_probs + 1e-8)
        
        return fused_logits, rd_logits, track_logits, track_weight.mean()


def train_track_model(device, train_loader, val_loader, epochs=30, conf_thresh=0.5, valid_classes=[0,1,2,3]):
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
        
        # 验证（只统计类别0-3）
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
                
                for i in range(len(y)):
                    if y[i].item() not in valid_classes:
                        continue
                    if conf[i] < conf_thresh:
                        continue
                    val_total += 1
                    if pred[i].item() == y[i].item():
                        val_correct += 1
        
        val_acc = 100 * val_correct / max(val_total, 1)
        
        mark = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()
            mark = "*"
        
        print(f"  Epoch {epoch:2d} | TrainAcc: {train_acc:.2f}% | ValAcc: {val_acc:.2f}% | Best: {best_acc:.2f}% {mark}")
    
    model.load_state_dict(best_state)
    print(f"\n航迹模型训练完成，最佳准确率: {best_acc:.2f}%")
    
    return model, best_acc


def test_fixed_weights(device, rd_model, track_model, val_loader, conf_thresh=0.5, valid_classes=[0,1,2,3]):
    """测试不同固定权重"""
    print("\n" + "="*60)
    print("第二阶段：测试固定权重融合（扩展范围）")
    print("="*60)
    
    rd_model.eval()
    track_model.eval()
    
    results = []
    
    # 扩展测试范围到0.50
    for track_weight in np.arange(0.05, 0.55, 0.05):
        rd_weight = 1.0 - track_weight
        
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for x_rd, x_track, x_stats, y in val_loader:
                x_rd = x_rd.to(device)
                x_track = x_track.to(device)
                x_stats = x_stats.to(device)
                y = y.to(device)
                
                rd_logits = rd_model(x_rd)
                track_logits = track_model(x_track, x_stats)
                
                rd_probs = torch.softmax(rd_logits, dim=1)
                track_probs = torch.softmax(track_logits, dim=1)
                fused_probs = rd_weight * rd_probs + track_weight * track_probs
                
                conf, pred = fused_probs.max(dim=1)
                
                for i in range(len(y)):
                    if y[i].item() not in valid_classes:
                        continue
                    if conf[i] < conf_thresh:
                        continue
                    val_total += 1
                    if pred[i].item() == y[i].item():
                        val_correct += 1
        
        acc = 100 * val_correct / max(val_total, 1)
        results.append((track_weight, acc))
        
        mark = ""
        if len(results) == 1 or acc > max(r[1] for r in results[:-1]):
            mark = " <-- Best"
        print(f"  track_weight={track_weight:.2f} | 融合准确率: {acc:.2f}%{mark}")
    
    best_weight, best_acc = max(results, key=lambda x: x[1])
    print(f"\n最佳固定权重: track={best_weight:.2f}, rd={1-best_weight:.2f}")
    print(f"最佳固定权重准确率: {best_acc:.2f}%")
    
    return best_weight, best_acc


def train_dynamic_fusion(device, rd_model, track_model, train_loader, val_loader, 
                        base_weight=0.40, epochs=20, conf_thresh=0.5, valid_classes=[0,1,2,3]):
    """训练动态权重融合"""
    print("\n" + "="*60)
    print("第三阶段：动态权重融合训练")
    print("="*60)
    
    model = DynamicFusionModel(rd_model, track_model, num_classes=6, base_track_weight=base_weight).to(device)
    
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数: {sum(p.numel() for p in trainable)}")
    
    optimizer = optim.Adam(trainable, lr=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 先测试固定权重的baseline
    model.eval()
    baseline_correct, baseline_total = 0, 0
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            y = y.to(device)
            
            fused, _, _, _ = model(x_rd, x_track, x_stats, use_dynamic=False)
            fused_probs = torch.softmax(fused, dim=1)
            conf, pred = fused_probs.max(dim=1)
            
            for i in range(len(y)):
                if y[i].item() not in valid_classes:
                    continue
                if conf[i] < conf_thresh:
                    continue
                baseline_total += 1
                if pred[i].item() == y[i].item():
                    baseline_correct += 1
    
    baseline_acc = 100 * baseline_correct / max(baseline_total, 1)
    print(f"固定权重(w={base_weight:.2f})基线: {baseline_acc:.2f}%")
    
    best_acc = baseline_acc
    best_state = None
    
    print(f"\n{'Ep':^4}|{'Loss':^8}|{'ValAcc':^8}|{'TrkW':^8}|{'Best':^8}")
    print("-" * 45)
    
    for epoch in range(epochs):
        model.train()
        run_loss = 0
        track_weights = []
        
        pbar = tqdm(train_loader, desc=f"Ep{epoch:02d}", ncols=50, leave=False)
        for x_rd, x_track, x_stats, y in pbar:
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            fused, _, _, tw = model(x_rd, x_track, x_stats, use_dynamic=True)
            loss = criterion(fused, y)
            loss.backward()
            optimizer.step()
            
            run_loss += loss.item()
            track_weights.append(tw.item())
        
        avg_loss = run_loss / len(train_loader)
        avg_tw = np.mean(track_weights)
        
        # 验证（只统计类别0-3）
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x_rd, x_track, x_stats, y in val_loader:
                x_rd = x_rd.to(device)
                x_track = x_track.to(device)
                x_stats = x_stats.to(device)
                y = y.to(device)
                
                fused, _, _, _ = model(x_rd, x_track, x_stats, use_dynamic=True)
                fused_probs = torch.softmax(fused, dim=1)
                conf, pred = fused_probs.max(dim=1)
                
                for i in range(len(y)):
                    if y[i].item() not in valid_classes:
                        continue
                    if conf[i] < conf_thresh:
                        continue
                    val_total += 1
                    if pred[i].item() == y[i].item():
                        val_correct += 1
        
        val_acc = 100 * val_correct / max(val_total, 1)
        
        mark = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()
            mark = "*"
        
        print(f"{epoch:^4}|{avg_loss:^8.4f}|{val_acc:^8.2f}|{avg_tw:^8.3f}|{best_acc:^8.2f} {mark}")
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_acc


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ==========================================
    # 路径配置
    # ==========================================
    RD_TRAIN = "./dataset/train_cleandata/train"
    RD_VAL = "./dataset/train_cleandata/val"
    TRACK_TRAIN = "./dataset/track_enhanced_cleandata/train"
    TRACK_VAL = "./dataset/track_enhanced_cleandata/val"
    PRETRAINED_PATH = "./checkpoint/ckpt_best_8_95.07.pth"
    SAVE_DIR = "./checkpoint/fusion_v13_final"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    if not os.path.exists(PRETRAINED_PATH):
        import glob
        pths = glob.glob("./checkpoint/*93*.pth") + glob.glob("./checkpoint/*94*.pth")
        if pths:
            PRETRAINED_PATH = pths[0]
    
    BATCH_SIZE = 16
    CONF_THRESH = 0.5
    VALID_CLASSES = [0, 1, 2, 3]  # 只统计这4类
    
    print(f"\n{'='*60}")
    print(f"双流融合训练 V13 (最终版)")
    print(f"{'='*60}")
    print(f"设备: {DEVICE}")
    print(f"只统计类别: {VALID_CLASSES}")
    
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
    
    # 验证RD基线（只统计类别0-3）
    rd_correct, rd_total = 0, 0
    with torch.no_grad():
        for x_rd, _, _, y in val_loader:
            x_rd, y = x_rd.to(DEVICE), y.to(DEVICE)
            out = rd_model(x_rd)
            out_softmax = torch.softmax(out, dim=1)
            conf, pred = out_softmax.max(dim=1)
            for i in range(len(y)):
                if y[i].item() not in VALID_CLASSES:
                    continue
                if conf[i] < CONF_THRESH:
                    continue
                rd_total += 1
                if pred[i].item() == y[i].item():
                    rd_correct += 1
    rd_baseline = 100 * rd_correct / max(rd_total, 1)
    print(f"RD基线准确率（类别0-3）: {rd_baseline:.2f}%")
    
    # Step 3: 训练航迹模型
    track_model, track_baseline = train_track_model(
        DEVICE, train_loader, val_loader, 
        epochs=30, conf_thresh=CONF_THRESH, valid_classes=VALID_CLASSES
    )
    
    # Step 4: 测试固定权重
    best_fixed_weight, best_fixed_acc = test_fixed_weights(
        DEVICE, rd_model, track_model, val_loader,
        conf_thresh=CONF_THRESH, valid_classes=VALID_CLASSES
    )
    
    # Step 5: 动态权重融合
    fusion_model, best_dynamic_acc = train_dynamic_fusion(
        DEVICE, rd_model, track_model, train_loader, val_loader,
        base_weight=best_fixed_weight, epochs=20, 
        conf_thresh=CONF_THRESH, valid_classes=VALID_CLASSES
    )
    
    # 保存最终模型
    final_acc = max(best_fixed_acc, best_dynamic_acc)
    save_path = f"{SAVE_DIR}/ckpt_best_{final_acc:.2f}.pth"
    torch.save({
        'fusion_model': fusion_model.state_dict(),
        'track_model': track_model.state_dict(),
        'rd_baseline': rd_baseline,
        'track_baseline': track_baseline,
        'best_fixed_weight': best_fixed_weight,
        'best_fixed_acc': best_fixed_acc,
        'best_dynamic_acc': best_dynamic_acc,
    }, save_path)
    
    # 最终总结
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"{'='*60}")
    print(f"  RD基线（类别0-3）:     {rd_baseline:.2f}%")
    print(f"  航迹基线（类别0-3）:   {track_baseline:.2f}%")
    print(f"  固定权重最佳:          {best_fixed_acc:.2f}% (track_weight={best_fixed_weight:.2f})")
    print(f"  动态权重最佳:          {best_dynamic_acc:.2f}%")
    print(f"  最终准确率:            {final_acc:.2f}%")
    print(f"  相比RD提升:            {final_acc - rd_baseline:+.2f}%")
    print(f"  模型保存: {save_path}")


if __name__ == '__main__':
    main()