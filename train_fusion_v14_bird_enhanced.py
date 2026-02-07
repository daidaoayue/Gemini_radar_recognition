"""
双流融合训练 V14 - 鸟类增强版
==============================
改进点：
1. Focal Loss - 关注困难样本
2. 类别权重 - 加大鸟类的惩罚
3. 更强的鸟类数据增强
4. 困难样本挖掘
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

from data_loader_fusion_v3 import FusionDataLoaderV3

try:
    from drsncww import rsnet34
except ImportError:
    print("错误: 找不到 drsncww.py")
    exit()


# ================= Focal Loss =================
class FocalLoss(nn.Module):
    """
    Focal Loss - 让模型更关注困难样本
    
    对于容易分类的样本（置信度高），降低其loss权重
    对于困难样本（置信度低），增加其loss权重
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 聚焦参数，越大越关注困难样本
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 预测正确类别的概率
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


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


def train_track_model_v2(device, train_loader, val_loader, epochs=30, 
                         conf_thresh=0.5, valid_classes=[0,1,2,3],
                         use_focal=True, bird_weight=1.5):
    """
    训练航迹模型 V2 - 增强版
    
    Args:
        use_focal: 是否使用Focal Loss
        bird_weight: 鸟类（类别2）的额外权重
    """
    print("\n" + "="*60)
    print("训练航迹模型（鸟类增强版）")
    print(f"  使用Focal Loss: {use_focal}")
    print(f"  鸟类权重: {bird_weight}x")
    print("="*60)
    
    model = TrackOnlyNetV3(num_classes=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 设置类别权重：鸟类权重更高
    # [轻型, 小型, 鸟类, 空飘球, 杂波, 其它]
    class_weights = torch.tensor([1.0, 1.0, bird_weight, 1.0, 0.5, 0.5]).to(device)
    
    if use_focal:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
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
        bird_correct, bird_total = 0, 0
        
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
                    
                    # 单独统计鸟类
                    if y[i].item() == 2:
                        bird_total += 1
                        if pred[i].item() == 2:
                            bird_correct += 1
        
        val_acc = 100 * val_correct / max(val_total, 1)
        bird_acc = 100 * bird_correct / max(bird_total, 1)
        
        mark = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()
            mark = "*"
        
        print(f"  Ep{epoch:2d} | Train:{train_acc:.1f}% | Val:{val_acc:.2f}% | Bird:{bird_acc:.1f}% | Best:{best_acc:.2f}% {mark}")
    
    model.load_state_dict(best_state)
    print(f"\n航迹模型训练完成，最佳准确率: {best_acc:.2f}%")
    
    return model, best_acc


def test_fusion(device, rd_model, track_model, val_loader, 
                track_weight=0.40, conf_thresh=0.5, valid_classes=[0,1,2,3]):
    """测试融合效果"""
    rd_model.eval()
    track_model.eval()
    
    rd_weight = 1.0 - track_weight
    
    results = {c: {'total': 0, 'correct': 0} for c in valid_classes}
    all_correct, all_total = 0, 0
    low_conf_count = 0
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            y = y.to(device)
            
            rd_probs = torch.softmax(rd_model(x_rd), dim=1)
            track_probs = torch.softmax(track_model(x_track, x_stats), dim=1)
            fused_probs = rd_weight * rd_probs + track_weight * track_probs
            
            conf, pred = fused_probs.max(dim=1)
            
            for i in range(len(y)):
                if y[i].item() not in valid_classes:
                    continue
                
                if conf[i] < conf_thresh:
                    low_conf_count += 1
                    continue
                
                c = y[i].item()
                results[c]['total'] += 1
                all_total += 1
                
                if pred[i].item() == c:
                    results[c]['correct'] += 1
                    all_correct += 1
    
    total_acc = 100 * all_correct / max(all_total, 1)
    coverage = 100 * all_total / (all_total + low_conf_count)
    
    return total_acc, coverage, results


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 路径配置
    RD_TRAIN = "./dataset/train_cleandata/train"
    RD_VAL = "./dataset/train_cleandata/val"
    TRACK_TRAIN = "./dataset/track_enhanced_cleandata/train"
    TRACK_VAL = "./dataset/track_enhanced_cleandata/val"
    PRETRAINED_PATH = "./checkpoint/ckpt_best_8_95.07.pth"
    SAVE_DIR = "./checkpoint/fusion_v14_bird_enhanced"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    import glob
    if not os.path.exists(PRETRAINED_PATH):
        pths = glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth")
        pths = [p for p in pths if 'fusion' not in p]
        if pths:
            PRETRAINED_PATH = pths[0]
    
    BATCH_SIZE = 16
    CONF_THRESH = 0.5
    VALID_CLASSES = [0, 1, 2, 3]
    
    print(f"\n{'='*60}")
    print(f"双流融合训练 V14 (鸟类增强版)")
    print(f"{'='*60}")
    print(f"设备: {DEVICE}")
    
    # 加载数据
    print(f"\n[Step 1] 加载数据...")
    train_ds = FusionDataLoaderV3(RD_TRAIN, TRACK_TRAIN, val=False)
    val_ds = FusionDataLoaderV3(RD_VAL, TRACK_VAL, val=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 加载RD模型
    print(f"\n[Step 2] 加载RD预训练模型...")
    rd_model = rsnet34()
    checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')
    state_dict = checkpoint['net_weight'] if 'net_weight' in checkpoint else checkpoint
    rd_model.load_state_dict(state_dict, strict=True)
    rd_model.to(DEVICE)
    rd_model.eval()
    print(f"已加载: {os.path.basename(PRETRAINED_PATH)}")
    
    # 测试不同配置
    print(f"\n[Step 3] 测试不同训练配置...")
    
    configs = [
        {'use_focal': False, 'bird_weight': 1.0, 'name': '基线(CE)'},
        {'use_focal': True, 'bird_weight': 1.0, 'name': 'Focal Loss'},
        {'use_focal': True, 'bird_weight': 1.5, 'name': 'Focal+鸟类1.5x'},
        {'use_focal': True, 'bird_weight': 2.0, 'name': 'Focal+鸟类2.0x'},
    ]
    
    best_config = None
    best_result = 0
    results_summary = []
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"测试配置: {config['name']}")
        print(f"{'='*60}")
        
        # 训练航迹模型
        track_model, track_acc = train_track_model_v2(
            DEVICE, train_loader, val_loader,
            epochs=20,  # 快速测试
            conf_thresh=CONF_THRESH,
            valid_classes=VALID_CLASSES,
            use_focal=config['use_focal'],
            bird_weight=config['bird_weight']
        )
        
        # 测试不同融合权重
        best_fusion_acc = 0
        best_weight = 0.40
        
        for tw in [0.30, 0.35, 0.40, 0.45]:
            acc, coverage, class_results = test_fusion(
                DEVICE, rd_model, track_model, val_loader,
                track_weight=tw, conf_thresh=CONF_THRESH, valid_classes=VALID_CLASSES
            )
            
            if acc > best_fusion_acc:
                best_fusion_acc = acc
                best_weight = tw
                best_class_results = class_results
                best_coverage = coverage
        
        # 计算鸟类准确率
        bird_acc = 100 * best_class_results[2]['correct'] / max(best_class_results[2]['total'], 1)
        
        result = {
            'config': config['name'],
            'track_acc': track_acc,
            'fusion_acc': best_fusion_acc,
            'bird_acc': bird_acc,
            'coverage': best_coverage,
            'best_weight': best_weight,
            'track_model': track_model.state_dict().copy()
        }
        results_summary.append(result)
        
        print(f"\n配置 [{config['name']}] 结果:")
        print(f"  航迹准确率: {track_acc:.2f}%")
        print(f"  融合准确率: {best_fusion_acc:.2f}% (weight={best_weight})")
        print(f"  鸟类准确率: {bird_acc:.1f}%")
        print(f"  覆盖率: {best_coverage:.1f}%")
        
        if best_fusion_acc > best_result:
            best_result = best_fusion_acc
            best_config = result
    
    # 总结
    print(f"\n{'='*60}")
    print(f"配置对比总结")
    print(f"{'='*60}")
    
    print(f"\n{'配置':^20}|{'融合准确率':^12}|{'鸟类准确率':^12}|{'覆盖率':^10}")
    print("-" * 60)
    
    for r in results_summary:
        mark = " *" if r['fusion_acc'] == best_result else ""
        print(f"{r['config']:^20}|{r['fusion_acc']:^12.2f}%|{r['bird_acc']:^12.1f}%|{r['coverage']:^10.1f}%{mark}")
    
    # 保存最佳模型
    if best_config:
        save_path = f"{SAVE_DIR}/ckpt_best_{best_config['fusion_acc']:.2f}.pth"
        torch.save({
            'track_model': best_config['track_model'],
            'best_fixed_weight': best_config['best_weight'],
            'track_weight': best_config['best_weight'],
            'fusion_acc': best_config['fusion_acc'],
            'bird_acc': best_config['bird_acc'],
            'coverage': best_config['coverage'],
            'config': best_config['config']
        }, save_path)
        
        print(f"\n最佳配置: {best_config['config']}")
        print(f"保存到: {save_path}")
    
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()