"""
航迹分支单独诊断脚本
目的：测试航迹特征本身的分类能力
如果航迹分支单独只能达到 60-70%，那融合时它会拖累 RD 分支
"""

import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

from data_loader_fusion import FusionDataLoader

print("✅ 模块加载完成")


class TrackOnlyNet(nn.Module):
    """纯航迹分类网络 - 诊断航迹特征的区分能力"""
    def __init__(self, num_classes=6):
        super(TrackOnlyNet, self).__init__()
        
        # 航迹分支 (与融合模型相同)
        self.track_net = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=5, padding=2),
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
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x_track):
        feat = self.track_net(x_track)
        return self.classifier(feat)


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # --- 配置 ---
    RD_TRAIN_DIR = "./dataset/train/2026-1-14/train"
    RD_VAL_DIR = "./dataset/train/2026-1-14/val"
    TRACK_DIR = "./dataset/track_test/"
    # TRACK_DIR =  "../Preprocess/KDE测量数据集/"

    

    BATCH_SIZE = 32
    EPOCHS = 50
    
    # --- 数据加载 ---
    print("\n📂 加载数据...")
    train_ds = FusionDataLoader(RD_TRAIN_DIR, TRACK_DIR, val=False)
    val_ds = FusionDataLoader(RD_VAL_DIR, TRACK_DIR, val=True)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=0, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    # --- 模型 ---
    print("\n🔧 初始化纯航迹模型...")
    model = TrackOnlyNet(num_classes=6).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0.0
    
    # --- 训练循环 ---
    print(f"\n{'='*60}")
    print(f"🔬 航迹分支诊断 | 设备: {DEVICE}")
    print(f"{'='*60}")
    print(f"{'Epoch':^6}|{'TrainLoss':^10}|{'TrainAcc':^10}|{'ValAcc':^10}|{'Best':^10}")
    print("-" * 50)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Ep{epoch:02d}", ncols=50, leave=False)
        
        for x_rd, x_track, labels in pbar:
            # 只用航迹数据，忽略 RD
            x_track = x_track.float().to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(x_track)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        avg_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # --- 验证 ---
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x_rd, x_track, labels in val_loader:
                x_track = x_track.float().to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(x_track)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        mark = ""
        if val_acc > best_acc:
            best_acc = val_acc
            mark = "⭐"
        
        print(f"{epoch:^6}|{avg_loss:^10.4f}|{train_acc:^9.2f}%|{val_acc:^9.2f}%|{best_acc:^9.2f}% {mark}")
    
    print(f"\n{'='*60}")
    print(f"📊 诊断结果：航迹分支最佳验证准确率 = {best_acc:.2f}%")
    print(f"{'='*60}")
    
    if best_acc < 50:
        print("⚠️ 航迹特征区分度极低，建议：")
        print("   1. 检查航迹数据是否正确加载")
        print("   2. 检查航迹与RD样本的对齐是否正确")
        print("   3. 可能需要更换航迹特征提取方式")
    elif best_acc < 70:
        print("⚠️ 航迹特征区分度一般，建议：")
        print("   1. 在融合时大幅降低航迹分支权重")
        print("   2. 或完全冻结RD分支，只让航迹学习辅助信息")
    elif best_acc < 85:
        print("✓ 航迹特征有一定区分度，可以尝试融合")
        print("   但需要谨慎设计融合策略，避免拖累RD分支")
    else:
        print("✅ 航迹特征区分度良好，融合应该能提升性能")


if __name__ == '__main__':
    main()
