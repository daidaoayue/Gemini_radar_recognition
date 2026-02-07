"""
置信度阈值分析
==============
分析不同置信度阈值下的准确率和覆盖率
找到最佳平衡点
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

warnings.filterwarnings("ignore")

try:
    from data_loader_fusion_v4 import FusionDataLoaderV4 as FusionDataLoader
except ImportError:
    from data_loader_fusion_v3 import FusionDataLoaderV3 as FusionDataLoader

try:
    from drsncww import rsnet34
except ImportError:
    exit()


# 网络定义（简化版）
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
        return x * self.fc(y).unsqueeze(2)


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
        t = F.dropout(self.ms_se_conv1(x_temporal), 0.2, self.training)
        t = F.dropout(self.ms_se_conv2(t), 0.2, self.training)
        t = self.temporal_pool(self.ms_se_conv3(t))
        s = F.relu(self.stats_bn1(self.stats_fc1(x_stats)))
        s = self.stats_fc2(s * self.stats_se(s))
        return self.classifier(torch.cat([t, s], dim=1))


def analyze_thresholds():
    # 配置
    RD_VAL = "./dataset/train_cleandata/val"
    TRACK_VAL = "./dataset/track_enhanced_v5_cleandata/val"
    TRACK_WEIGHT = 0.5
    VALID_CLASSES = [0, 1, 2, 3]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("置信度阈值分析")
    print("="*70)
    
    # 加载模型
    print("\n加载模型...")
    
    rd_model = rsnet34()
    rd_ckpts = glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth")
    rd_ckpts = [p for p in rd_ckpts if 'fusion' not in p]
    if rd_ckpts:
        rd_model.load_state_dict(torch.load(rd_ckpts[0], map_location='cpu').get('net_weight', {}))
    rd_model.to(device).eval()
    
    model_dir = "./checkpoint/fusion_v17_multiscale_se"
    model_files = glob.glob(os.path.join(model_dir, "ckpt_best_*.pth"))
    model_path = sorted(model_files, key=os.path.getmtime)[-1]
    
    track_model = TrackNetV17_MultiScaleSE(num_classes=6, stats_dim=46)
    track_model.load_state_dict(torch.load(model_path, map_location='cpu')['track_model'])
    track_model.to(device).eval()
    
    # 数据
    val_ds = FusionDataLoader(RD_VAL, TRACK_VAL, val=True, stats_dim=46)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # 收集所有预测结果
    all_labels = []
    all_preds = []
    all_confs = []
    
    rd_w = 1.0 - TRACK_WEIGHT
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd, x_track, x_stats = x_rd.to(device), x_track.to(device), x_stats.to(device)
            
            rd_probs = torch.softmax(rd_model(x_rd), dim=1)
            track_probs = torch.softmax(track_model(x_track, x_stats), dim=1)
            fused_probs = rd_w * rd_probs + TRACK_WEIGHT * track_probs
            
            conf, pred = fused_probs.max(dim=1)
            
            for i in range(len(y)):
                if y[i].item() in VALID_CLASSES:
                    all_labels.append(y[i].item())
                    all_preds.append(pred[i].item())
                    all_confs.append(conf[i].item())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_confs = np.array(all_confs)
    
    total = len(all_labels)
    
    # 分析不同阈值
    print(f"\n总样本数: {total}")
    print(f"\n{'阈值':^8}|{'覆盖率':^10}|{'准确率':^10}|{'高置信数':^10}|{'低置信数':^10}|{'低置信准确率':^12}")
    print("-"*70)
    
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    
    best_score = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        mask_high = all_confs >= thresh
        mask_low = ~mask_high
        
        n_high = mask_high.sum()
        n_low = mask_low.sum()
        
        coverage = 100 * n_high / total
        
        if n_high > 0:
            acc_high = 100 * (all_preds[mask_high] == all_labels[mask_high]).sum() / n_high
        else:
            acc_high = 0
        
        if n_low > 0:
            acc_low = 100 * (all_preds[mask_low] == all_labels[mask_low]).sum() / n_low
        else:
            acc_low = 0
        
        # 综合得分 = 准确率 * 覆盖率 / 100
        score = acc_high * coverage / 100
        
        mark = ""
        if score > best_score:
            best_score = score
            best_thresh = thresh
            mark = " ★"
        
        print(f"{thresh:^8.2f}|{coverage:^10.1f}%|{acc_high:^10.2f}%|{n_high:^10}|{n_low:^10}|{acc_low:^12.1f}%{mark}")
    
    print("-"*70)
    print(f"\n最佳阈值: {best_thresh} (综合得分: {best_score:.2f})")
    
    # 详细分析最佳阈值附近
    print(f"\n{'='*70}")
    print(f"阈值 {best_thresh} 详细分析")
    print(f"{'='*70}")
    
    mask_high = all_confs >= best_thresh
    
    for c in VALID_CLASSES:
        mask_c = mask_high & (all_labels == c)
        n_c = mask_c.sum()
        correct_c = (all_preds[mask_c] == c).sum()
        acc_c = 100 * correct_c / max(n_c, 1)
        
        mask_c_all = all_labels == c
        cov_c = 100 * n_c / mask_c_all.sum()
        
        class_names = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
        print(f"  {class_names[c]}: 准确率 {acc_c:.2f}%, 覆盖率 {cov_c:.1f}%")
    
    # 置信度分布
    print(f"\n置信度分布:")
    print(f"  最小: {all_confs.min():.3f}")
    print(f"  最大: {all_confs.max():.3f}")
    print(f"  均值: {all_confs.mean():.3f}")
    print(f"  中位数: {np.median(all_confs):.3f}")
    
    # 置信度区间统计
    print(f"\n置信度区间统计:")
    bins = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    for low, high in bins:
        mask = (all_confs >= low) & (all_confs < high)
        n = mask.sum()
        if n > 0:
            acc = 100 * (all_preds[mask] == all_labels[mask]).sum() / n
            print(f"  [{low:.1f}, {high:.1f}): {n:4} 个, 准确率 {acc:.1f}%")


if __name__ == '__main__':
    analyze_thresholds()