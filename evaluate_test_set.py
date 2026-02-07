"""
测试集评估脚本 - V17 完整系统
==============================
在测试集上评估主模型+级联分类器的完整系统性能

评估内容:
1. 主模型单独性能
2. 级联策略性能
3. 各类别详细指标
4. 混淆矩阵
5. 误分类分析
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
from datetime import datetime

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


# ==================== 网络定义 ====================
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
        """获取融合特征"""
        t = self.ms_se_conv1(x_temporal)
        t = self.ms_se_conv2(t)
        t = self.ms_se_conv3(t)
        t = self.temporal_pool(t)
        
        s = F.relu(self.stats_bn1(self.stats_fc1(x_stats)))
        s = s * self.stats_se(s)
        s = self.stats_fc2(s)
        
        return torch.cat([t, s], dim=1)


class CascadeClassifier(nn.Module):
    def __init__(self, input_dim=460, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def evaluate_test_set():
    """在测试集上评估完整系统"""
    
    # ==================== 配置 ====================
    RD_TEST = "./dataset/train_cleandata/test"
    
    if os.path.exists("./dataset/track_enhanced_v5_cleandata/test"):
        TRACK_TEST = "./dataset/track_enhanced_v5_cleandata/test"
    elif os.path.exists("./dataset/track_enhanced_v4_cleandata/test"):
        TRACK_TEST = "./dataset/track_enhanced_v4_cleandata/test"
    else:
        TRACK_TEST = "./dataset/track_enhanced_cleandata/test"
    
    CONF_THRESH = 0.5
    TRACK_WEIGHT = 0.5
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球', 4: '杂波', 5: '其他'}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("V17 完整系统 - 测试集评估")
    print("="*70)
    print(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    print(f"置信度阈值: {CONF_THRESH}")
    print(f"Track权重: {TRACK_WEIGHT}")
    
    # ==================== 检查数据目录 ====================
    if not os.path.exists(RD_TEST):
        print(f"\n错误: 测试集RD目录不存在: {RD_TEST}")
        print("请确认测试集路径是否正确")
        return
    
    if not os.path.exists(TRACK_TEST):
        print(f"\n错误: 测试集Track目录不存在: {TRACK_TEST}")
        return
    
    # ==================== 加载模型 ====================
    print("\n" + "-"*70)
    print("加载模型")
    print("-"*70)
    
    # 1. RD模型
    print("\n1. RD模型...")
    rd_model = rsnet34()
    rd_ckpts = glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth")
    rd_ckpts = [p for p in rd_ckpts if 'fusion' not in p]
    
    if rd_ckpts:
        rd_ckpt = torch.load(rd_ckpts[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
        print(f"   已加载: {os.path.basename(rd_ckpts[0])}")
    else:
        print("   警告: 未找到RD模型权重!")
    
    rd_model.to(device)
    rd_model.eval()
    
    # 2. Track模型
    print("\n2. Track模型 (V17 MultiScale+SE)...")
    track_model = TrackNetV17_MultiScaleSE(num_classes=6, stats_dim=46)
    
    model_dir = "./checkpoint/fusion_v17_multiscale_se"
    model_files = glob.glob(os.path.join(model_dir, "ckpt_best_*.pth"))
    
    if model_files:
        model_path = sorted(model_files, key=os.path.getmtime)[-1]
        checkpoint = torch.load(model_path, map_location='cpu')
        track_model.load_state_dict(checkpoint['track_model'])
        print(f"   已加载: {os.path.basename(model_path)}")
        print(f"   训练准确率: {checkpoint.get('best_acc', 'N/A'):.2f}%")
    else:
        print("   错误: 未找到Track模型!")
        return
    
    track_model.to(device)
    track_model.eval()
    
    # 3. 级联分类器
    print("\n3. 级联分类器...")
    cascade_model = CascadeClassifier(input_dim=460, num_classes=4)
    
    cascade_dir = "./checkpoint/fusion_v17_cascade"
    cascade_files = glob.glob(os.path.join(cascade_dir, "cascade_*.pth"))
    
    has_cascade = False
    if cascade_files:
        cascade_path = sorted(cascade_files)[-1]
        cascade_ckpt = torch.load(cascade_path, map_location='cpu')
        cascade_model.load_state_dict(cascade_ckpt['cascade_model'])
        cascade_model.to(device)
        cascade_model.eval()
        has_cascade = True
        print(f"   已加载: {os.path.basename(cascade_path)}")
    else:
        print("   警告: 未找到级联分类器，将只使用主模型")
    
    # ==================== 加载测试数据 ====================
    print("\n" + "-"*70)
    print("加载测试数据")
    print("-"*70)
    
    test_ds = FusionDataLoader(RD_TEST, TRACK_TEST, val=True, stats_dim=46)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # ==================== 推理 ====================
    print("\n开始推理...")
    
    all_labels = []
    all_preds_main = []      # 主模型预测
    all_preds_cascade = []   # 级联策略预测
    all_confs = []
    all_rd_preds = []
    all_track_preds = []
    
    rd_w = 1.0 - TRACK_WEIGHT
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in test_loader:
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            
            # 主模型预测
            rd_probs = torch.softmax(rd_model(x_rd), dim=1)
            track_probs = torch.softmax(track_model(x_track, x_stats), dim=1)
            fused_probs = rd_w * rd_probs + TRACK_WEIGHT * track_probs
            
            conf, main_pred = fused_probs.max(dim=1)
            _, rd_pred = rd_probs.max(dim=1)
            _, track_pred = track_probs.max(dim=1)
            
            # 级联策略预测
            if has_cascade:
                track_features = track_model.get_features(x_track, x_stats)
                cascade_input = torch.cat([track_features, rd_probs, track_probs], dim=1)
                cascade_logits = cascade_model(cascade_input)
                cascade_pred = cascade_logits.argmax(dim=1)
                
                # 根据置信度选择
                final_pred = torch.where(conf >= CONF_THRESH, main_pred, cascade_pred)
            else:
                final_pred = main_pred
            
            for i in range(len(y)):
                all_labels.append(y[i].item())
                all_preds_main.append(main_pred[i].item())
                all_preds_cascade.append(final_pred[i].item())
                all_confs.append(conf[i].item())
                all_rd_preds.append(rd_pred[i].item())
                all_track_preds.append(track_pred[i].item())
    
    # 转换为numpy
    all_labels = np.array(all_labels)
    all_preds_main = np.array(all_preds_main)
    all_preds_cascade = np.array(all_preds_cascade)
    all_confs = np.array(all_confs)
    all_rd_preds = np.array(all_rd_preds)
    all_track_preds = np.array(all_track_preds)
    
    # 有效样本mask
    mask_valid = np.isin(all_labels, VALID_CLASSES)
    mask_high = all_confs >= CONF_THRESH
    mask_low = ~mask_high
    
    total_valid = mask_valid.sum()
    total_high = (mask_valid & mask_high).sum()
    total_low = (mask_valid & mask_low).sum()
    
    # ==================== 结果分析 ====================
    print("\n" + "="*70)
    print("测试集评估结果")
    print("="*70)
    
    # 1. 总体统计
    print("\n【1. 总体统计】")
    print(f"   测试样本总数: {len(all_labels)}")
    print(f"   有效样本数 (类别0-3): {total_valid}")
    print(f"   高置信度样本: {total_high} ({100*total_high/total_valid:.1f}%)")
    print(f"   低置信度样本: {total_low} ({100*total_low/total_valid:.1f}%)")
    
    # 2. 主模型性能
    print("\n【2. 主模型性能】")
    
    # 全部样本
    correct_main_all = (all_preds_main[mask_valid] == all_labels[mask_valid]).sum()
    acc_main_all = 100 * correct_main_all / total_valid
    print(f"   全部样本准确率: {acc_main_all:.2f}%")
    
    # 高置信度
    mask_hv = mask_valid & mask_high
    correct_main_high = (all_preds_main[mask_hv] == all_labels[mask_hv]).sum()
    acc_main_high = 100 * correct_main_high / total_high if total_high > 0 else 0
    print(f"   高置信度准确率: {acc_main_high:.2f}%")
    
    # 低置信度
    mask_lv = mask_valid & mask_low
    if total_low > 0:
        correct_main_low = (all_preds_main[mask_lv] == all_labels[mask_lv]).sum()
        acc_main_low = 100 * correct_main_low / total_low
        print(f"   低置信度准确率: {acc_main_low:.2f}%")
    
    # 3. 级联策略性能
    if has_cascade:
        print("\n【3. 级联策略性能】")
        
        correct_cascade_all = (all_preds_cascade[mask_valid] == all_labels[mask_valid]).sum()
        acc_cascade_all = 100 * correct_cascade_all / total_valid
        print(f"   全部样本准确率: {acc_cascade_all:.2f}%")
        
        if total_low > 0:
            correct_cascade_low = (all_preds_cascade[mask_lv] == all_labels[mask_lv]).sum()
            acc_cascade_low = 100 * correct_cascade_low / total_low
            print(f"   低置信度准确率: {acc_cascade_low:.2f}%")
            print(f"   低置信度提升: {acc_cascade_low - acc_main_low:+.2f}%")
        
        print(f"\n   整体提升: {acc_cascade_all - acc_main_all:+.2f}%")
    
    # 4. 各类别性能（使用级联策略结果）
    print("\n【4. 各类别性能（级联策略）】")
    print(f"   {'类别':<12} {'样本数':>8} {'正确':>8} {'准确率':>10} {'召回率':>10}")
    print("   " + "-"*55)
    
    final_preds = all_preds_cascade if has_cascade else all_preds_main
    
    for c in VALID_CLASSES:
        mask_c = mask_valid & (all_labels == c)
        total_c = mask_c.sum()
        correct_c = (final_preds[mask_c] == c).sum()
        recall = 100 * correct_c / total_c if total_c > 0 else 0
        
        # 精确率
        mask_pred_c = mask_valid & (final_preds == c)
        pred_c = mask_pred_c.sum()
        true_pos = (all_labels[mask_pred_c] == c).sum()
        precision = 100 * true_pos / pred_c if pred_c > 0 else 0
        
        print(f"   {CLASS_NAMES[c]:<12} {total_c:>8} {correct_c:>8} {precision:>9.2f}% {recall:>9.2f}%")
    
    # 5. 混淆矩阵
    print("\n【5. 混淆矩阵（级联策略）】")
    print(f"\n   预测→")
    print(f"   真实↓    ", end="")
    for c in VALID_CLASSES:
        print(f"{CLASS_NAMES[c][:4]:>8}", end="")
    print(f"{'总计':>8}")
    print("   " + "-"*50)
    
    conf_matrix = np.zeros((4, 4), dtype=int)
    for i, c_true in enumerate(VALID_CLASSES):
        for j, c_pred in enumerate(VALID_CLASSES):
            mask = mask_valid & (all_labels == c_true) & (final_preds == c_pred)
            conf_matrix[i, j] = mask.sum()
    
    for i, c_true in enumerate(VALID_CLASSES):
        print(f"   {CLASS_NAMES[c_true][:4]:<8}", end="")
        for j in range(4):
            val = conf_matrix[i, j]
            if i == j:
                print(f"[{val:>5}]", end=" ")
            elif val > 0:
                print(f" {val:>5} ", end=" ")
            else:
                print(f" {'·':>5} ", end=" ")
        print(f"{conf_matrix[i].sum():>8}")
    
    # 6. 误分类分析
    print("\n【6. 误分类分析】")
    
    misclassified = mask_valid & (final_preds != all_labels)
    n_mis = misclassified.sum()
    
    print(f"   误分类总数: {n_mis} ({100*n_mis/total_valid:.2f}%)")
    
    if n_mis > 0:
        print(f"\n   {'真实类别':<12} → {'预测类别':<12} | {'数量':>6}")
        print("   " + "-"*45)
        
        from collections import defaultdict
        mis_pairs = defaultdict(int)
        for i in np.where(misclassified)[0]:
            true_c = all_labels[i]
            pred_c = final_preds[i]
            mis_pairs[(true_c, pred_c)] += 1
        
        for (true_c, pred_c), count in sorted(mis_pairs.items(), key=lambda x: -x[1]):
            print(f"   {CLASS_NAMES[true_c]:<12} → {CLASS_NAMES[pred_c]:<12} | {count:>6}")
    
    # 7. RD vs Track对比
    print("\n【7. 模型对比】")
    
    rd_correct = (all_rd_preds[mask_valid] == all_labels[mask_valid]).sum()
    track_correct = (all_track_preds[mask_valid] == all_labels[mask_valid]).sum()
    
    print(f"   RD模型准确率:     {100*rd_correct/total_valid:.2f}%")
    print(f"   Track模型准确率:  {100*track_correct/total_valid:.2f}%")
    print(f"   主模型(融合)准确率: {acc_main_all:.2f}%")
    if has_cascade:
        print(f"   级联策略准确率:   {acc_cascade_all:.2f}%")
    
    # 8. 总结
    print("\n" + "="*70)
    print("测试集评估总结")
    print("="*70)
    
    final_acc = acc_cascade_all if has_cascade else acc_main_all
    
    print(f"""
┌─────────────────────────────────────────────────────────┐
│                   测试集最终结果                          │
├─────────────────────────────────────────────────────────┤
│  整体准确率:         {final_acc:>6.2f}%                         │
│  高置信度准确率:     {acc_main_high:>6.2f}% (覆盖{100*total_high/total_valid:.1f}%)              │
│  低置信度准确率:     {acc_cascade_low if has_cascade and total_low > 0 else acc_main_low if total_low > 0 else 0:>6.2f}% (级联分类器)            │
│  误分类数:           {n_mis:>6} / {total_valid}                      │
└─────────────────────────────────────────────────────────┘
    """)
    
    # 保存结果到文件
    result_file = f"./test_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"V17完整系统测试集评估结果\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"="*50 + "\n")
        f.write(f"测试样本数: {total_valid}\n")
        f.write(f"整体准确率: {final_acc:.2f}%\n")
        f.write(f"高置信度准确率: {acc_main_high:.2f}%\n")
        f.write(f"高置信度覆盖率: {100*total_high/total_valid:.1f}%\n")
        f.write(f"误分类数: {n_mis}\n")
    
    print(f"\n结果已保存到: {result_file}")


if __name__ == '__main__':
    evaluate_test_set()