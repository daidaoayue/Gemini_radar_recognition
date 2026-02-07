"""
低置信度样本分析脚本
====================
分析被过滤掉的147个样本：
1. 它们是哪些类别？
2. 它们的置信度分布如何？
3. 它们预测对了还是错了？
4. 如果不过滤，准确率会是多少？

这些信息帮助我们决定：
- 是否需要改进数据处理
- 是否需要调整训练策略
- 最佳置信度阈值是多少
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from data_loader_fusion_v3 import FusionDataLoaderV3

try:
    from drsncww import rsnet34
except ImportError:
    print("错误: 找不到 drsncww.py")
    exit()


class TrackOnlyNetV3(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.temporal_net = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten()
        )
        self.stats_net = nn.Sequential(
            nn.Linear(20, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 6)
        )
    
    def forward(self, x_temporal, x_stats):
        feat_temporal = self.temporal_net(x_temporal)
        feat_stats = self.stats_net(x_stats)
        return self.classifier(torch.cat([feat_temporal, feat_stats], dim=1))


def main():
    # 配置
    RD_VAL = "./dataset/train_cleandata/val"
    TRACK_VAL = "./dataset/track_enhanced_cleandata/val"
    CHECKPOINT = "./checkpoint/fusion_v13_final/ckpt_best_98.43.pth"
    
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球', 4: '杂波', 5: '其它'}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("低置信度样本分析")
    print("="*60)
    
    # 加载数据
    print("\n加载数据...")
    val_ds = FusionDataLoaderV3(RD_VAL, TRACK_VAL, val=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # 加载模型
    print("加载模型...")
    import glob
    if not os.path.exists(CHECKPOINT):
        pths = glob.glob("./checkpoint/fusion_v13*/ckpt_best*.pth")
        if pths:
            CHECKPOINT = sorted(pths)[-1]
    
    ckpt = torch.load(CHECKPOINT, map_location='cpu')
    track_weight = ckpt.get('best_fixed_weight', 0.40)
    rd_weight = 1.0 - track_weight
    
    rd_model = rsnet34()
    rd_pths = glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth")
    rd_pths = [p for p in rd_pths if 'fusion' not in p]
    if rd_pths:
        rd_ckpt = torch.load(rd_pths[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
    rd_model.to(device).eval()
    
    track_model = TrackOnlyNetV3()
    track_model.load_state_dict(ckpt['track_model'])
    track_model.to(device).eval()
    
    print(f"融合权重: RD={rd_weight:.2f}, Track={track_weight:.2f}")
    
    # 收集所有预测结果
    print("\n预测所有样本...")
    all_results = []
    
    sample_idx = 0
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            
            rd_probs = torch.softmax(rd_model(x_rd), dim=1)
            track_probs = torch.softmax(track_model(x_track, x_stats), dim=1)
            fused_probs = rd_weight * rd_probs + track_weight * track_probs
            
            confs, preds = fused_probs.max(dim=1)
            
            for i in range(len(y)):
                true_label = y[i].item()
                
                if true_label not in VALID_CLASSES:
                    sample_idx += 1
                    continue
                
                # 获取文件名
                if sample_idx < len(val_ds.samples):
                    _, rd_path, _ = val_ds.samples[sample_idx]
                    filename = os.path.basename(rd_path)
                else:
                    filename = f"sample_{sample_idx}"
                
                all_results.append({
                    'idx': sample_idx,
                    'filename': filename,
                    'true_label': true_label,
                    'pred_label': preds[i].item(),
                    'confidence': confs[i].item(),
                    'correct': preds[i].item() == true_label,
                    'probs': fused_probs[i].cpu().numpy()
                })
                
                sample_idx += 1
    
    print(f"类别0-3总样本数: {len(all_results)}")
    
    # 分析不同置信度阈值
    print("\n" + "="*60)
    print("不同置信度阈值的准确率和覆盖率")
    print("="*60)
    
    print(f"\n{'阈值':^8}|{'覆盖样本':^10}|{'覆盖率':^10}|{'正确数':^10}|{'准确率':^10}")
    print("-" * 55)
    
    best_f1_thresh = 0
    best_f1 = 0
    
    for thresh in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        covered = [r for r in all_results if r['confidence'] >= thresh]
        n_covered = len(covered)
        coverage = 100 * n_covered / len(all_results)
        n_correct = sum(1 for r in covered if r['correct'])
        acc = 100 * n_correct / n_covered if n_covered > 0 else 0
        
        # 计算F1-like分数（准确率和覆盖率的调和平均）
        if acc > 0 and coverage > 0:
            f1 = 2 * (acc/100) * (coverage/100) / ((acc/100) + (coverage/100))
            if f1 > best_f1:
                best_f1 = f1
                best_f1_thresh = thresh
        
        mark = " *" if thresh == 0.5 else ""
        print(f"{thresh:^8.1f}|{n_covered:^10}|{coverage:^10.1f}%|{n_correct:^10}|{acc:^10.2f}%{mark}")
    
    print(f"\n最佳F1阈值: {best_f1_thresh} (F1={best_f1:.3f})")
    
    # 分析低置信度样本
    print("\n" + "="*60)
    print("低置信度样本分析 (conf < 0.5)")
    print("="*60)
    
    low_conf = [r for r in all_results if r['confidence'] < 0.5]
    print(f"\n低置信度样本数: {len(low_conf)} ({100*len(low_conf)/len(all_results):.1f}%)")
    
    # 按类别统计
    print(f"\n按真实类别分布:")
    print(f"{'类别':^12}|{'样本数':^8}|{'预测正确':^10}|{'准确率':^10}")
    print("-" * 45)
    
    for c in VALID_CLASSES:
        c_samples = [r for r in low_conf if r['true_label'] == c]
        c_correct = sum(1 for r in c_samples if r['correct'])
        c_acc = 100 * c_correct / len(c_samples) if c_samples else 0
        print(f"{CLASS_NAMES[c]:^12}|{len(c_samples):^8}|{c_correct:^10}|{c_acc:^10.1f}%")
    
    # 低置信度样本的预测正确率
    low_conf_correct = sum(1 for r in low_conf if r['correct'])
    low_conf_acc = 100 * low_conf_correct / len(low_conf) if low_conf else 0
    print(f"\n低置信度样本整体准确率: {low_conf_acc:.1f}%")
    
    # 分析置信度分布
    print("\n" + "="*60)
    print("置信度分布")
    print("="*60)
    
    conf_ranges = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
    
    print(f"\n{'置信度范围':^15}|{'样本数':^8}|{'正确数':^8}|{'准确率':^10}")
    print("-" * 50)
    
    for low, high in conf_ranges:
        in_range = [r for r in all_results if low <= r['confidence'] < high]
        n_in = len(in_range)
        n_correct = sum(1 for r in in_range if r['correct'])
        acc = 100 * n_correct / n_in if n_in > 0 else 0
        print(f"[{low:.1f}, {high:.1f})     |{n_in:^8}|{n_correct:^8}|{acc:^10.1f}%")
    
    # 分析低置信度样本的具体错误
    print("\n" + "="*60)
    print("低置信度且预测错误的样本 (最危险)")
    print("="*60)
    
    low_conf_wrong = [r for r in low_conf if not r['correct']]
    print(f"\n低置信度且错误: {len(low_conf_wrong)}个")
    
    if len(low_conf_wrong) > 0:
        print(f"\n{'序号':^6}|{'真实':^10}|{'预测':^10}|{'置信度':^8}|{'第二可能':^10}")
        print("-" * 55)
        
        for r in low_conf_wrong[:20]:
            true_name = CLASS_NAMES.get(r['true_label'], str(r['true_label']))[:8]
            pred_name = CLASS_NAMES.get(r['pred_label'], str(r['pred_label']))[:8]
            
            # 找第二高的概率
            probs = r['probs'][:4]  # 只看前4类
            sorted_idx = np.argsort(probs)[::-1]
            second_class = sorted_idx[1]
            second_prob = probs[second_class]
            second_name = CLASS_NAMES.get(second_class, str(second_class))[:8]
            
            print(f"{r['idx']:^6}|{true_name:^10}|{pred_name:^10}|{r['confidence']:^8.3f}|{second_name}({second_prob:.2f})")
    
    # 建议
    print("\n" + "="*60)
    print("分析结论与建议")
    print("="*60)
    
    # 计算关键指标
    high_conf_acc = 100 * sum(1 for r in all_results if r['confidence'] >= 0.5 and r['correct']) / \
                    sum(1 for r in all_results if r['confidence'] >= 0.5)
    
    print(f"""
当前状态:
  - 高置信度(≥0.5)准确率: {high_conf_acc:.2f}%
  - 低置信度(<0.5)准确率: {low_conf_acc:.1f}%
  - 低置信度样本占比: {100*len(low_conf)/len(all_results):.1f}%

如果比赛要求100%覆盖率:
  - 全部样本准确率: {100*sum(1 for r in all_results if r['correct'])/len(all_results):.2f}%
  - 需要提升低置信度样本的准确率

建议方向:
""")
    
    # 判断低置信度样本是哪类问题
    bird_low_conf = len([r for r in low_conf if r['true_label'] == 2])
    bird_ratio = bird_low_conf / len(low_conf) if low_conf else 0
    
    if bird_ratio > 0.4:
        print(f"  1. [重点] 鸟类样本低置信度占{100*bird_ratio:.0f}%，需要改进鸟类特征提取")
    
    if low_conf_acc < 70:
        print(f"  2. [重点] 低置信度样本准确率仅{low_conf_acc:.0f}%，考虑:")
        print(f"     - 增加训练数据增强")
        print(f"     - 使用温度缩放(Temperature Scaling)调整置信度")
        print(f"     - 针对困难样本重新训练")
    
    if len(low_conf) > 100:
        print(f"  3. [可选] 考虑降低置信度阈值到0.4，牺牲少量准确率换取覆盖率")


if __name__ == '__main__':
    main()