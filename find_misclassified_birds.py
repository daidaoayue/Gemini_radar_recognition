"""
找出所有误分类的鸟类样本
==========================
输出具体文件路径，方便人工检查
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import warnings
import glob
from collections import defaultdict

warnings.filterwarnings("ignore")

from data_loader_fusion_v3 import FusionDataLoaderV3
from drsncww import rsnet34


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
        return self.classifier(torch.cat([self.temporal_net(x_temporal), self.stats_net(x_stats)], dim=1))


def calibrate_bn(model, data_loader, device, num_batches=50):
    model.train()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            _ = model(batch[1].to(device), batch[2].to(device))
            if i >= num_batches - 1:
                break
    model.eval()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球', 4: '类别4', 5: '类别5'}
    BIRD_CLASS = 2
    
    print("="*70)
    print("找出所有误分类的鸟类样本")
    print("="*70)
    
    # 加载模型
    v14_pths = glob.glob("./checkpoint/fusion_v14*/ckpt_best*.pth")
    ckpt = torch.load(sorted(v14_pths)[-1], map_location='cpu')
    
    rd_model = rsnet34()
    rd_pths = [p for p in glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth") if 'fusion' not in p]
    if rd_pths:
        rd_ckpt = torch.load(rd_pths[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
    rd_model.to(device).eval()
    
    track_model = TrackOnlyNetV3()
    track_model.load_state_dict(ckpt['track_model'])
    track_model.to(device)
    
    # 手动构建样本列表（需要文件路径）
    RD_VAL = "./dataset/train_cleandata/val"
    TRACK_VAL = "./dataset/track_enhanced_cleandata/val"
    
    # 收集鸟类样本文件路径
    bird_files = []
    bird_label_dir = os.path.join(RD_VAL, "2")  # 鸟类是类别2
    
    if os.path.exists(bird_label_dir):
        rd_files = [f for f in os.listdir(bird_label_dir) 
                   if f.endswith('.mat') and '_track' not in f and '_motion' not in f]
        for f in sorted(rd_files):
            bird_files.append({
                'rd_path': os.path.join(bird_label_dir, f),
                'filename': f
            })
    
    print(f"\n鸟类样本文件数: {len(bird_files)}")
    
    # 加载数据集
    val_ds = FusionDataLoaderV3(RD_VAL, TRACK_VAL, val=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)  # batch_size=1方便定位
    
    # 校准BN（用完整数据集）
    full_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    calibrate_bn(track_model, full_loader, device)
    
    # 遍历所有样本，找出鸟类样本
    print("\n分析鸟类样本...")
    
    misclassified_birds = []
    correct_birds = []
    sample_idx = 0
    bird_sample_idx = 0
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            true_label = y[0].item()
            
            if true_label == BIRD_CLASS:
                x_rd = x_rd.to(device)
                x_track = x_track.to(device)
                x_stats_d = x_stats.to(device)
                
                rd_probs = torch.softmax(rd_model(x_rd), dim=1).cpu().numpy()[0]
                track_probs = torch.softmax(track_model(x_track, x_stats_d), dim=1).cpu().numpy()[0]
                stats = x_stats[0].numpy()
                
                # 方案3融合
                rd_w, track_w = 0.55, 0.45
                instability = (stats[1]/1.0 + stats[7]/0.5 + stats[8]/5.0 + stats[9]/6.5) / 4.0
                rd_pred = np.argmax(rd_probs)
                if rd_pred == BIRD_CLASS:
                    rd_w += 0.10
                    track_w -= 0.10
                if instability > 1.2 and rd_pred == BIRD_CLASS:
                    rd_w += 0.10
                    track_w -= 0.10
                rd_w = max(0.3, min(0.8, rd_w))
                track_w = 1.0 - rd_w
                
                fused_probs = rd_w * rd_probs + track_w * track_probs
                pred = np.argmax(fused_probs)
                conf = fused_probs[pred]
                
                # 获取文件名（从数据集获取）
                if bird_sample_idx < len(bird_files):
                    filename = bird_files[bird_sample_idx]['filename']
                    rd_path = bird_files[bird_sample_idx]['rd_path']
                else:
                    filename = f"sample_{bird_sample_idx}"
                    rd_path = "unknown"
                
                sample_info = {
                    'idx': bird_sample_idx,
                    'filename': filename,
                    'rd_path': rd_path,
                    'pred': pred,
                    'pred_name': CLASS_NAMES.get(pred, f'类别{pred}'),
                    'conf': conf,
                    'rd_probs': rd_probs,
                    'track_probs': track_probs,
                    'fused_probs': fused_probs,
                    'rd_pred': np.argmax(rd_probs),
                    'track_pred': np.argmax(track_probs),
                    'instability': instability,
                    'stats': stats,
                }
                
                if pred != BIRD_CLASS:
                    misclassified_birds.append(sample_info)
                else:
                    correct_birds.append(sample_info)
                
                bird_sample_idx += 1
            
            sample_idx += 1
    
    print(f"\n鸟类样本总数: {len(misclassified_birds) + len(correct_birds)}")
    print(f"正确分类: {len(correct_birds)} ({100*len(correct_birds)/(len(misclassified_birds)+len(correct_birds)):.1f}%)")
    print(f"误分类: {len(misclassified_birds)} ({100*len(misclassified_birds)/(len(misclassified_birds)+len(correct_birds)):.1f}%)")
    
    # ==================== 输出误分类样本详情 ====================
    print("\n" + "="*70)
    print(f"误分类的鸟类样本详情 ({len(misclassified_birds)}个)")
    print("="*70)
    
    # 按误分类目标分组
    by_pred = defaultdict(list)
    for s in misclassified_birds:
        by_pred[s['pred']].append(s)
    
    for pred_class in sorted(by_pred.keys()):
        samples = by_pred[pred_class]
        pred_name = CLASS_NAMES.get(pred_class, f'类别{pred_class}')
        
        print(f"\n--- 误分类为 {pred_name} ({len(samples)}个) ---")
        
        for i, s in enumerate(samples):
            print(f"\n  [{i+1}] {s['filename']}")
            print(f"      路径: {s['rd_path']}")
            print(f"      融合置信度: {s['conf']:.3f}")
            print(f"      RD预测: {CLASS_NAMES.get(s['rd_pred'], s['rd_pred'])} (鸟={s['rd_probs'][BIRD_CLASS]:.3f})")
            print(f"      Track预测: {CLASS_NAMES.get(s['track_pred'], s['track_pred'])} (鸟={s['track_probs'][BIRD_CLASS]:.3f})")
            print(f"      融合概率: 鸟={s['fused_probs'][BIRD_CLASS]:.3f}, {pred_name}={s['fused_probs'][pred_class]:.3f}")
            print(f"      运动不稳定性: {s['instability']:.2f}")
    
    # ==================== 输出文件列表（方便复制） ====================
    print("\n" + "="*70)
    print("误分类鸟类文件列表（可复制）")
    print("="*70)
    
    print("\n所有误分类的鸟类文件:")
    for s in misclassified_birds:
        print(f"  {s['filename']}")
    
    # 按置信度分类
    print("\n" + "="*70)
    print("按置信度分类")
    print("="*70)
    
    high_conf_wrong = [s for s in misclassified_birds if s['conf'] >= 0.5]
    low_conf_wrong = [s for s in misclassified_birds if s['conf'] < 0.5]
    
    print(f"\n高置信度误分类 (conf >= 0.5): {len(high_conf_wrong)}个")
    for s in high_conf_wrong:
        print(f"  {s['filename']} → {s['pred_name']} (conf={s['conf']:.3f})")
    
    print(f"\n低置信度误分类 (conf < 0.5): {len(low_conf_wrong)}个")
    for s in low_conf_wrong:
        print(f"  {s['filename']} → {s['pred_name']} (conf={s['conf']:.3f})")
    
    # ==================== 保存到文件 ====================
    output_file = "./misclassified_birds_list.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("误分类的鸟类样本列表\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"总数: {len(misclassified_birds)}个\n\n")
        
        f.write("--- 高置信度误分类 (conf >= 0.5) ---\n")
        for s in high_conf_wrong:
            f.write(f"{s['filename']}\n")
            f.write(f"  路径: {s['rd_path']}\n")
            f.write(f"  误判为: {s['pred_name']}, 置信度: {s['conf']:.3f}\n")
            f.write(f"  RD鸟概率: {s['rd_probs'][BIRD_CLASS]:.3f}, Track鸟概率: {s['track_probs'][BIRD_CLASS]:.3f}\n\n")
        
        f.write("\n--- 低置信度误分类 (conf < 0.5) ---\n")
        for s in low_conf_wrong:
            f.write(f"{s['filename']}\n")
            f.write(f"  路径: {s['rd_path']}\n")
            f.write(f"  误判为: {s['pred_name']}, 置信度: {s['conf']:.3f}\n")
            f.write(f"  RD鸟概率: {s['rd_probs'][BIRD_CLASS]:.3f}, Track鸟概率: {s['track_probs'][BIRD_CLASS]:.3f}\n\n")
    
    print(f"\n详细列表已保存到: {output_file}")
    
    # ==================== 特征分析 ====================
    print("\n" + "="*70)
    print("误分类样本的特征分析")
    print("="*70)
    
    FEAT_NAMES = [
        'mean_vel', 'std_vel', 'max_vel', 'min_vel',
        'mean_vz', 'std_vz', 'mean_accel', 'max_accel',
        'turn_rate', 'heading_stab', 'mean_range', 'range_change',
        'mean_pitch', 'std_pitch', 'mean_amp', 'std_amp',
        'mean_snr', 'mean_pts', 'n_pts', 'track_len'
    ]
    
    if misclassified_birds and correct_birds:
        wrong_stats = np.array([s['stats'] for s in misclassified_birds])
        correct_stats = np.array([s['stats'] for s in correct_birds])
        
        print(f"\n{'特征':^15}|{'正确鸟类':^12}|{'误分类鸟类':^12}|{'差异':^10}")
        print("-" * 55)
        
        for i, name in enumerate(FEAT_NAMES):
            correct_mean = correct_stats[:, i].mean()
            wrong_mean = wrong_stats[:, i].mean()
            
            if abs(correct_mean) > 0.001:
                diff_pct = 100 * (wrong_mean - correct_mean) / abs(correct_mean)
            else:
                diff_pct = 0
            
            mark = " ⚠️" if abs(diff_pct) > 50 else ""
            print(f"{name:^15}|{correct_mean:^12.3f}|{wrong_mean:^12.3f}|{diff_pct:^+10.1f}%{mark}")


if __name__ == '__main__':
    main()