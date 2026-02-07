"""
V15 单独测试脚本（修复版）
==========================
确保正确加载28维特征数据
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import scipy.io as scio
import warnings
import glob
from collections import defaultdict

warnings.filterwarnings("ignore")

from drsncww import rsnet34


class TrackOnlyNetV4(nn.Module):
    """V15用的28维版本"""
    def __init__(self, num_classes=6, stats_dim=28):
        super().__init__()
        self.stats_dim = stats_dim
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
            nn.Linear(stats_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 6)
        )
    
    def forward(self, x_temporal, x_stats):
        feat_temporal = self.temporal_net(x_temporal)
        feat_stats = self.stats_net(x_stats)
        return self.classifier(torch.cat([feat_temporal, feat_stats], dim=1))


class FusionDataLoaderV4_Inline:
    """内联的V4数据加载器，确保正确加载28维特征"""
    def __init__(self, rd_root_dir, track_feat_dir, val=False, stats_dim=28):
        self.rd_root_dir = rd_root_dir
        self.track_feat_dir = track_feat_dir
        self.val = val
        self.stats_dim = stats_dim
        self.data_rows = 32
        self.data_cols = 64
        self.samples = []
        
        if os.path.exists(rd_root_dir):
            for label_str in os.listdir(rd_root_dir):
                rd_label_dir = os.path.join(rd_root_dir, label_str)
                if not os.path.isdir(rd_label_dir):
                    continue
                try:
                    label = int(label_str)
                except:
                    continue
                
                track_label_dir = None
                if track_feat_dir and os.path.exists(track_feat_dir):
                    potential_track_dir = os.path.join(track_feat_dir, label_str)
                    if os.path.exists(potential_track_dir):
                        track_label_dir = potential_track_dir
                
                rd_files = [f for f in os.listdir(rd_label_dir) 
                           if f.endswith('.mat') and '_track' not in f and '_motion' not in f]
                
                for rd_file in rd_files:
                    rd_path = os.path.join(rd_label_dir, rd_file)
                    track_path = None
                    if track_label_dir:
                        base_name = rd_file.replace('.mat', '')
                        possible_names = [f"{base_name}_track.mat", f"{base_name}.mat"]
                        for name in possible_names:
                            potential_path = os.path.join(track_label_dir, name)
                            if os.path.exists(potential_path):
                                track_path = potential_path
                                break
                    self.samples.append((label, rd_path, track_path))
        
        matched = sum(1 for s in self.samples if s[2] is not None)
        total = len(self.samples)
        print(f"[{'验证' if val else '训练'}] RD样本数: {total}, 航迹匹配: {matched} ({100*matched/max(total,1):.1f}%)")
        print(f"   -> 统计特征维度: {self.stats_dim}")
        
        # 检查第一个样本的维度
        if matched > 0:
            for s in self.samples:
                if s[2]:
                    try:
                        mat = scio.loadmat(s[2])
                        if 'track_stats' in mat:
                            actual_dim = len(mat['track_stats'].flatten())
                            print(f"   -> 实际数据维度: {actual_dim}")
                            if actual_dim != stats_dim:
                                print(f"   ⚠️ 警告: 数据维度({actual_dim})与期望维度({stats_dim})不匹配!")
                    except:
                        pass
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, rd_path, track_path = self.samples[idx]
        
        # 加载RD
        try:
            original_data = scio.loadmat(rd_path)
            if 'data' in original_data:
                data0 = original_data['data']
            elif 'rangePower1' in original_data:
                data0 = original_data['rangePower1']
            else:
                data_keys = [k for k in original_data.keys() if not k.startswith('__')]
                data0 = original_data[data_keys[0]] if data_keys else np.zeros((32, 64))
            
            data1 = np.array(data0)
            # 裁剪/填充
            if data1.shape[0] > self.data_rows:
                data2 = data1[:self.data_rows, :]
            else:
                data2 = np.pad(data1, ((0, self.data_rows - data1.shape[0]), (0, 0)), 'constant')
            if data2.shape[1] > self.data_cols:
                data2 = data2[:, :self.data_cols]
            else:
                data2 = np.pad(data2, ((0, 0), (0, self.data_cols - data2.shape[1])), 'constant')
            
            data2 = np.abs(data2) + 1e-7
            data_log = 20 * np.log(data2)
            data_norm = (data_log - np.mean(data_log)) / (np.std(data_log) + 1e-6)
            x_rd = torch.from_numpy(data_norm.reshape(1, self.data_rows, self.data_cols)).float()
        except:
            x_rd = torch.zeros(1, self.data_rows, self.data_cols)

        # 加载航迹特征
        x_track = torch.zeros(12, 16)
        x_stats = torch.zeros(self.stats_dim)
        
        if track_path and os.path.exists(track_path):
            try:
                mat_t = scio.loadmat(track_path)
                
                if 'track_data' in mat_t:
                    feat = mat_t['track_data'].astype(np.float32)
                    if feat.shape[0] == 16 and feat.shape[1] == 12:
                        feat = feat.T
                    n_ch = min(feat.shape[0], 12)
                    n_len = min(feat.shape[1], 16)
                    x_track[:n_ch, :n_len] = torch.from_numpy(feat[:n_ch, :n_len].copy())
                
                if 'track_stats' in mat_t:
                    stats = mat_t['track_stats'].astype(np.float32).flatten()
                    actual_len = len(stats)
                    if actual_len >= self.stats_dim:
                        x_stats = torch.from_numpy(stats[:self.stats_dim].copy())
                    else:
                        x_stats[:actual_len] = torch.from_numpy(stats.copy())
                    x_stats = torch.nan_to_num(x_stats, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception as e:
                pass
        
        return x_rd, x_track, x_stats, label


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球'}
    
    print("="*70)
    print("V15 详细测试（修复版）")
    print("="*70)
    
    # 检查V4数据目录
    rd_val_dir = "./dataset/train_cleandata/val"
    
    # 尝试多个可能的V4目录
    possible_track_dirs = [
        "./dataset/track_enhanced_v4_cleandata/val",
        "./dataset/track_enhanced_v4/val",
        "./dataset/track_v4_cleandata/val",
    ]
    
    track_val_dir = None
    for d in possible_track_dirs:
        if os.path.exists(d):
            track_val_dir = d
            print(f"找到V4航迹目录: {d}")
            break
    
    if not track_val_dir:
        print("错误: 找不到V4航迹特征目录!")
        print("请确认目录名称，可能的目录:")
        print("  - ./dataset/track_enhanced_v4_cleandata/val")
        return
    
    # 检查目录内容
    print(f"\n检查目录结构:")
    for label in range(4):
        label_dir = os.path.join(track_val_dir, str(label))
        if os.path.exists(label_dir):
            files = [f for f in os.listdir(label_dir) if f.endswith('.mat')]
            print(f"  类别{label}: {len(files)}个文件")
            if files:
                # 检查第一个文件的维度
                try:
                    mat = scio.loadmat(os.path.join(label_dir, files[0]))
                    if 'track_stats' in mat:
                        dim = len(mat['track_stats'].flatten())
                        print(f"    -> track_stats维度: {dim}")
                except:
                    pass
    
    # 加载V15 checkpoint
    v15_pths = glob.glob("./checkpoint/fusion_v15*/ckpt_best*.pth")
    if not v15_pths:
        print("\n错误: 找不到V15 checkpoint!")
        return
    
    v15_ckpt_path = sorted(v15_pths)[-1]
    print(f"\n加载checkpoint: {v15_ckpt_path}")
    
    v15_ckpt = torch.load(v15_ckpt_path, map_location='cpu')
    v15_weight = v15_ckpt.get('best_fixed_weight', 0.5)
    stats_dim = v15_ckpt.get('stats_dim', 28)
    
    print(f"  Track权重: {v15_weight}")
    print(f"  stats_dim: {stats_dim}")
    
    # 加载RD模型
    rd_model = rsnet34()
    rd_pths = glob.glob("./checkpoint/*95*.pth") + glob.glob("./checkpoint/*94*.pth")
    rd_pths = [p for p in rd_pths if 'fusion' not in p]
    if rd_pths:
        rd_ckpt = torch.load(rd_pths[0], map_location='cpu')
        rd_model.load_state_dict(rd_ckpt.get('net_weight', rd_ckpt))
        print(f"  RD模型: {rd_pths[0]}")
    rd_model.to(device).eval()
    
    # 加载Track模型
    track_model = TrackOnlyNetV4(stats_dim=stats_dim)
    track_model.load_state_dict(v15_ckpt['track_model'])
    track_model.to(device).eval()
    
    # 加载数据
    print(f"\n加载数据...")
    val_ds = FusionDataLoaderV4_Inline(rd_val_dir, track_val_dir, val=True, stats_dim=stats_dim)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # 测试
    print(f"\n开始测试...")
    rd_w = 1.0 - v15_weight
    track_w = v15_weight
    
    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'high_correct': 0, 'high_total': 0})
    all_preds = []
    all_labels = []
    all_confs = []
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            x_rd = x_rd.to(device)
            x_track = x_track.to(device)
            x_stats = x_stats.to(device)
            
            rd_probs = torch.softmax(rd_model(x_rd), dim=1)
            track_probs = torch.softmax(track_model(x_track, x_stats), dim=1)
            fused_probs = rd_w * rd_probs + track_w * track_probs
            conf, pred = fused_probs.max(dim=1)
            
            for i in range(len(y)):
                true_label = y[i].item()
                if true_label not in VALID_CLASSES:
                    continue
                
                all_preds.append(pred[i].item())
                all_labels.append(true_label)
                all_confs.append(conf[i].item())
                
                class_stats[true_label]['total'] += 1
                if pred[i].item() == true_label:
                    class_stats[true_label]['correct'] += 1
                
                if conf[i].item() >= 0.5:
                    class_stats[true_label]['high_total'] += 1
                    if pred[i].item() == true_label:
                        class_stats[true_label]['high_correct'] += 1
    
    # 输出结果
    print("\n" + "="*70)
    print("测试结果")
    print("="*70)
    
    total_samples = len(all_preds)
    total_correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    high_conf_samples = sum(1 for c in all_confs if c >= 0.5)
    high_conf_correct = sum(1 for p, l, c in zip(all_preds, all_labels, all_confs) if c >= 0.5 and p == l)
    
    print(f"\n总样本数: {total_samples}")
    print(f"总体准确率: {100*total_correct/total_samples:.2f}%")
    print(f"高置信度样本: {high_conf_samples} ({100*high_conf_samples/total_samples:.1f}%)")
    print(f"高置信度准确率: {100*high_conf_correct/max(high_conf_samples,1):.2f}%")
    
    # 分类别
    print(f"\n{'类别':^12}|{'总数':^8}|{'准确率':^10}|{'高置信度':^10}|{'高置信度准确率':^14}")
    print("-" * 60)
    
    for c in VALID_CLASSES:
        stats = class_stats[c]
        acc = 100 * stats['correct'] / max(stats['total'], 1)
        high_acc = 100 * stats['high_correct'] / max(stats['high_total'], 1)
        print(f"{CLASS_NAMES[c]:^12}|{stats['total']:^8}|{acc:^10.2f}%|{stats['high_total']:^10}|{high_acc:^14.2f}%")
    
    # 预测分布
    print(f"\n预测分布:")
    pred_counts = defaultdict(int)
    for p in all_preds:
        pred_counts[p] += 1
    for c in sorted(pred_counts.keys()):
        name = CLASS_NAMES.get(c, f"类别{c}")
        print(f"  {name}: {pred_counts[c]}个 ({100*pred_counts[c]/total_samples:.1f}%)")
    
    # 置信度分布
    print(f"\n置信度分布:")
    conf_bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    for i in range(len(conf_bins)-1):
        low, high = conf_bins[i], conf_bins[i+1]
        count = sum(1 for c in all_confs if low <= c < high)
        print(f"  [{low:.1f}, {high:.1f}): {count}个 ({100*count/total_samples:.1f}%)")


if __name__ == '__main__':
    main()