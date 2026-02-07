"""
问题航迹深度分析
================
目标：找出40条问题航迹的共同特点，指导特征增强方向

分析内容：
1. RD图特征：能量、对比度、峰值分布
2. 航迹特征：速度、航向、加速度等
3. 对比高置信度样本，找出差异
4. 输出具体改进建议
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
import re

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


def compute_rd_features(rd_data):
    """计算RD图的详细特征"""
    rd = rd_data.numpy().squeeze()  # [32, 64]
    
    features = {}
    
    # 基础统计
    features['mean'] = np.mean(rd)
    features['std'] = np.std(rd)
    features['max'] = np.max(rd)
    features['min'] = np.min(rd)
    
    # 能量特征
    features['total_energy'] = np.sum(rd ** 2)
    features['mean_energy'] = np.mean(rd ** 2)
    
    # 对比度（标准差/均值）
    features['contrast'] = features['std'] / (abs(features['mean']) + 1e-6)
    
    # 动态范围
    features['dynamic_range'] = features['max'] - features['min']
    
    # 峰值特征
    features['peak_ratio'] = features['max'] / (features['mean'] + 1e-6)
    
    # 速度维特征（列方向）
    vel_profile = np.mean(np.abs(rd), axis=0)  # [64]
    features['vel_peak_idx'] = np.argmax(vel_profile)
    features['vel_peak_val'] = np.max(vel_profile)
    features['vel_spread'] = np.std(vel_profile)
    
    # 零速度附近能量（假设第32列是零速度）
    zero_vel_idx = 32
    zero_band = vel_profile[max(0, zero_vel_idx-3):min(64, zero_vel_idx+4)]
    features['zero_vel_energy'] = np.sum(zero_band)
    features['zero_vel_ratio'] = features['zero_vel_energy'] / (np.sum(vel_profile) + 1e-6)
    
    # 时间维特征（行方向）
    time_profile = np.mean(np.abs(rd), axis=1)  # [32]
    features['time_var'] = np.var(time_profile)
    features['time_trend'] = time_profile[-1] - time_profile[0]  # 能量变化趋势
    
    # 微多普勒特征（速度维的高频成分）
    vel_fft = np.abs(np.fft.fft(vel_profile - np.mean(vel_profile)))
    features['micro_doppler'] = np.sum(vel_fft[5:32])  # 高频成分
    
    # 稀疏度（非零元素比例）
    threshold = features['mean'] + features['std']
    features['sparsity'] = np.sum(rd > threshold) / rd.size
    
    return features


def compute_track_features(track_data, stats_data):
    """计算航迹的详细特征"""
    track = track_data.numpy()  # [12, 16]
    stats = stats_data.numpy()  # [20]
    
    features = {}
    
    # 基础统计
    features['track_mean'] = np.mean(track)
    features['track_std'] = np.std(track)
    features['track_energy'] = np.sum(track ** 2)
    
    # 各通道的活跃度
    channel_vars = np.var(track, axis=1)  # [12]
    features['active_channels'] = np.sum(channel_vars > 0.1)
    features['max_channel_var'] = np.max(channel_vars)
    features['min_channel_var'] = np.min(channel_vars)
    
    # 时序变化
    time_diffs = np.diff(track, axis=1)  # [12, 15]
    features['temporal_change'] = np.mean(np.abs(time_diffs))
    features['temporal_smoothness'] = 1.0 / (np.std(time_diffs) + 1e-6)
    
    # 从stats提取关键特征
    features['mean_velocity'] = stats[0]  # 平均速度
    features['velocity_std'] = stats[1]   # 速度标准差
    features['max_velocity'] = stats[2]   # 最大速度
    features['mean_vz'] = stats[4]        # 平均垂直速度
    features['vz_std'] = stats[5]         # 垂直速度波动
    features['mean_accel'] = stats[6]     # 平均加速度
    features['max_accel'] = stats[7]      # 最大加速度
    features['turn_rate'] = stats[8]      # 平均转弯率
    features['heading_stability'] = stats[9]  # 航向稳定性
    features['mean_range'] = stats[10]    # 平均距离
    features['range_change'] = stats[11]  # 距离变化
    features['mean_pitch'] = stats[12]    # 平均俯仰
    features['pitch_std'] = stats[13]     # 俯仰波动
    features['n_points'] = stats[18]      # 点数
    features['track_length'] = stats[19]  # 轨迹长度
    
    return features


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 配置
    RD_VAL = "./dataset/train_cleandata/val"
    TRACK_VAL = "./dataset/track_enhanced_cleandata/val"
    VALID_CLASSES = [0, 1, 2, 3]
    CLASS_NAMES = {0: '轻型无人机', 1: '小型无人机', 2: '鸟类', 3: '空飘球', 4: '杂波', 5: '其它'}
    
    print("="*70)
    print("问题航迹深度分析")
    print("="*70)
    
    # 加载数据
    print("\n加载数据...")
    val_ds = FusionDataLoaderV3(RD_VAL, TRACK_VAL, val=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    
    # 加载模型
    print("加载模型...")
    
    v14_pths = glob.glob("./checkpoint/fusion_v14*/ckpt_best*.pth")
    if not v14_pths:
        v14_pths = glob.glob("./checkpoint/fusion_v13*/ckpt_best*.pth")
    CKPT_PATH = sorted(v14_pths)[-1]
    
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    track_weight = ckpt.get('best_fixed_weight', ckpt.get('track_weight', 0.45))
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
    
    # 收集所有样本的详细特征
    print("\n分析所有样本（这可能需要几分钟）...")
    
    all_samples = []
    sample_idx = 0
    
    with torch.no_grad():
        for x_rd, x_track, x_stats, y in val_loader:
            true_label = y[0].item()
            if true_label not in VALID_CLASSES:
                sample_idx += 1
                continue
            
            # 模型预测
            x_rd_dev = x_rd.to(device)
            x_track_dev = x_track.to(device)
            x_stats_dev = x_stats.to(device)
            
            rd_probs = torch.softmax(rd_model(x_rd_dev), dim=1)
            track_probs = torch.softmax(track_model(x_track_dev, x_stats_dev), dim=1)
            fused_probs = rd_weight * rd_probs + track_weight * track_probs
            fused_conf, fused_pred = fused_probs.max(dim=1)
            
            # 获取文件信息
            if sample_idx < len(val_ds.samples):
                _, rd_path, _ = val_ds.samples[sample_idx]
                filename = os.path.basename(rd_path)
                match = re.match(r'Track(\d+)_Label(\d+)_Group(\d+)_Points(\d+)-(\d+)', filename)
                if match:
                    track_id = int(match.group(1))
                    group_id = int(match.group(3))
                    point_start = int(match.group(4))
                    point_end = int(match.group(5))
                    n_points = point_end - point_start + 1
                else:
                    track_id, group_id, point_start, point_end, n_points = 0, 0, 0, 0, 0
            else:
                filename = f"sample_{sample_idx}"
                track_id, group_id, point_start, point_end, n_points = 0, 0, 0, 0, 0
            
            # 计算详细特征
            rd_features = compute_rd_features(x_rd[0])
            track_features = compute_track_features(x_track[0], x_stats[0])
            
            sample = {
                'idx': sample_idx,
                'filename': filename,
                'track_id': track_id,
                'group_id': group_id,
                'n_points': n_points,
                'true_label': true_label,
                'fused_pred': fused_pred[0].item(),
                'fused_conf': fused_conf[0].item(),
                'correct': fused_pred[0].item() == true_label,
                'rd_features': rd_features,
                'track_features': track_features,
            }
            
            all_samples.append(sample)
            sample_idx += 1
            
            # 进度显示
            if sample_idx % 200 == 0:
                print(f"  已处理 {sample_idx} 个样本...")
    
    print(f"总样本数: {len(all_samples)}")
    
    # 分离高/低置信度
    high_conf = [s for s in all_samples if s['fused_conf'] >= 0.5]
    low_conf = [s for s in all_samples if s['fused_conf'] < 0.5]
    
    print(f"高置信度: {len(high_conf)}个")
    print(f"低置信度: {len(low_conf)}个")
    
    # ========== 分析1: 问题航迹识别 ==========
    print("\n" + "="*70)
    print("分析1: 问题航迹识别")
    print("="*70)
    
    track_samples = defaultdict(list)
    for s in all_samples:
        track_samples[s['track_id']].append(s)
    
    problem_tracks = {}
    for track_id, samples in track_samples.items():
        low_conf_samples = [s for s in samples if s['fused_conf'] < 0.5]
        if len(low_conf_samples) >= 2:
            problem_tracks[track_id] = {
                'all_samples': samples,
                'low_conf_samples': low_conf_samples,
                'high_conf_samples': [s for s in samples if s['fused_conf'] >= 0.5],
                'true_label': samples[0]['true_label'],
            }
    
    print(f"\n问题航迹数: {len(problem_tracks)}条")
    
    # ========== 分析2: RD特征对比 ==========
    print("\n" + "="*70)
    print("分析2: RD特征对比（高置信度 vs 低置信度）")
    print("="*70)
    
    rd_feature_names = ['mean', 'std', 'contrast', 'dynamic_range', 'peak_ratio',
                        'total_energy', 'vel_spread', 'zero_vel_ratio', 'micro_doppler', 'sparsity']
    
    print(f"\n{'RD特征':^18}|{'高置信度':^12}|{'低置信度':^12}|{'差异%':^10}|{'判断':^10}")
    print("-" * 65)
    
    significant_rd_features = []
    
    for feat_name in rd_feature_names:
        high_vals = [s['rd_features'][feat_name] for s in high_conf]
        low_vals = [s['rd_features'][feat_name] for s in low_conf]
        
        high_mean = np.mean(high_vals)
        low_mean = np.mean(low_vals)
        
        if abs(high_mean) > 1e-6:
            diff_pct = 100 * (low_mean - high_mean) / abs(high_mean)
        else:
            diff_pct = 0
        
        if abs(diff_pct) > 20:
            judgment = "⚠ 显著"
            significant_rd_features.append((feat_name, diff_pct))
        elif abs(diff_pct) > 10:
            judgment = "注意"
        else:
            judgment = "正常"
        
        print(f"{feat_name:^18}|{high_mean:^12.3f}|{low_mean:^12.3f}|{diff_pct:^+10.1f}|{judgment:^10}")
    
    # ========== 分析3: 航迹特征对比 ==========
    print("\n" + "="*70)
    print("分析3: 航迹特征对比（高置信度 vs 低置信度）")
    print("="*70)
    
    track_feature_names = ['mean_velocity', 'velocity_std', 'max_velocity', 'mean_vz', 'vz_std',
                           'mean_accel', 'max_accel', 'turn_rate', 'heading_stability',
                           'mean_range', 'range_change', 'n_points', 'track_length',
                           'temporal_change', 'active_channels']
    
    print(f"\n{'航迹特征':^18}|{'高置信度':^12}|{'低置信度':^12}|{'差异%':^10}|{'判断':^10}")
    print("-" * 65)
    
    significant_track_features = []
    
    for feat_name in track_feature_names:
        high_vals = [s['track_features'][feat_name] for s in high_conf]
        low_vals = [s['track_features'][feat_name] for s in low_conf]
        
        high_mean = np.mean(high_vals)
        low_mean = np.mean(low_vals)
        
        if abs(high_mean) > 1e-6:
            diff_pct = 100 * (low_mean - high_mean) / abs(high_mean)
        else:
            diff_pct = 0
        
        if abs(diff_pct) > 20:
            judgment = "⚠ 显著"
            significant_track_features.append((feat_name, diff_pct))
        elif abs(diff_pct) > 10:
            judgment = "注意"
        else:
            judgment = "正常"
        
        print(f"{feat_name:^18}|{high_mean:^12.3f}|{low_mean:^12.3f}|{diff_pct:^+10.1f}|{judgment:^10}")
    
    # ========== 分析4: 按类别分析 ==========
    print("\n" + "="*70)
    print("分析4: 各类别的问题航迹特点")
    print("="*70)
    
    for c in VALID_CLASSES:
        c_problem_tracks = {tid: info for tid, info in problem_tracks.items() if info['true_label'] == c}
        
        if not c_problem_tracks:
            continue
        
        print(f"\n【{CLASS_NAMES[c]}】 问题航迹: {len(c_problem_tracks)}条")
        
        c_low_samples = []
        for tid, info in c_problem_tracks.items():
            c_low_samples.extend(info['low_conf_samples'])
        
        c_high_samples = [s for s in high_conf if s['true_label'] == c]
        
        if not c_low_samples or not c_high_samples:
            continue
        
        print(f"  低置信度样本: {len(c_low_samples)}个")
        print(f"  高置信度样本: {len(c_high_samples)}个")
        
        # RD特征差异
        print(f"\n  RD特征差异（低 vs 高）:")
        for feat_name in ['contrast', 'micro_doppler', 'zero_vel_ratio', 'vel_spread']:
            high_mean = np.mean([s['rd_features'][feat_name] for s in c_high_samples])
            low_mean = np.mean([s['rd_features'][feat_name] for s in c_low_samples])
            if abs(high_mean) > 1e-6:
                diff_pct = 100 * (low_mean - high_mean) / abs(high_mean)
            else:
                diff_pct = 0
            mark = "⚠" if abs(diff_pct) > 20 else ""
            print(f"    {feat_name}: {low_mean:.3f} vs {high_mean:.3f} ({diff_pct:+.1f}%) {mark}")
        
        # 航迹特征差异
        print(f"\n  航迹特征差异:")
        for feat_name in ['mean_velocity', 'velocity_std', 'turn_rate', 'n_points']:
            high_mean = np.mean([s['track_features'][feat_name] for s in c_high_samples])
            low_mean = np.mean([s['track_features'][feat_name] for s in c_low_samples])
            if abs(high_mean) > 1e-6:
                diff_pct = 100 * (low_mean - high_mean) / abs(high_mean)
            else:
                diff_pct = 0
            mark = "⚠" if abs(diff_pct) > 20 else ""
            print(f"    {feat_name}: {low_mean:.3f} vs {high_mean:.3f} ({diff_pct:+.1f}%) {mark}")
    
    # ========== 分析5: TOP问题航迹 ==========
    print("\n" + "="*70)
    print("分析5: TOP10问题航迹详细")
    print("="*70)
    
    sorted_problem_tracks = sorted(problem_tracks.items(), 
                                   key=lambda x: len(x[1]['low_conf_samples']), reverse=True)
    
    for track_id, info in sorted_problem_tracks[:10]:
        low_samples = info['low_conf_samples']
        high_samples = info['high_conf_samples']
        true_label = info['true_label']
        
        print(f"\n航迹 {track_id} ({CLASS_NAMES[true_label]}):")
        print(f"  低置信度: {len(low_samples)}个, 高置信度: {len(high_samples)}个")
        
        # 预测分布
        pred_dist = defaultdict(int)
        for s in low_samples:
            pred_dist[s['fused_pred']] += 1
        print(f"  预测分布: ", end='')
        for pred, count in sorted(pred_dist.items(), key=lambda x: -x[1]):
            print(f"{CLASS_NAMES.get(pred, str(pred))}({count}) ", end='')
        print()
        
        # 特征
        avg_contrast = np.mean([s['rd_features']['contrast'] for s in low_samples])
        avg_micro = np.mean([s['rd_features']['micro_doppler'] for s in low_samples])
        avg_vel = np.mean([s['track_features']['mean_velocity'] for s in low_samples])
        avg_points = np.mean([s['track_features']['n_points'] for s in low_samples])
        
        print(f"  RD对比度: {avg_contrast:.3f}, 微多普勒: {avg_micro:.1f}")
        print(f"  平均速度: {avg_vel:.2f}, 平均点数: {avg_points:.1f}")
    
    # ========== 结论 ==========
    print("\n" + "="*70)
    print("分析结论与改进建议")
    print("="*70)
    
    print(f"""
═══════════════════════════════════════════════════════════════════
                        关键发现
═══════════════════════════════════════════════════════════════════

1. RD特征差异显著的指标:""")
    
    if significant_rd_features:
        for feat, diff in significant_rd_features:
            direction = "更低" if diff < 0 else "更高"
            print(f"   ▶ {feat}: 低置信度样本{direction} ({diff:+.1f}%)")
    else:
        print("   （无显著差异）")
    
    print(f"""
2. 航迹特征差异显著的指标:""")
    
    if significant_track_features:
        for feat, diff in significant_track_features:
            direction = "更低" if diff < 0 else "更高"
            print(f"   ▶ {feat}: 低置信度样本{direction} ({diff:+.1f}%)")
    else:
        print("   （无显著差异）")
    
    print(f"""
═══════════════════════════════════════════════════════════════════
                        改进建议
═══════════════════════════════════════════════════════════════════
""")
    
    # RD改进建议
    print("[RD预处理改进]")
    
    contrast_diff = next((d for f, d in significant_rd_features if 'contrast' in f), None)
    if contrast_diff and contrast_diff < -10:
        print(f"""
  ⭐ 问题: 低置信度样本对比度低 ({contrast_diff:+.1f}%)
  
  解决方案 - 在MATLAB中修改:
  ```matlab
  % 原代码
  compressed_vectors(i, :) = sum(abs(frames_rd{{i}}), 1);
  
  % 改进1: 增加对比度增强
  rd_temp = abs(frames_rd{{i}});
  rd_enhanced = (rd_temp - mean(rd_temp(:))) ./ (std(rd_temp(:)) + 1e-6);
  rd_enhanced = rd_enhanced * 2;  % 增强系数
  compressed_vectors(i, :) = sum(rd_enhanced, 1);
  
  % 改进2: 或者使用max代替sum保留峰值
  compressed_vectors(i, :) = max(abs(frames_rd{{i}}), [], 1);
  ```
""")
    
    micro_diff = next((d for f, d in significant_rd_features if 'micro' in f), None)
    if micro_diff and abs(micro_diff) > 10:
        print(f"""
  ⭐ 问题: 微多普勒特征差异 ({micro_diff:+.1f}%)
  
  解决方案 - 提取距离维标准差:
  ```matlab
  % 在距离维压缩时，同时提取std
  compressed_sum = sum(abs(frames_rd{{i}}), 1);   % 能量
  compressed_std = std(abs(frames_rd{{i}}), 0, 1); % 微多普勒
  
  % 组合为2通道
  compressed_vectors(i, :, 1) = compressed_sum;
  compressed_vectors(i, :, 2) = compressed_std;
  ```
""")
    
    zero_vel_diff = next((d for f, d in significant_rd_features if 'zero_vel' in f), None)
    if zero_vel_diff and abs(zero_vel_diff) > 10:
        print(f"""
  ⭐ 问题: 零速度能量分布差异 ({zero_vel_diff:+.1f}%)
  
  解决方案 - 零速度抑制/增强:
  ```matlab
  % 在process_frame_to_rd中，找到零速度列
  zero_col = VelocityNum/2;
  
  % 方案1: 抑制零速度（减少杂波）
  rd_matrix(:, zero_col-2:zero_col+2) = rd_matrix(:, zero_col-2:zero_col+2) * 0.5;
  
  % 方案2: 或者单独提取零速度特征
  zero_vel_feature = rd_matrix(:, zero_col);
  ```
""")
    
    # 航迹改进建议
    print("\n[航迹特征改进]")
    
    vel_diff = next((d for f, d in significant_track_features if 'velocity' in f and 'std' not in f), None)
    if vel_diff and abs(vel_diff) > 10:
        print(f"""
  ⭐ 问题: 速度特征差异 ({vel_diff:+.1f}%)
  
  解决方案 - 添加频域特征:
  ```matlab
  % 在extract_enhanced_features中添加
  vel_fft = abs(fft(vel - mean(vel)));
  track_stats(21) = max(vel_fft(2:floor(end/2)));  % 主频幅度
  track_stats(22) = std(vel_fft(2:floor(end/2)));  % 频域波动
  ```
""")
    
    points_diff = next((d for f, d in significant_track_features if 'n_points' in f), None)
    if points_diff and points_diff < -10:
        print(f"""
  ⭐ 问题: 低置信度样本点数较少 ({points_diff:+.1f}%)
  
  这说明短航迹更难识别，建议:
  1. 对短航迹(<8点)使用更多帧累积
  2. 或者训练时对短航迹做数据增强
  
  MATLAB修改 (process_track_file函数):
  ```matlab
  % 修改目标帧数判断
  if n_track_points < 8
      target_frames = 24;  % 短航迹累积更多帧
  else
      target_frames = 16;
  end
  ```
""")
    
    turn_diff = next((d for f, d in significant_track_features if 'turn' in f or 'heading' in f), None)
    if turn_diff and abs(turn_diff) > 10:
        print(f"""
  ⭐ 问题: 航向/转弯特征差异 ({turn_diff:+.1f}%)
  
  解决方案 - 添加运动曲率特征:
  ```matlab
  % 在extract_enhanced_features中
  curvature = abs(d_heading) ./ (vel + 0.1);
  track_stats(23) = mean(curvature);
  track_stats(24) = max(curvature);
  ```
""")
    
    print(f"""
═══════════════════════════════════════════════════════════════════
                        执行优先级
═══════════════════════════════════════════════════════════════════

  优先级1 (最可能有效):
    □ 修改RD距离维压缩方式（sum→max 或 添加std通道）
    
  优先级2:
    □ 添加航迹频域特征
    □ 短航迹特殊处理
    
  优先级3:
    □ 零速度处理优化
    □ 运动曲率特征

  注意: 任何修改都需要重新生成数据并重新训练模型
""")
    
    # 保存报告
    report_file = "problem_tracks_analysis.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("问题航迹详细报告\n")
        f.write("="*60 + "\n\n")
        
        for track_id, info in sorted_problem_tracks:
            f.write(f"航迹 {track_id} ({CLASS_NAMES[info['true_label']]})\n")
            f.write(f"  低置信度: {len(info['low_conf_samples'])}个\n")
            
            for s in info['low_conf_samples']:
                f.write(f"    - {s['filename']}\n")
                f.write(f"      置信度: {s['fused_conf']:.3f}, 预测: {CLASS_NAMES.get(s['fused_pred'], s['fused_pred'])}\n")
            f.write("\n")
    
    print(f"\n详细报告已保存到: {report_file}")


if __name__ == '__main__':
    main()