"""
诊断航迹特征数据
================
检查track_data和track_stats的实际内容
"""

import os
import scipy.io as scio
import numpy as np
import glob

def check_track_features(track_dir, num_samples=5):
    """检查航迹特征文件的内容"""
    
    print(f"\n检查目录: {track_dir}")
    print("="*70)
    
    if not os.path.exists(track_dir):
        print(f"错误: 目录不存在!")
        return
    
    for label in range(4):
        label_dir = os.path.join(track_dir, str(label))
        if not os.path.exists(label_dir):
            continue
        
        mat_files = glob.glob(os.path.join(label_dir, "*.mat"))
        print(f"\n类别 {label}: 共 {len(mat_files)} 个文件")
        
        if not mat_files:
            continue
        
        # 检查前几个文件
        for i, mat_path in enumerate(mat_files[:num_samples]):
            filename = os.path.basename(mat_path)
            print(f"\n  [{i+1}] {filename}")
            
            try:
                mat = scio.loadmat(mat_path)
                
                # 列出所有键
                keys = [k for k in mat.keys() if not k.startswith('__')]
                print(f"      键: {keys}")
                
                # 检查 track_data
                if 'track_data' in mat:
                    td = mat['track_data']
                    print(f"      track_data: shape={td.shape}, dtype={td.dtype}")
                    print(f"                  min={np.nanmin(td):.4f}, max={np.nanmax(td):.4f}, mean={np.nanmean(td):.4f}")
                    print(f"                  nan_count={np.sum(np.isnan(td))}, zero_count={np.sum(td==0)}")
                    
                    # 检查每个通道
                    if td.shape[0] == 12:
                        non_zero_channels = np.sum(np.abs(td).sum(axis=1) > 1e-6)
                        print(f"                  非零通道数: {non_zero_channels}/12")
                else:
                    print(f"      ⚠️ 没有 track_data!")
                
                # 检查 track_stats
                if 'track_stats' in mat:
                    ts = mat['track_stats'].flatten()
                    print(f"      track_stats: shape={ts.shape}, dtype={ts.dtype}")
                    print(f"                   min={np.nanmin(ts):.4f}, max={np.nanmax(ts):.4f}, mean={np.nanmean(ts):.4f}")
                    print(f"                   nan_count={np.sum(np.isnan(ts))}, zero_count={np.sum(ts==0)}")
                    
                    # 显示前几个值
                    print(f"                   前10个值: {ts[:10]}")
                    
                    # 检查新增的8维特征(21-28)
                    if len(ts) >= 28:
                        print(f"                   新增特征(21-28): {ts[20:28]}")
                else:
                    print(f"      ⚠️ 没有 track_stats!")
                    
            except Exception as e:
                print(f"      ❌ 错误: {e}")
        
        # 统计整体情况
        print(f"\n  --- 类别{label}整体统计 ---")
        
        all_stats = []
        all_data = []
        error_count = 0
        
        for mat_path in mat_files:
            try:
                mat = scio.loadmat(mat_path)
                if 'track_stats' in mat:
                    all_stats.append(mat['track_stats'].flatten())
                if 'track_data' in mat:
                    all_data.append(mat['track_data'])
            except:
                error_count += 1
        
        if all_stats:
            all_stats = np.array(all_stats)
            print(f"  track_stats整体: shape={all_stats.shape}")
            print(f"    全零行数: {np.sum(np.all(all_stats == 0, axis=1))}/{len(all_stats)}")
            print(f"    含NaN行数: {np.sum(np.any(np.isnan(all_stats), axis=1))}/{len(all_stats)}")
            
            # 每个特征的统计
            print(f"    各特征均值:")
            feature_names = [
                'mean_vel', 'std_vel', 'max_vel', 'min_vel',  # 1-4
                'mean_vz', 'std_vz',                          # 5-6
                'mean_accel', 'max_accel',                    # 7-8
                'turn_rate', 'heading_stab',                  # 9-10
                'mean_range', 'range_change',                 # 11-12
                'mean_pitch', 'std_pitch',                    # 13-14
                'mean_amp', 'std_amp',                        # 15-16
                'mean_snr', 'mean_pts',                       # 17-18
                'n_pts', 'track_len',                         # 19-20
                'stability', 'curv_mean', 'curv_max', 'curv_std',  # 21-24
                'vel_cv', 'vz_ratio', 'fft_peak', 'dir_consist'   # 25-28
            ]
            
            for j in range(min(28, all_stats.shape[1])):
                col = all_stats[:, j]
                name = feature_names[j] if j < len(feature_names) else f'feat_{j+1}'
                print(f"      [{j+1:2d}] {name:12s}: mean={np.nanmean(col):8.3f}, std={np.nanstd(col):8.3f}, zeros={np.sum(col==0)}")
        
        if error_count > 0:
            print(f"  ⚠️ 读取错误: {error_count}个文件")


def main():
    print("航迹特征数据诊断")
    print("="*70)
    
    # 检查可能的目录
    possible_dirs = [
        "./dataset/track_enhanced_v4_cleandata/val",
        "./dataset/track_enhanced_v4/val",
        "./dataset/track_v4_cleandata/val",
        "./dataset/track_enhanced_cleandata/val",  # V3目录对比
    ]
    
    for d in possible_dirs:
        if os.path.exists(d):
            check_track_features(d, num_samples=3)
        else:
            print(f"\n目录不存在: {d}")


if __name__ == '__main__':
    main()