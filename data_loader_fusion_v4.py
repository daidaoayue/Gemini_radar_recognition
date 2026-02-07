"""
双流融合数据加载器 V4
====================
支持28维航迹统计特征（原20维 + 新增8维运动稳定性特征）

新增特征说明:
  21. stability_score  - 运动稳定性综合指数
  22. curvature_mean   - 运动曲率均值
  23. curvature_max    - 运动曲率最大值
  24. curvature_std    - 运动曲率标准差
  25. velocity_cv      - 速度变异系数
  26. vz_ratio         - 垂直运动比例
  27. vel_fft_peak     - 速度FFT主频幅度
  28. direction_consistency - 运动方向一致性
"""

import os
import torch
import scipy.io as scio
import numpy as np
from torch.utils.data import Dataset
import random


class FusionDataLoaderV4(Dataset):
    def __init__(self, rd_root_dir, track_feat_dir, val=False, stats_dim=28):
        """
        Args:
            rd_root_dir: RD数据目录
            track_feat_dir: 航迹特征目录
            val: 是否为验证模式
            stats_dim: 统计特征维度（20或28）
        """
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
                        possible_names = [
                            f"{base_name}_track.mat",
                            f"{base_name}.mat",
                        ]
                        for name in possible_names:
                            potential_path = os.path.join(track_label_dir, name)
                            if os.path.exists(potential_path):
                                track_path = potential_path
                                break
                    
                    self.samples.append((label, rd_path, track_path))
        
        matched = sum(1 for s in self.samples if s[2] is not None)
        total = len(self.samples)
        match_rate = 100 * matched / max(total, 1)
        
        print(f"[{'验证' if val else '训练'}] RD样本数: {total}, 航迹匹配: {matched} ({match_rate:.1f}%)")
        print(f"   -> 统计特征维度: {self.stats_dim}")

    def __len__(self):
        return len(self.samples)

    def _augment(self, data):
        """数据增强"""
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.01, data.shape)
            data = data + noise
            
        if random.random() < 0.5:
            t0 = random.randint(0, self.data_rows - 5)
            data[:, t0:t0+5, :] = 0
        
        if random.random() < 0.5:
            f0 = random.randint(0, self.data_cols - 5)
            data[:, :, f0:f0+5] = 0
            
        return data

    def __getitem__(self, idx):
        label, rd_path, track_path = self.samples[idx]
        
        # ============================================
        # A. 加载 RD 矩阵
        # ============================================
        try:
            original_data = scio.loadmat(rd_path)
            
            if 'data' in original_data:
                data0 = original_data['data']
            elif 'rangePower1' in original_data:
                data0 = original_data['rangePower1']
            else:
                data_keys = [k for k in original_data.keys() if not k.startswith('__')]
                if data_keys:
                    data0 = original_data[data_keys[0]]
                else:
                    raise ValueError("无法找到数据")
            
            data1 = np.array(data0)
            
            curr_rows = data1.shape[0]
            if curr_rows > self.data_rows:
                data2 = data1[:self.data_rows, :]
            else:
                num_pad = self.data_rows - data1.shape[0]
                data2 = np.pad(data1, ((0, num_pad), (0, 0)), 'constant', constant_values=(0, 0))
            
            curr_cols = data1.shape[1]
            if curr_cols > self.data_cols:
                data2 = data2[:, :self.data_cols]
            else:
                num_pad = self.data_cols - data1.shape[1]
                data2 = np.pad(data2, ((0, 0), (0, num_pad)), 'constant', constant_values=(0, 0))
            
            data2 = np.abs(data2) + 1e-7
            data_log = 20 * np.log(data2)
            
            mean_val = np.mean(data_log)
            std_val = np.std(data_log)
            data_norm = (data_log - mean_val) / (std_val + 1e-6)
            
            _data = data_norm.reshape((1, self.data_rows, self.data_cols))
            
            if not self.val:
                _data = self._augment(_data)
            
            x_rd = torch.from_numpy(_data).float()
            
        except Exception as e:
            x_rd = torch.zeros(1, self.data_rows, self.data_cols)

        # ============================================
        # B. 加载航迹特征
        # ============================================
        x_track = torch.zeros(12, 16)
        x_stats = torch.zeros(self.stats_dim)
        
        if track_path and os.path.exists(track_path):
            try:
                mat_t = scio.loadmat(track_path)
                
                # 时序特征 [12, 16]
                if 'track_data' in mat_t:
                    feat = mat_t['track_data'].astype(np.float32)
                    
                    if feat.shape[0] == 16 and feat.shape[1] == 12:
                        feat = feat.T
                    
                    n_channels = min(feat.shape[0], 12)
                    n_length = min(feat.shape[1], 16)
                    
                    x_track[:n_channels, :n_length] = torch.from_numpy(feat[:n_channels, :n_length].copy())
                
                # 统计特征 [stats_dim]
                if 'track_stats' in mat_t:
                    stats = mat_t['track_stats'].astype(np.float32).flatten()
                    
                    # 处理维度不匹配
                    actual_len = len(stats)
                    if actual_len >= self.stats_dim:
                        x_stats = torch.from_numpy(stats[:self.stats_dim].copy())
                    else:
                        # 如果数据是旧版本(20维)，填充0
                        x_stats[:actual_len] = torch.from_numpy(stats.copy())
                    
                    # 处理NaN和Inf
                    x_stats = torch.nan_to_num(x_stats, nan=0.0, posinf=0.0, neginf=0.0)
                    
            except Exception as e:
                pass
        
        return x_rd, x_track, x_stats, label


# 兼容性别名
FusionDataLoader = FusionDataLoaderV4
FusionDataLoaderV3 = FusionDataLoaderV4