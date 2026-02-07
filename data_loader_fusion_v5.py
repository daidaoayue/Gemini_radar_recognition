"""
V16: 微多普勒特征融合数据加载器
================================
在V14基础上增加12维微多普勒特征

特征维度:
  - RD图像: [1, 32, 64]
  - 航迹时序: [12, 16]
  - 航迹统计: [20]
  - 微多普勒: [12]  ← 新增

总统计特征: 20 + 12 = 32维
"""

import os
import torch
import scipy.io as scio
import numpy as np
from torch.utils.data import Dataset
import random


class FusionDataLoaderV5(Dataset):
    """
    V16专用数据加载器
    整合RD + 航迹 + 微多普勒特征
    """
    
    def __init__(self, rd_root_dir, track_feat_dir, micro_doppler_dir=None, val=False):
        """
        Args:
            rd_root_dir: RD数据目录
            track_feat_dir: 航迹特征目录
            micro_doppler_dir: 微多普勒特征目录（可选）
            val: 是否为验证模式
        """
        self.rd_root_dir = rd_root_dir
        self.track_feat_dir = track_feat_dir
        self.micro_doppler_dir = micro_doppler_dir
        self.val = val
        
        self.data_rows = 32
        self.data_cols = 64
        self.stats_dim = 20  # 原始统计特征维度
        self.micro_doppler_dim = 12  # 微多普勒特征维度
        
        # 扫描样本
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
                
                # 航迹特征目录
                track_label_dir = None
                if track_feat_dir and os.path.exists(track_feat_dir):
                    potential = os.path.join(track_feat_dir, label_str)
                    if os.path.exists(potential):
                        track_label_dir = potential
                
                # 微多普勒特征目录
                md_label_dir = None
                if micro_doppler_dir and os.path.exists(micro_doppler_dir):
                    potential = os.path.join(micro_doppler_dir, label_str)
                    if os.path.exists(potential):
                        md_label_dir = potential
                
                # 遍历RD文件
                rd_files = [f for f in os.listdir(rd_label_dir) 
                           if f.endswith('.mat') and '_track' not in f and '_motion' not in f and '_microdoppler' not in f]
                
                for rd_file in rd_files:
                    rd_path = os.path.join(rd_label_dir, rd_file)
                    base_name = rd_file.replace('.mat', '')
                    
                    # 查找航迹特征
                    track_path = None
                    if track_label_dir:
                        for suffix in ['_track.mat', '.mat']:
                            potential = os.path.join(track_label_dir, base_name + suffix)
                            if os.path.exists(potential):
                                track_path = potential
                                break
                    
                    # 查找微多普勒特征
                    md_path = None
                    if md_label_dir:
                        potential = os.path.join(md_label_dir, base_name + '_microdoppler.mat')
                        if os.path.exists(potential):
                            md_path = potential
                    
                    self.samples.append((label, rd_path, track_path, md_path))
        
        # 统计
        track_matched = sum(1 for s in self.samples if s[2] is not None)
        md_matched = sum(1 for s in self.samples if s[3] is not None)
        total = len(self.samples)
        
        print(f"[{'验证' if val else '训练'}] 样本数: {total}")
        print(f"   -> 航迹匹配: {track_matched} ({100*track_matched/max(total,1):.1f}%)")
        print(f"   -> 微多普勒匹配: {md_matched} ({100*md_matched/max(total,1):.1f}%)")
        print(f"   -> 总特征维度: {self.stats_dim + self.micro_doppler_dim}")

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
        label, rd_path, track_path, md_path = self.samples[idx]
        
        # ========== A. 加载RD ==========
        try:
            mat = scio.loadmat(rd_path)
            
            if 'data' in mat:
                rd = mat['data']
            elif 'rangePower1' in mat:
                rd = mat['rangePower1']
            else:
                keys = [k for k in mat.keys() if not k.startswith('__')]
                rd = mat[keys[0]] if keys else np.zeros((32, 64))
            
            rd = np.array(rd)
            
            # 裁剪/填充
            if rd.shape[0] > self.data_rows:
                rd = rd[:self.data_rows, :]
            else:
                rd = np.pad(rd, ((0, self.data_rows - rd.shape[0]), (0, 0)))
            
            if rd.shape[1] > self.data_cols:
                rd = rd[:, :self.data_cols]
            else:
                rd = np.pad(rd, ((0, 0), (0, self.data_cols - rd.shape[1])))
            
            # 预处理
            rd = np.abs(rd) + 1e-7
            rd = 20 * np.log(rd)
            rd = (rd - rd.mean()) / (rd.std() + 1e-6)
            rd = rd.reshape(1, self.data_rows, self.data_cols)
            
            if not self.val:
                rd = self._augment(rd)
            
            x_rd = torch.from_numpy(rd).float()
            
        except:
            x_rd = torch.zeros(1, self.data_rows, self.data_cols)
        
        # ========== B. 加载航迹特征 ==========
        x_track = torch.zeros(12, 16)
        x_stats = torch.zeros(self.stats_dim)
        
        if track_path and os.path.exists(track_path):
            try:
                mat = scio.loadmat(track_path)
                
                if 'track_data' in mat:
                    feat = mat['track_data'].astype(np.float32)
                    if feat.shape[0] == 16 and feat.shape[1] == 12:
                        feat = feat.T
                    n_ch = min(feat.shape[0], 12)
                    n_len = min(feat.shape[1], 16)
                    x_track[:n_ch, :n_len] = torch.from_numpy(feat[:n_ch, :n_len].copy())
                
                if 'track_stats' in mat:
                    stats = mat['track_stats'].flatten()[:self.stats_dim]
                    x_stats[:len(stats)] = torch.from_numpy(stats.astype(np.float32).copy())
            except:
                pass
        
        # ========== C. 加载微多普勒特征 ==========
        x_micro_doppler = torch.zeros(self.micro_doppler_dim)
        
        if md_path and os.path.exists(md_path):
            try:
                mat = scio.loadmat(md_path)
                
                if 'micro_doppler_features' in mat:
                    md_feat = mat['micro_doppler_features'].flatten()[:self.micro_doppler_dim]
                    x_micro_doppler[:len(md_feat)] = torch.from_numpy(md_feat.astype(np.float32).copy())
            except:
                pass
        
        # ========== D. 合并统计特征 ==========
        # 原始20维 + 微多普勒12维 = 32维
        x_combined_stats = torch.cat([x_stats, x_micro_doppler], dim=0)
        
        return x_rd, x_track, x_combined_stats, label


# 为了测试是否有微多普勒特征
def check_micro_doppler_coverage(rd_dir, md_dir):
    """检查微多普勒特征覆盖率"""
    total = 0
    matched = 0
    
    for label in range(6):
        rd_label_dir = os.path.join(rd_dir, str(label))
        md_label_dir = os.path.join(md_dir, str(label))
        
        if not os.path.exists(rd_label_dir):
            continue
        
        rd_files = [f for f in os.listdir(rd_label_dir) 
                   if f.endswith('.mat') and '_track' not in f and '_microdoppler' not in f]
        
        for f in rd_files:
            total += 1
            base_name = f.replace('.mat', '')
            md_file = os.path.join(md_label_dir, base_name + '_microdoppler.mat')
            if os.path.exists(md_file):
                matched += 1
    
    print(f"微多普勒特征覆盖率: {matched}/{total} ({100*matched/max(total,1):.1f}%)")
    return matched, total


if __name__ == '__main__':
    # 测试
    print("测试数据加载器...")
    
    # 检查覆盖率
    check_micro_doppler_coverage(
        "./dataset/train_cleandata/val",
        "./dataset/micro_doppler_features/val"
    )