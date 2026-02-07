"""
V17: 真正的微多普勒特征融合数据加载器
======================================
从原始回波提取的微多普勒特征

目录结构：
  - RD图谱: dataset/train_cleandata/train/0/Track1_Label0_Group001_Points1-4.mat
  - 微多普勒: dataset/micro_doppler_raw/Track1_Label0_microdoppler.mat

根据RD文件名中的Track ID匹配微多普勒特征
"""

import os
import torch
import scipy.io as scio
import numpy as np
from torch.utils.data import Dataset
import random
import re


class FusionDataLoaderV6(Dataset):
    """
    V17专用数据加载器
    整合RD + 航迹 + 原始回波微多普勒特征
    """
    
    def __init__(self, rd_root_dir, track_feat_dir, micro_doppler_dir=None, val=False):
        """
        Args:
            rd_root_dir: RD数据目录
            track_feat_dir: 航迹特征目录
            micro_doppler_dir: 原始回波微多普勒特征目录（按Track ID组织）
            val: 是否为验证模式
        """
        self.rd_root_dir = rd_root_dir
        self.track_feat_dir = track_feat_dir
        self.micro_doppler_dir = micro_doppler_dir
        self.val = val
        
        self.data_rows = 32
        self.data_cols = 64
        self.stats_dim = 20
        self.micro_doppler_dim = 12
        
        # 预加载所有微多普勒特征到内存（按Track ID索引）
        self.micro_doppler_cache = {}
        if micro_doppler_dir and os.path.exists(micro_doppler_dir):
            self._load_micro_doppler_cache()
        
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
                
                # 遍历RD文件
                rd_files = [f for f in os.listdir(rd_label_dir) 
                           if f.endswith('.mat') and '_track' not in f and '_motion' not in f]
                
                for rd_file in rd_files:
                    rd_path = os.path.join(rd_label_dir, rd_file)
                    base_name = rd_file.replace('.mat', '')
                    
                    # 从文件名提取Track ID
                    track_id = self._extract_track_id(rd_file)
                    
                    # 查找航迹特征
                    track_path = None
                    if track_label_dir:
                        for suffix in ['_track.mat', '.mat']:
                            potential = os.path.join(track_label_dir, base_name + suffix)
                            if os.path.exists(potential):
                                track_path = potential
                                break
                    
                    self.samples.append((label, rd_path, track_path, track_id))
        
        # 统计
        track_matched = sum(1 for s in self.samples if s[2] is not None)
        md_matched = sum(1 for s in self.samples if s[3] is not None and s[3] in self.micro_doppler_cache)
        total = len(self.samples)
        
        print(f"[{'验证' if val else '训练'}] 样本数: {total}")
        print(f"   -> 航迹匹配: {track_matched} ({100*track_matched/max(total,1):.1f}%)")
        print(f"   -> 微多普勒匹配: {md_matched} ({100*md_matched/max(total,1):.1f}%)")
        print(f"   -> 总特征维度: {self.stats_dim + self.micro_doppler_dim}")

    def _load_micro_doppler_cache(self):
        """预加载所有微多普勒特征"""
        md_files = [f for f in os.listdir(self.micro_doppler_dir) if f.endswith('.mat')]
        
        for f in md_files:
            # 从文件名提取Track ID
            # 格式: Track1_Label0_microdoppler.mat
            match = re.search(r'Track(\d+)_Label(\d+)', f)
            if match:
                track_id = int(match.group(1))
                
                try:
                    mat = scio.loadmat(os.path.join(self.micro_doppler_dir, f))
                    if 'micro_doppler_features' in mat:
                        feat = mat['micro_doppler_features'].flatten()[:self.micro_doppler_dim]
                        self.micro_doppler_cache[track_id] = feat.astype(np.float32)
                except:
                    pass
        
        print(f"   -> 预加载微多普勒特征: {len(self.micro_doppler_cache)} 个Track")

    def _extract_track_id(self, filename):
        """从RD文件名提取Track ID"""
        # 格式: Track1_Label0_Group001_Points1-4.mat
        match = re.search(r'Track(\d+)', filename)
        if match:
            return int(match.group(1))
        return None

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
        label, rd_path, track_path, track_id = self.samples[idx]
        
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
        
        # ========== C. 加载微多普勒特征（从缓存） ==========
        x_micro_doppler = torch.zeros(self.micro_doppler_dim)
        
        if track_id is not None and track_id in self.micro_doppler_cache:
            md_feat = self.micro_doppler_cache[track_id]
            x_micro_doppler[:len(md_feat)] = torch.from_numpy(md_feat.copy())
        
        # ========== D. 合并统计特征 ==========
        x_combined_stats = torch.cat([x_stats, x_micro_doppler], dim=0)
        
        return x_rd, x_track, x_combined_stats, label


def check_micro_doppler_raw_coverage(rd_dir, md_dir):
    """检查原始回波微多普勒特征覆盖率"""
    
    # 加载所有微多普勒的Track ID
    md_track_ids = set()
    if os.path.exists(md_dir):
        for f in os.listdir(md_dir):
            match = re.search(r'Track(\d+)', f)
            if match:
                md_track_ids.add(int(match.group(1)))
    
    print(f"微多普勒特征: {len(md_track_ids)} 个Track")
    
    # 检查RD样本的覆盖
    total = 0
    matched = 0
    
    for label in range(6):
        rd_label_dir = os.path.join(rd_dir, str(label))
        if not os.path.exists(rd_label_dir):
            continue
        
        for f in os.listdir(rd_label_dir):
            if not f.endswith('.mat') or '_track' in f:
                continue
            
            total += 1
            match = re.search(r'Track(\d+)', f)
            if match:
                track_id = int(match.group(1))
                if track_id in md_track_ids:
                    matched += 1
    
    print(f"RD样本覆盖率: {matched}/{total} ({100*matched/max(total,1):.1f}%)")
    return matched, total


if __name__ == '__main__':
    print("检查微多普勒特征覆盖率...\n")
    
    check_micro_doppler_raw_coverage(
        "./dataset/train_cleandata/val",
        "./dataset/micro_doppler_raw"
    )