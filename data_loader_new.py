import torch
from torch.utils.data import Dataset
import os
import numpy as np
import scipy.io as scio
from pathlib import Path
import random

class GesDataLoaderNew(Dataset):
    def __init__(self, path_name, data_rows, data_cols, val=False, test_mode=False):
        """
        :param path_name: 数据集路径
        :param data_rows: 数据行数
        :param data_cols: 数据列数
        :param val: 是否为验证模式
        :param test_mode: 是否为测试模式
        """
        self.path = path_name
        self.data_rows = data_rows
        self.data_cols = data_cols
        self.val = val          # [关键] 保存 val 状态供 __getitem__ 使用
        self.test_mode = test_mode
        self.label_and_data_file_name = []

        if self.test_mode:
            # 测试模式：递归查找所有.mat文件
            self._load_test_data()
        else:
            # 训练/验证模式：按原来的逻辑加载
            self._load_train_val_data()

        self.file_number = len(self.label_and_data_file_name)

        if not val and not test_mode:
            print("训练数据集数量:", self.file_number)
        elif val and not test_mode:
            print("验证数据集数量:", self.file_number)
        else:
            print("测试数据集数量:", self.file_number)

    def _load_train_val_data(self):
        """加载训练/验证数据"""
        self.label = os.listdir(self.path)
        for i in self.label:
            path_ = os.path.join(self.path, i)
            if os.path.isdir(path_):
                file_name_list = os.listdir(path_)
                for j in file_name_list:
                    if j.endswith('.mat'):
                        self.label_and_data_file_name.append((i, j))

    def _load_test_data(self):
        """加载测试数据"""
        path_obj = Path(self.path)
        for mat_file in path_obj.rglob('*.mat'):
            label = 0 # 测试集默认标签
            relative_path = mat_file.relative_to(path_obj)
            parent_dir = str(relative_path.parent) if relative_path.parent != Path('.') else ''
            self.label_and_data_file_name.append((str(label), mat_file.name, parent_dir))

    def _augment(self, data):
        """频谱增强：高斯噪声 + SpecAugment"""
        # 1. 随机高斯噪声
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.01, data.shape)
            data = data + noise
            
        # 2. SpecAugment (时间维掩膜)
        if random.random() < 0.5:
            t0 = random.randint(0, self.data_rows - 5)
            data[:, t0:t0+5, :] = 0
        
        # 3. SpecAugment (频率维掩膜)
        if random.random() < 0.5:
            f0 = random.randint(0, self.data_cols - 5)
            data[:, :, f0:f0+5] = 0
            
        return data

    def __getitem__(self, index):
        if self.test_mode and len(self.label_and_data_file_name[index]) == 3:
            its_label, file_name, sub_dir = self.label_and_data_file_name[index]
            file_path = os.path.join(self.path, sub_dir, file_name) if sub_dir else os.path.join(self.path, file_name)
        else:
            its_label, file_name = self.label_and_data_file_name[index]
            file_path = os.path.join(self.path, its_label, file_name)

        try:
            # 加载 .mat 文件
            original_data = scio.loadmat(file_path)

            if 'data' in original_data:
                data0 = original_data['data']
            elif 'rangePower1' in original_data:
                data0 = original_data['rangePower1']
            else:
                data_keys = [k for k in original_data.keys() if not k.startswith('__')]
                if data_keys:
                    data0 = original_data[data_keys[0]]
                else:
                    raise ValueError(f"无法在文件 {file_name} 中找到数据")

            data1 = np.array(data0)

            # 维度裁剪/填充
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

            # === [核心修改：改为 Z-Score 标准化] ===
            # 1. 基础处理
            data2 = np.abs(data2) + 1e-7
            
            # 2. 对数变换
            data_log = 20 * np.log(data2)

            # 3. Z-Score 标准化 (关键修复！)
            # 不依赖固定的最大最小值，而是根据当前样本的统计特性进行归一化
            # 效果：让数据均值为0，标准差为1
            mean_val = np.mean(data_log)
            std_val = np.std(data_log)
            data_norm = (data_log - mean_val) / (std_val + 1e-6)

            # 4. 维度调整为 (1, H, W)
            _data = data_norm.reshape((1, self.data_rows, self.data_cols))

            # 5. 数据增强 (仅训练集)
            if not self.val and not self.test_mode:
                _data = self._augment(_data)

            return _data, int(its_label), file_name

        except Exception as e:
            print(f"加载错误 {file_path}: {e}")
            return np.zeros((1, self.data_rows, self.data_cols)), 0, file_name

    def __len__(self):
        # [关键] DataLoader 必须依赖这个方法来获取数据集大小
        return self.file_number