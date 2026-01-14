# 修改后的data_loader_new.py

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import scipy.io as scio
from torchvision.transforms import transforms
import re
from pathlib import Path


class GesDataLoaderNew(Dataset):
    def __init__(self, path_name, data_rows, data_cols, val=False, test_mode=False):
        """
        :param path_name: 数据集路径
        :param data_rows: 数据行数
        :param data_cols: 数据列数
        :param val: 是否为验证模式
        :param test_mode: 是否为测试模式（新增参数）
        """
        self.path = path_name
        self.data_rows = data_rows
        self.data_cols = data_cols
        self.test_mode = test_mode
        self.label_and_data_file_name = []

        if self.test_mode:
            # 测试模式：递归查找所有.mat文件，标签设为占位符
            self._load_test_data()
        else:
            # 训练/验证模式：按原来的逻辑加载
            self._load_train_val_data()

        self.file_number = len(self.label_and_data_file_name)

        if val == False and not test_mode:
            print("训练数据集数量:", self.file_number)
        elif val == True and not test_mode:
            print("验证数据集数量:", self.file_number)
        else:
            print("测试数据集数量:", self.file_number)

    def _load_train_val_data(self):
        """加载训练/验证数据（原逻辑）"""
        self.label = os.listdir(self.path)
        for i in self.label:
            path_ = os.path.join(self.path, i)
            if os.path.isdir(path_):  # 确保是目录
                file_name_list = os.listdir(path_)
                for j in file_name_list:
                    if j.endswith('.mat'):  # 只处理.mat文件
                        self.label_and_data_file_name.append((i, j))

    def _load_test_data(self):
        """加载测试数据（新逻辑）"""
        path_obj = Path(self.path)

        # 递归查找所有.mat文件
        for mat_file in path_obj.rglob('*.mat'):
            # 从文件名推断标签，或者设置默认标签
            label = self._extract_label_from_filename(mat_file.name)

            # 计算相对路径
            relative_path = mat_file.relative_to(path_obj)
            parent_dir = str(relative_path.parent) if relative_path.parent != Path('.') else ''

            # 存储 (标签, 文件名, 子目录路径)
            self.label_and_data_file_name.append((str(label), mat_file.name, parent_dir))

    def _extract_label_from_filename(self, filename):
        """
        从文件名推断标签
        你需要根据实际情况修改这个函数
        """
        # 方法1：如果所有测试文件都是同一类，返回固定标签
        return 0  # 默认标签

        # 方法2：如果文件名包含标签信息
        # 例如：如果文件名格式是 "class1_data.mat"
        # match = re.search(r'class(\d+)', filename.lower())
        # if match:
        #     return int(match.group(1))

        # 方法3：如果有其他命名规则
        # 根据你的实际文件命名规则来实现

        # 方法4：如果需要从文件内容读取标签
        # 这需要在__getitem__中处理

    def __getitem__(self, index):
        if self.test_mode and len(self.label_and_data_file_name[index]) == 3:
            # 测试模式：(标签, 文件名, 子目录路径)
            its_label, file_name, sub_dir = self.label_and_data_file_name[index]
            file_path = os.path.join(self.path, sub_dir, file_name) if sub_dir else os.path.join(self.path, file_name)
        else:
            # 训练/验证模式：(标签, 文件名)
            its_label, file_name = self.label_and_data_file_name[index]
            file_path = os.path.join(self.path, its_label, file_name)

        try:
            # 加载.mat文件
            original_data = scio.loadmat(file_path)

            # 尝试不同的数据键
            if 'data' in original_data:
                data0 = original_data['data']
            elif 'rangePower1' in original_data:
                data0 = original_data['rangePower1']
            else:
                # 自动找到数据键
                data_keys = [k for k in original_data.keys() if not k.startswith('__')]
                if data_keys:
                    data0 = original_data[data_keys[0]]
                    print(f"使用数据键: {data_keys[0]} 来自文件: {file_name}")
                else:
                    raise ValueError(f"无法在文件 {file_name} 中找到数据")

            data1 = np.array(data0)

            # 处理数据维度
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

            # 数据预处理
            data2 = np.abs(data2) + 1e-7
            _data = data2.reshape((1, self.data_rows, self.data_cols))
            _data = 20 * np.log(_data + 1e-7)

            return _data, int(its_label), file_name

        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
            # 返回零数据和默认标签
            _data = np.zeros((1, self.data_rows, self.data_cols))
            return _data, 0, file_name

    def __len__(self):
        return self.file_number


# 专门用于测试的数据加载器
class TestDataLoader(Dataset):
    """
    专门用于测试数据的加载器，自动处理不同的目录结构
    """

    def __init__(self, path_name, data_rows=32, data_cols=64):
        self.path = path_name
        self.data_rows = data_rows
        self.data_cols = data_cols
        self.data_files = []

        # 收集所有.mat文件
        path_obj = Path(path_name)

        if not path_obj.exists():
            raise ValueError(f"路径不存在: {path_name}")

        # 递归查找所有.mat文件
        mat_files = list(path_obj.rglob('*.mat'))

        if not mat_files:
            raise ValueError(f"在 {path_name} 中未找到.mat文件")

        for mat_file in mat_files:
            # 从文件名或其他信息推断标签
            label = self._infer_label(mat_file)
            self.data_files.append((mat_file, label))

        print(f"测试数据集数量: {len(self.data_files)}")

        # 显示标签分布
        labels = [label for _, label in self.data_files]
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"标签分布: {dict(zip(unique_labels, counts))}")

    def _infer_label(self, mat_file):
        """
        推断文件的真实标签
        你需要根据实际情况修改这个函数
        """
        filename = mat_file.name

        # 方法1：所有测试文件都是同一类
        return 0

        # 方法2：从文件名推断
        # 例如：Track1 -> 类别0, Track2 -> 类别1 等
        # if 'Track1' in filename:
        #     return 0
        # elif 'Track2' in filename:
        #     return 1
        # # ... 更多规则

        # 方法3：从父目录名推断（如果有意义的话）
        # parent_name = mat_file.parent.name
        # if parent_name.isdigit():
        #     return int(parent_name)

        # 方法4：从文件内容推断（需要在__getitem__中实现）

        # 默认返回0
        return 0

    def __getitem__(self, index):
        mat_file, label = self.data_files[index]

        try:
            # 加载数据
            original_data = scio.loadmat(str(mat_file))

            # 自动找到数据
            if 'data' in original_data:
                data0 = original_data['data']
            else:
                data_keys = [k for k in original_data.keys() if not k.startswith('__')]
                if data_keys:
                    data0 = original_data[data_keys[0]]
                else:
                    raise ValueError("无法找到数据")

            data1 = np.array(data0)

            # 数据预处理（与原来相同）
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
            _data = data2.reshape((1, self.data_rows, self.data_cols))
            _data = 20 * np.log(_data + 1e-7)

            return _data, label, mat_file.name

        except Exception as e:
            print(f"加载文件 {mat_file} 时出错: {e}")
            _data = np.zeros((1, self.data_rows, self.data_cols))
            return _data, 0, mat_file.name

    def __len__(self):
        return len(self.data_files)


# 使用示例
if __name__ == "__main__":
    # 训练/验证模式（原来的方式）
    # path = "./dataset/data0507-0/val"
    # dataset = GesDataLoaderNew(path, data_rows=128, data_cols=128, val=True)

    # 测试模式（新方式）
    test_path = "./dataset/preprocess_data/test"

    # 方法1：使用修改后的原加载器
    test_dataset1 = GesDataLoaderNew(test_path, data_rows=32, data_cols=64, val=True, test_mode=True)

    # 方法2：使用专门的测试加载器
    test_dataset2 = TestDataLoader(test_path, data_rows=32, data_cols=64)

    # 测试
    train_dataloader = DataLoader(test_dataset2, batch_size=1, num_workers=0, drop_last=False, shuffle=False)

    for index, (data, label, filename) in enumerate(train_dataloader):
        print(f"样本 {index}: 形状={data.shape}, 标签={label.item()}, 文件={filename[0]}")
        if index >= 2:  # 只显示前3个样本
            break