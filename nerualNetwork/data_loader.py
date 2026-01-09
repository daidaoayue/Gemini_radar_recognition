# 数据集构建 https://zhuanlan.zhihu.com/p/105507334
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import scipy.io as scio
from torchvision.transforms import transforms


class GesDataLoader(Dataset):  # 需要继承data.Dataset
    def __init__(self, path_name, data_rows, data_cols, val=False):
        """
        :param path_name:数据集路径
        """
        self.path = path_name
        self.data_rows = data_rows
        self.data_cols = data_cols
        self.label = os.listdir(path_name)  # 获得文件夹名字
        self.label_and_data_file_name = []  # 创建一个保存标签与数据文件名的list
        for i in self.label:
            path_ = os.path.join(path_name, i)  # 打开每个标签文件夹
            file_name_list = os.listdir(path_)  # 获得标签文件夹内所有数据文件名
            for j in file_name_list:
                self.label_and_data_file_name.append((i, j))  # 把标签和对应的文件名的存在list中

        self.file_number = len(self.label_and_data_file_name)  # 返回数据集数量
        if val == False:
            print("训练数据集数量:", self.file_number)
        else:
            print("验证数据集数量:", self.file_number)

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data
        file_name = self.label_and_data_file_name[index][1]
        its_label = self.label_and_data_file_name[index][0]
        file_path = os.path.join(self.path, its_label, file_name)

        original_data = scio.loadmat(file_path)
        data0 = original_data['data']
        data1 = np.array(data0)
        num_pad = self.data_cols - data1.shape[1]
        data2 = np.pad(data1, ((0, 0), (0, num_pad)), 'constant', constant_values=(0, 0))
        _data = data2.reshape((1, self.data_rows, self.data_cols))

        return _data, int(its_label), file_name

    def __len__(self):
        #:return: 返回数据集长度

        return self.file_number


if __name__ == "__main__":
    path = "./data/train"

    dataset = GesDataLoader(path, data_rows=128, data_cols=200)  # 加载数据集

    train_dataloder = DataLoader(dataset, batch_size=64,
                                 num_workers=0, drop_last=True, shuffle=True)

    for index, data in enumerate(train_dataloder):
        print(data)