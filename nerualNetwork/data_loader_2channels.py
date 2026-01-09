# 数据集构建 https://zhuanlan.zhihu.com/p/105507334
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import scipy.io as scio
from torchvision.transforms import transforms


class GesDataLoaderNew(Dataset):  # 需要继承data.Dataset
    def __init__(self, path_name1, path_name2, data_rows, data_cols, val=False):
        """
        :param path_name:数据集路径
        """
        self.path1 = path_name1
        self.path2 = path_name2
        self.data_rows = data_rows
        self.data_cols = data_cols
        self.label = os.listdir(path_name1)  # 获得文件夹名字
        self.label_and_data_file_name1 = []  # 创建一个保存标签与数据文件名的list
        self.label_and_data_file_name2 = []  # 创建一个保存标签与数据文件名的list
        for i in self.label:
            path1_ = os.path.join(path_name1, i)  # 打开每个标签文件夹
            file_name_list = os.listdir(path1_)  # 获得标签文件夹内所有数据文件名
            for j in file_name_list:
                self.label_and_data_file_name1.append((i, j))  # 把标签和对应的文件名的存在list中

            path2_ = os.path.join(path_name2, i)  # 打开每个标签文件夹
            file_name_list = os.listdir(path2_)  # 获得标签文件夹内所有数据文件名
            for k in file_name_list:
                self.label_and_data_file_name2.append((i, k))  # 把标签和对应的文件名的存在list中

        self.file_number = len(self.label_and_data_file_name1)
        if val == False:
            print("训练数据集数量:", self.file_number)
        else:
            print("验证数据集数量:", self.file_number)

    def __getitem__(self, index):
        _data = np.zeros((2, self.data_rows, self.data_cols))
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data
        file_name1 = self.label_and_data_file_name1[index][1]
        file_name2 = self.label_and_data_file_name2[index][1]
        its_label = self.label_and_data_file_name1[index][0]
        file_path1 = os.path.join(self.path1, its_label, file_name1)
        file_path2 = os.path.join(self.path2, its_label, file_name2)

        original_data1 = scio.loadmat(file_path1)
        data01 = original_data1['data']
        data11 = np.array(data01)

        curr_rows = data11.shape[0]
        if curr_rows > self.data_rows:
            if False and curr_rows == 181:
                data21 = data11[58:122, :]
            else:
                data21 = data11[:self.data_rows, :]
        else:
            num_pad = self.data_rows - data11.shape[0]
            data21 = np.pad(data11, ((0, num_pad), (0, 0)), 'constant', constant_values=(0, 0))

        curr_cols = data11.shape[1]
        if curr_cols > self.data_cols:
            data21 = data21[:, :self.data_cols]
        else:
            num_pad = self.data_cols - data11.shape[1]
            data21 = np.pad(data21, ((0, 0), (0, num_pad)), 'constant', constant_values=(0, 0))

        _data11 = data21.reshape((1, self.data_rows, self.data_cols))
        _data11 = 20 * np.log(_data11 + 1e-7)

        _data[0, :, :] = _data11

        original_data2 = scio.loadmat(file_path2)
        data02 = original_data2['data']
        data12 = np.array(data02)

        curr_rows = data12.shape[0]
        if curr_rows > self.data_rows:
            if curr_rows == 181:
                data22 = data12[58:122, :]
            else:
                data22 = data12[:self.data_rows, :]
        else:
            num_pad = self.data_rows - data12.shape[0]
            data22 = np.pad(data12, ((0, num_pad), (0, 0)), 'constant', constant_values=(0, 0))

        curr_cols = data12.shape[1]
        if curr_cols > self.data_cols:
            data22 = data22[:, :self.data_cols]
        else:
            num_pad = self.data_cols - data12.shape[1]
            data22 = np.pad(data22, ((0, 0), (0, num_pad)), 'constant', constant_values=(0, 0))

        _data12 = data22.reshape((1, self.data_rows, self.data_cols))
        _data12 = 20 * np.log(_data12 + 1e-7)

        _data[1, :, :] = _data12

        return _data, int(its_label), file_name1

    def __len__(self):
        #:return: 返回数据集长度

        return self.file_number


if __name__ == "__main__":
    path = "./data/train"

    dataset = GesDataLoaderNew(path, data_rows=128, data_cols=128)  # 加载数据集

    train_dataloder = DataLoader(dataset, batch_size=64,
                                 num_workers=0, drop_last=True, shuffle=True)

    for index, data in enumerate(train_dataloder):
        print(data)
