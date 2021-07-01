import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import cifar10

class cifar10_kfold():
    """
    :param val_index : 用作验证集的折数的地址
    :param root : 数据的地址
    """

    def __init__(self, root, fold_times, val_index):
        super(cifar10_kfold, self).__init__()
        self.root = root.strip("/") + "/"
        self.img_list = os.listdir(root)
        self.fold_len = int(len(self.img_list) / fold_times)


    def __len__(self):
        return len(self.img_list)

    def __iter__(self, index):
        # img_name = self.img_list[index]
        # img_path = self.root + img_name
        # img = np.array(Image.open(img_path).convert('RGB'))
        # img = torch.tensor(img)
        # img = img.permute([2, 1, 0])
        # label = int(img_name[0])
        train_data = cifar10(self.file_list_train)
        val_data = cifar10(self.file_list_test)

        return train_data, val_data

