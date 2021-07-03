import os
import random
from datasets.cifar10 import cifar10

class cifar10_kfold():
    """
    :param fold_times : 交叉折数
    :param root : 数据的地址
    """
    def __init__(self, root, fold_times):
        super(cifar10_kfold, self).__init__()
        self.root = root.strip("/") + "/"
        self.img_list = os.listdir(root)
        self.per_fold_len = int(len(self.img_list) / fold_times)
        self.fold_dict = {}
        self.fold_times = fold_times
        self._split_data(fold_times)


    def __len__(self):
        return len(self.img_list)

    def get_one_pair(self, index):
        train = []
        val = []
        for i in range(self.fold_times):
            if i == index:
                val = self.fold_dict[i]
                continue
            train += self.fold_dict[i]

        print(len(train))
        print(len(val))
        train_data = cifar10(file_list=train)
        val_data = cifar10(file_list=val)
        return train_data, val_data

    def get_kf_dict(self):
        kf_dict = {}
        for i in range(self.fold_times):
            # 以第i块作为验证数据分割训练集
            train_data, val_data = self.get_one_pair(i)
            kf_dict[i] = (train_data, val_data)
        return kf_dict

    def _split_data(self, k):
        random.shuffle(self.img_list)
        for i in range(k-1):
            start = i*self.per_fold_len
            end = (i+1)*self.per_fold_len
            self.fold_dict[i] = self.img_list[start:end]
        self.fold_dict[k-1] = self.img_list[(k-1)*self.per_fold_len:]
