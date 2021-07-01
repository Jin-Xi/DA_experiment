import os
import cifar10

class cifar10_kfold():
    """
    :param val_index : 用作验证集的折数的地址
    :param root : 数据的地址
    """

    def __init__(self, root, fold_times):
        super(cifar10_kfold, self).__init__()
        self.root = root.strip("/") + "/"
        self.img_list = os.listdir(root)
        self.per_fold_len = int(len(self.img_list) / fold_times)
        self.fold_dict = {}
        self.fold_times = fold_times
        self.split_data(fold_times)


    def __len__(self):
        return len(self.img_list)

    def get_data(self, index):
        train = []
        val = []
        for i in range(self.fold_times):
            if i == index:
                val = self.fold_dict[i]
            train += self.fold_dict[i]

        train_data = cifar10(train)
        val_data = cifar10(val)
        return train_data, val_data

    def split_data(self, k):
        for i in range(k-1):
            start = i*self.per_fold_len
            end = (i+1)*self.per_fold_len
            self.fold_dict[i] = self.img_list[start:end]
        self.fold_dict[k-1] = self.img_list[(k-1)*self.per_fold_len:]
