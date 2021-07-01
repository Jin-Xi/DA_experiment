import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os


class cifar10(Dataset):
    def __init__(self, root=None, file_list=[]):
        super(cifar10, self).__init__()
        self.root = root.strip("/") + "/"
        if root != None:
            self.img_list = os.listdir(root)
        elif len(file_list) == 0:
            raise IndexError("root未指定，或者file_list为空 ")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_path = self.root + img_name
        img = np.array(Image.open(img_path).convert('RGB'))
        img = torch.tensor(img)
        img = img.permute([2, 1, 0])
        label = int(img_name[0])
        return img, label
