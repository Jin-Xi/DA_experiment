import random
import os
import shutil
from tqdm import tqdm

def cifar_partition(root, dst, rate = 0.1):
    root = root.strip("/") + "/"
    data_list = os.listdir(root)
    total_len = len(data_list)
    data_dict = {}
    sampled_data = []
    for i in range(10):
        data_dict[i] = [data for data in data_list if str(i)+"_" in data]

    for i in range(10):
        file_len = len(data_dict[i])
        # print(file_len)
        pick_number = int(file_len * rate)
        #从列表中sample出pick_number个元素
        sampled_data += random.sample(data_dict[i], pick_number)

    sample_data_bar = tqdm(sampled_data)
    for step, data in enumerate(sample_data_bar):
        shutil.move(root+data, dst)
        sample_data_bar.desc = "moving {} to {}".format(data, dst)

    print("done!")

if __name__ == "__main__":
    cifar_partition("../datasets/cifar-10/test", "../datasets/cifar-10/val")
