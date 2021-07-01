import numpy as np
import os
import cv2


# 解压 返回解压后的字典
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == "__main__":
    # 生成训练集图片
    for j in range(1, 6):
        dataName = "../datasets/cifar-10-batches-py/data_batch_" + str(j)  # 读取当前目录下的data_batch1~5文件。
        Xtr = unpickle(dataName)
        print(dataName + " is loading...")

        for i in range(0, 10000):
            img = np.reshape(Xtr[b'data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
            img = img.transpose(1, 2, 0)  # 读取image
            picName = '../datasets/cifar-10/train/' + str(Xtr[b'labels'][i]) + '_' + str(i + (j - 1) * 10000) + '.jpg'
            # Xtr['labels']为图片的标签，值范围0-9，本文中，train文件夹需要存在，并与脚本文件在同一目录下。
            cv2.imwrite(picName, img)
        print(dataName + " loaded.")

    print("test_batch is loading...")

    # 生成测试集图片
    testXtr = unpickle("../datasets/cifar-10-batches-py/test_batch")
    for i in range(0, 10000):
        img = np.reshape(testXtr[b'data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        picName = '../datasets/cifar-10/test/' + str(testXtr[b'labels'][i]) + '_' + str(i) + '.jpg'
        cv2.imwrite(picName, img)
    print("test_batch loaded.")
