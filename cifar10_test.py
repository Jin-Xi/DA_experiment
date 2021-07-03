import os.path
import collections
import torch.cuda
from torch.utils.data import DataLoader
from datasets.cifar10 import cifar10
from models.simple_net import simple_net
from models.resnet import resnet32

from tqdm import tqdm
import time


def transform_weight_dict(th_dict):
    params = collections.OrderedDict()
    for k in th_dict.items():
        if 'module' in k[0]:
            key = k[0].replace('module.', '')
        params[key] = k[1]
    return params


epochs = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    test_data = cifar10("./datasets/cifar-10/test/")
    test_loader = DataLoader(test_data, batch_size=100, shuffle=True)
    test_num = len(test_data)

    net = resnet32()
    net = net.to(device)
    if os.path.isfile("./saved_model/resnet_cifar_latest.pth"):
        net.load_state_dict(torch.load("./saved_model/resnet_cifar_latest.pth"))

    for epoch in range(epochs):
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            test_bar = tqdm(test_loader)
            for val_data in test_bar:
                val_images, test_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)
                predict_y = predict_y[1]
                acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

                val_accurate = acc / test_num
                test_bar.desc = "valid epoch[{}/{}] val_acc:{:.4}".format(epoch + 1,
                                                                          epochs,
                                                                          val_accurate)

    # tm = time.localtime(time.time())
    # torch.save(net.state_dict(), "./saved_model/resnet_cifar_latest" + ".pth")
    # torch.save(net.state_dict(), "./saved_model/resnet_cifar" + str(tm.tm_hour) + "_" + str(tm.tm_min) + ".pth")
