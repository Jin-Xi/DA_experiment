import os.path

import torch
import torch.cuda
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from datasets.cifar10 import cifar10
from models.simple_net import simple_net
from models.resnet import resnet32



from tqdm import tqdm
import time

first_train = False
epochs = 10
best_acc = 0.0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    train_data = cifar10("./datasets/cifar-10/train/")
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    train_num = len(train_data)

    val_data = cifar10("./datasets/cifar-10/val")
    val_loader = DataLoader(val_data, batch_size=1000, shuffle=True)
    val_num = len(val_data)

    net = resnet32()
    net = net.to(device)
    loss_function = torch.nn.CrossEntropyLoss()

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=1, patience=3)

    if os.path.isfile("./saved_model/resnet_cifar_latest.pth") and ~first_train:
        net.load_state_dict(torch.load("./saved_model/resnet_cifar_latest.pth"))

    for epoch in range(epochs):
        acc = 0
        net.train()
        train_bar = tqdm(train_loader)
        for imgs, labels in train_bar:
            optimizer.zero_grad()
            logits = net(imgs.float().to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            predict_y = torch.max(logits, dim=1)
            predict_y = predict_y[1]
            acc += torch.eq(predict_y, labels.to(device)).sum().item()
            train_accurate = acc / train_num
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} acc:{}".format(epoch + 1, epochs, loss, train_accurate)


        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                predict_y = torch.max(outputs, dim=1)
                predict_y = predict_y[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_accurate = acc / val_num
                val_bar.desc = "valid epoch[{}/{}] val_acc:{}".format(epoch + 1,
                                                                      epochs,
                                                                      val_accurate)
        # 推进学习率策略
        scheduler.step(loss)


    tm = time.localtime(time.time())
    torch.save(net.state_dict(), "./saved_model/resnet_cifar_latest" + ".pth")
    torch.save(net.state_dict(), "./saved_model/resnet_cifar" + str(tm.tm_hour) + "_" + str(tm.tm_min) + ".pth")

    # torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
    #             'optimizer': optimizer.state_dict(), 'alpha': loss.alpha, 'gamma': loss.gamma},
    #            checkpoint_path + '/m-' + launchTimestamp + '-' + str("%.4f" % lossMIN) + '.pth.tar')