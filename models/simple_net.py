import torch
import torch.nn as nn
import torchvision


class simple_net(nn.Module):
    def __init__(self):
        super(simple_net, self).__init__()
        self.resNet = torchvision.models.resnet34()
        self.l1 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.resNet(x.float())
        x = self.l1(x)
        return x

