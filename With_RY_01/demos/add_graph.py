#-*-coding:utf-8-*-
import torch
import torchvision
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch.nn as nn

class N1(nn.Module):
    def __init__(self):
        super(N1, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.Conv2d(64, 128, kernel_size=3),
        )

    def forward(self, input):
        out = self.net(input)
        return out


input_data = Variable(torch.rand(16, 1, 225, 224))

net = N1()

print(type(net), isinstance(net, nn.Module))

writer = SummaryWriter(log_dir='./log', comment='resnet18')
with writer:
    writer.add_graph(net, (input_data,))