# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 第一卷积层：输入1通道（灰度图像），输出6通道，卷积核大小为5x5
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        # 第一池化层：最大池化，池化窗口大小为2x2，步幅为2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二卷积层：输入6通道，输出16通道，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 第二池化层：最大池化，池化窗口大小为2x2，步幅为2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第一个全连接层：输入维度是16*5*5，输出维度是120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 第二个全连接层：输入维度是120，输出维度是84
        self.fc2 = nn.Linear(120, 84)
        # 第三个全连接层：输入维度是84，输出维度是10，对应10个类别
        self.fc3 = nn.Linear(84, 10)
        self._initialize_weights()

    def forward(self, x):
        # 前向传播函数定义网络的数据流向
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)