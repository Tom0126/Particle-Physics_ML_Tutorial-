# -*- coding: utf-8 -*-
"""
# @file name  : lenet.py
# @author     : siyuansong@sjtu.edu.cn
# @date       : 2023-01-16 14:09:00
# @brief      : lenet
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self, classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()


class LeNet2(nn.Module):
    def __init__(self, classes):
        super(LeNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

class LeNet_bn(nn.Module):
    def __init__(self, classes):
        super(LeNet_bn, self).__init__()
        self.conv1 = nn.Conv2d(40, 80, 3)# 80*16*16
        self.bn1 = nn.BatchNorm2d(num_features=80)

        self.conv2 = nn.Conv2d(80, 160, 3)
        self.bn2 = nn.BatchNorm2d(num_features=160) #160*14*14

        # insert a max_pool

        self.conv3 = nn.Conv2d(160, 320, 3)
        self.bn3 = nn.BatchNorm2d(num_features=320) #320*5*5

        self.conv4 = nn.Conv2d(320, 480 , 3)
        self.bn4 = nn.BatchNorm2d(num_features=480)  # 480*3*3

        self.fc1 = nn.Linear(480 * 3 * 3, 960)
        self.bn5 = nn.BatchNorm1d(num_features=960)

        self.fc2 = nn.Linear(960, 480)
        self.bn6 = nn.BatchNorm1d(num_features=480)

        self.fc3 = nn.Linear(480, 120)
        self.fc4 = nn.Linear(120, classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out=F.max_pool2d(out,2)

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = F.relu(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.bn5(out)
        out = F.relu(out)

        out = self.fc2(out)
        out = self.bn6(out)
        out = F.relu(out)

        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 1)
                m.bias.data.zero_()





if __name__=='__main__':


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = LeNet_bn(3).to(device)
    summary(t,(40,18,18))
