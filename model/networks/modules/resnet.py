import torch
from torch import nn
from torchvision.models.resnet import ResNet, Bottleneck


class ResNetModel(nn.Module):
    def __init__(self, num_blocks, in_channel=3):
        super(ResNetModel, self).__init__()
        resnet = ResNet(Bottleneck, num_blocks)

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.grid_avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.layer4(x)

        x1 = self.avg_pool(x1)

        return self.flatten(x1), self.grid_avg_pool(x)
