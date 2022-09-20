
import numpy
import torch
from torch import nn
from torch.nn import ReLU


PRECISION = 1e-15


class MinMax(nn.Module):
    def __init__(self):
        super(MinMax, self).__init__()

    def forward(self, p):
        lower = (p > PRECISION).float()
        upper = (p < (1-PRECISION)).float()
        return p * lower * upper + PRECISION * (1-lower) + (1-PRECISION) * (1-upper)


class WeightedMultiClassLogLoss(nn.Module):
    def __init__(self):
        super(WeightedMultiClassLogLoss, self).__init__()
        self.mm = MinMax()
        # self.num = torch.tensor([[457, 175]])

    def forward(self, prob, target, n=(457, 175)):
        # print(prob.shape, target.shape)
        ce = torch.dot(torch.log(self.mm(prob[:, 0])) / n[0], target.float()) * 0.5
        laa = torch.dot(torch.log(self.mm(prob[:, 1])) / n[1], 1-target.float()) * 0.5
        return -ce - laa


if __name__ == "__main__":
    a = nn.functional.sigmoid(torch.rand(5, 2))
    t = torch.zeros(5, 2)
    t[[0, 1, 2, 4], 0] = 1
    t[3, 1] = 1
    loss = WeightedMultiClassLogLoss()
    print(loss(a, t))

