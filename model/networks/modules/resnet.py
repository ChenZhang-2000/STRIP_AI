from torch import nn


class ResBlock(nn.Module):
    def __init__(self, channel=64):
        super(ResBlock, self).__init__()
        self.channel = channel

        self.res = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(self.channel, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(self.channel, affine=True)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.res(x)
        out += x  # self.shortcut(x)
        out = self.act(out)
        return out


class ResStrideBlock(nn.Module):

    def __init__(self, in_channel=64, out_channel=128):
        super(ResStrideBlock, self).__init__()
        self.in_dim = in_channel
        self.out_dim = out_channel

        self.res = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.out_dim, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(self.out_dim, affine=True)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1,
                      stride=2, bias=False),
            nn.InstanceNorm2d(self.out_dim, affine=True))
        self.act = nn.LeakyReLU()

    def forward(self, x):
        out = self.res(x)
        out += self.shortcut(x)
        out = self.act(out)
        return out


class BottleNeck(nn.Module):
    def __init__(self, channel):
        super(BottleNeck, self).__init__()

        self.channel = channel

        self.main = nn.Sequential(
            nn.Conv2d(4*channel, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channel, affine=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channel, affine=True),
            nn.ReLU(),
            nn.Conv2d(channel, 4*channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(4*channel, affine=True),
        )

        self.act = nn.ReLU()

    def forward(self, x):
        out = self.main(x)
        out += x
        return self.act(out)


class BottleNeckStride(nn.Module):
    def __init__(self, in_channel):
        super(BottleNeckStride, self).__init__()

        channel = in_channel

        self.main = nn.Sequential(
            nn.Conv2d(4*channel, 2*channel, kernel_size=1, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(2*channel, affine=True),
            nn.ReLU(),
            nn.Conv2d(2*channel, 2*channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(2*channel, affine=True),
            nn.ReLU(),
            nn.Conv2d(2*channel, 8*channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(8*channel, affine=True),
        )

        self.identity = nn.Sequential(
            nn.Conv2d(4*channel, 8*channel, kernel_size=1, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8*channel, affine=True),
        )

        self.act = nn.ReLU()

    def forward(self, x):
        out = self.main(x)
        out += self.identity(x)
        return self.act(out)


class InitBottleNeck(nn.Module):
    def __init__(self, channel):
        super(InitBottleNeck, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channel, affine=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channel, affine=True),
            nn.ReLU(),
            nn.Conv2d(channel, 4*channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(4*channel, affine=True),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(channel, 4*channel, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(4*channel, affine=True),
        )

        self.act = nn.ReLU()

    def forward(self, x):
        out = self.main(x)
        out += self.shortcut(x)
        return self.act(out)


class ResNet50(nn.Module):
    def __init__(self, channel):
        super(ResNet50, self).__init__()

        self.channel = channel

        blocks = [InitBottleNeck(channel=self.channel),
                  BottleNeck(channel=self.channel),
                  BottleNeck(channel=self.channel),
                  BottleNeck(channel=self.channel),
                  nn.Dropout2d(0.1),
                  BottleNeckStride(in_channel=self.channel),
                  BottleNeck(channel=2*self.channel),
                  BottleNeck(channel=2*self.channel),
                  BottleNeck(channel=2*self.channel),
                  BottleNeck(channel=2*self.channel),
                  nn.Dropout2d(0.1),
                  BottleNeckStride(in_channel=2*self.channel),
                  BottleNeck(channel=4*self.channel),
                  BottleNeck(channel=4*self.channel),
                  BottleNeck(channel=4*self.channel),
                  BottleNeck(channel=4*self.channel),
                  BottleNeck(channel=4*self.channel),
                  BottleNeck(channel=4*self.channel),
                  nn.Dropout2d(0.1),
                  BottleNeckStride(in_channel=4*self.channel),
                  BottleNeck(channel=8*self.channel),
                  BottleNeck(channel=8*self.channel),
                  BottleNeck(channel=8*self.channel)]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        output = self.blocks(x)
        # print(output.shape)
        return output
