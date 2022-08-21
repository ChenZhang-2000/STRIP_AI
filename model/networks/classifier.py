from torch import nn
from .modules.resnet import ResNet50


def test_func(img):
    print(img.shape)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self._in = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.resnet = ResNet50(64)

    def forward(self, data):
        for images, indexes in data:
            # print(images.shape)
            for index in indexes:
                # print(index.shape)
                image = images[index]
                image = self._in(image)
                image = self.resnet(image)
                test_func(image)
        pass