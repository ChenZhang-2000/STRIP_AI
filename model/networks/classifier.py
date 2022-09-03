import torch
from torch import nn
from torch.nn import Sigmoid

from .register import register_classifier
from .modules.resnet import ResNetModel
from .modules.attention import NonLocalSelfAttention, MergeBatchSelfAttention, SelfAttentionStage


def test_func(img):
    print(img.shape)


@register_classifier
class Classifier(nn.Module):
    name = "20220824"

    def __init__(self, img_size, backbone_layers, attention_layers, in_channel=3):
        super().__init__()
        self.resnet = ResNetModel(backbone_layers, in_channel)

        global_feature, grid_feature = self.resnet(torch.zeros(1, in_channel, img_size, img_size))
        global_feature_shape, grid_feature_shape = global_feature.shape, grid_feature.shape
        self.grid_attention = nn.Sequential(NonLocalSelfAttention(*grid_feature_shape[1:]),
                                            nn.BatchNorm2d(grid_feature_shape[1], affine=False),
                                            nn.Conv2d(grid_feature_shape[1], grid_feature_shape[1], 7, 1))

        feature_shape = self.grid_attention(torch.zeros(*grid_feature_shape)).shape[:2][1] + global_feature_shape[1]
        self.feature_attention = nn.Sequential(SelfAttentionStage(feature_shape, attention_layers[0]),
                                               MergeBatchSelfAttention(feature_shape))
        self.multi_scan_attention = nn.Sequential(SelfAttentionStage(feature_shape, attention_layers[1]),
                                                  MergeBatchSelfAttention(feature_shape))

        self.classifier = nn.Sequential(nn.Linear(feature_shape, feature_shape//2),
                                        nn.Linear(feature_shape//2, 2),
                                        Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _forward(self, image):
        global_feature, grid_feature = self.resnet(image)
        grid_feature = self.grid_attention(grid_feature).reshape(*(grid_feature.shape[:2]))
        feature = torch.cat([global_feature, grid_feature], dim=1)
        feature = self.feature_attention(feature)
        return feature

    def forward(self, data):
        out = []
        for images, indexes in data:
            features = []
            for index in indexes:
                image = images[index]
                feature = self._forward(image)
                features.append(feature)
            features = torch.cat(features, dim=0)
            out_feature = self.multi_scan_attention(features)
            out.append(out_feature)
        return self.classifier(torch.cat(out))


@register_classifier
class ResNet50(nn.Module):
    name = "ResNet50"

    def __init__(self, img_size, backbone_layers, attention_layers, in_channel=3):
        super().__init__()
        self.resnet = ResNetModel(backbone_layers, in_channel)

        global_feature, grid_feature = self.resnet(torch.zeros(1, in_channel, img_size, img_size))
        global_feature_shape, grid_feature_shape = global_feature.shape, grid_feature.shape

        self.classifier = nn.Sequential(nn.Linear(global_feature_shape[1], global_feature_shape[1] // 2),
                                        nn.Linear(global_feature_shape[1] // 2, 2),
                                        Sigmoid())

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")

    def _forward(self, image):
        global_feature, _ = self.resnet(image)
        return global_feature

    def forward(self, data):
        out = []
        for images, indexes in data:
            features = []
            for index in indexes:
                image = images[index]
                feature = self._forward(image)
                features.append(feature)
            # print(features)
            features = torch.cat(features, dim=0)
            features = torch.sum(features, dim=0)
            # print(features.shape)
            preds = self.classifier(features)
            # print(preds.shape)
            out.append(preds)
        out = torch.stack(out, dim=0)
        return out
