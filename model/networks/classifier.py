import torch
from torch import nn
from torch.nn import Sigmoid

from .register import register_classifier
from .modules.resnet import ResNetModel
from .modules.attention import NonLocalSelfAttention, MergeBatchSelfAttention, SelfAttentionStage, MHSelfAttention


@register_classifier
class ResNet50(nn.Module):
    name = "ResNet50"

    def __init__(self, img_size, backbone_layers, attention_layers, in_channel=3, backbone=ResNetModel, pretrain=None):
        super().__init__()
        self.backbone = ResNetModel(backbone_layers, in_channel)

        global_feature, grid_feature = self.backbone(torch.zeros(1, in_channel, img_size, img_size))
        global_feature_shape, grid_feature_shape = global_feature.shape, grid_feature.shape

        self.classifier = nn.Sequential(nn.Linear(global_feature_shape[1], global_feature_shape[1] // 2),
                                        nn.Linear(global_feature_shape[1] // 2, 2),
                                        Sigmoid())

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")

    def _forward(self, image):
        assert not image.isnan().any(), "NaN input"
        global_feature, _ = self.backbone(image)
        return global_feature

    def forward(self, data):
        out = []
        for images, indexes in data:
            features = []
            for index in indexes:
                image = images[index]
                feature = self._forward(image)
                assert not feature.isnan().any(), f"NaN feature {torch.isfinite(image).all()}"
                features.append(feature)
            # print(features)
            features = torch.cat(features, dim=0)
            features = torch.sum(features, dim=0)
            # print(features.shape)
            assert not features.isnan().any(), "NaN features"
            preds = self.classifier(features)
            # print(preds.shape)
            out.append(preds)
        out = torch.stack(out, dim=0)
        return out


@register_classifier
class Classifier(nn.Module):
    name = "20220824"

    def __init__(self, img_size, backbone_layers, attention_layers, in_channel=3, backbone=ResNet50, pretrain=None):
        super().__init__()
        if not pretrain:
            self.backbone = backbone(backbone_layers, in_channel)
        else:
            backbone = backbone(img_size, backbone_layers, attention_layers, in_channel, backbone, pretrain)
            backbone.load_state_dict(torch.load(rf"runs\{pretrain}"))
            self.backbone = backbone.backbone
            del backbone

        global_feature, grid_feature = self.backbone(torch.zeros(1, in_channel, img_size, img_size))
        global_feature_shape, grid_feature_shape = global_feature.shape, grid_feature.shape  # N x 2048, n x 1024 x 7 x 7
        # print(global_feature_shape, grid_feature_shape)
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
        global_feature, grid_feature = self.backbone(image)
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
class Classifier(nn.Module):
    name = "20220919"

    def __init__(self, img_size, backbone_layers, attention_layers, in_channel=3, backbone=ResNet50, pretrain=None):
        super().__init__()
        if not pretrain:
            self.backbone = backbone(backbone_layers, in_channel)
        else:
            backbone = backbone(img_size, backbone_layers, attention_layers, in_channel, backbone, pretrain)
            backbone.load_state_dict(torch.load(rf"runs\{pretrain}"))
            self.backbone = backbone.backbone
            del backbone

        self.global_attention = nn.Sequential(MHSelfAttention(2048),
                                              nn.Linear(2048, 1024))

        # self.feature_mlp = nn.Sequential(nn.Linear(2048, 256),
        #                                  nn.BatchNorm2d(1024, affine=False),
        #                                  nn.ReLU(),
        #                                  nn.Linear(256, 32),
        #                                  nn.BatchNorm2d(1024, affine=False),
        #                                  nn.ReLU())

        self.feature_attention = nn.Sequential(SelfAttentionStage(1024, attention_layers[0]),
                                               MergeBatchSelfAttention(1024, 1024, 64, 32))
        self.multi_scan_attention = nn.Sequential(SelfAttentionStage(1024, attention_layers[1]),
                                                  MergeBatchSelfAttention(1024, 2048, 32, 8, keep_dim=True),
                                                  MergeBatchSelfAttention(2048, 4096, 8, 1))

        self.classifier = nn.Sequential(nn.Linear(4096, 2048),
                                        nn.Linear(2048, 2),
                                        Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _forward(self, image):
        n = image.shape[0]
        global_feature, grid_feature = self.backbone(image)  # N x 2048, n x 1024 x 7 x 7
        grid_feature: torch.tensor = grid_feature.flatten(start_dim=2).transpose(1, 2)  # n x 49 x 1024
        global_feature = global_feature.reshape(n, 1, -1).repeat(1, 15, 1)  # n x 15 x 1024
        # print(global_feature.shape)
        global_feature = self.global_attention(global_feature)  # n x 15 x 1024
        feature = torch.cat([global_feature, grid_feature], dim=1)  # n x 64 x 1024

        feature = self.feature_attention(feature)
        return feature

    def forward(self, data):
        out = []
        for images, indexes in data:
            features = []
            for index in indexes:
                image = images[index]
                feature = self._forward(image)  # 32N_i x 1024
                features.append(feature)
            features = torch.cat(features, dim=0).reshape(-1, 32, 1024)  # N x 32 x 1024
            out_feature = self.multi_scan_attention(features)  # N x 4096
            out_feature = torch.sum(out_feature, dim=0)
            out.append(out_feature)
        out = torch.stack(out)  # BS x 4096
        return self.classifier(out)  # BS x 2
